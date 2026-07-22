// RL Ackermann controller v3 for Nav2 (parallel plugin, does NOT replace v1 or v2).
//
// Model: carrot_l6.onnx  (CnnGruActorCritic family, carrot/lookahead-point config)
//   inputs : obs   [1, 1098]   (raw, no extra normalization)
//            h_in  [1, 1, 256] (GRU hidden state, carried across steps)
//   outputs: action [1, 2]     (raw, UNBOUNDED -> clamp to [-1,1] here)
//            h_out  [1, 1, 256]
//
// Observation layout (1098), exact order per training spec:
//   [0   .. 1079] lidar_stacked   6 frames x 180 rays, newest-first, frames 5 steps apart
//                 [t, t-5, t-10, t-15, t-20, t-25]; ray i at bearing i*2deg CCW from robot +X;
//                 value = clamp(range,0,4)/4 ; no-return/nan/inf -> 1.0
//   [1080..1087] lidar_temporal_sector_diff  8 sectors (45 deg each), [-1,1], -1 = approaching fast
//   [1088..1089] carrot_rel      [fwd/8.0, lat/2.5], carrot in robot body frame
//   [1090..1091] carrot_heading  [sin(theta), cos(theta)], theta = path tangent at carrot - yaw
//   [1092..1093] robot_velocity  [fwd/2.0, yaw_rate/3.0]
//   [1094..1095] previous_actions [throttle, steering] raw [-1,1]
//   [1096]       steering_angle  actual front-wheel angle / 0.60
//   [1097]       steering_rate   front-wheel angular velocity / 10.0
// CNN split boundary inside the model is [0:1080] lidar | [1080:1098] proprio -- this
// controller only has to build the flat 1098 vector in this order, the split happens
// inside the ONNX graph.
//
// Carrot / lookahead point: NOT the discrete resampled waypoints v1 uses. Instead, each
// cycle we find the closest point on Nav2's raw (dense) path to the robot, then walk
// forward along the path by LOOKAHEAD_DIST (2.0 m) accumulating arc length, interpolating
// the exact point and using the local path segment as the tangent -- i.e. Regulated Pure
// Pursuit's lookahead-point computation, just feeding the result into the observation
// instead of RPP's own steering law. The closest-point index only moves forward and only
// advances once the robot is within CARROT_CAPTURE (1.2 m) of it, so it can't jump
// backward or chatter on a self-intersecting path.
//
// Action -> Ackermann (forward-only, same convention as v1 -- this model was NOT trained
// to reverse; all reverse motion here is the same hardcoded e-stop/stuck recovery v1 uses,
// ported unchanged):
//   throttle = clamp(out[0],-1,1) ; steer = clamp(out[1],-1,1)
//   v = max(0, throttle) * speed_scale (m/s, forward only) ; d = steer * 0.60 (rad)
//   Twist: linear.x = v ; angular.z = v * tan(d) / 0.213

#include "rclcpp/rclcpp.hpp"
#include "nav2_core/controller.hpp"
#include "nav2_core/exceptions.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"
#include "nav_msgs/msg/path.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <deque>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace
{
constexpr int N_RAYS = 180;
constexpr int N_FRAMES = 6;
constexpr int FRAME_STRIDE = 5;                 // control steps between stacked frames
constexpr int N_SECTORS = 8;
constexpr int OBS_DIM = 1098;
constexpr int HIDDEN = 256;

constexpr float LIDAR_CAP = 4.0f;               // LIDAR_MAX_DISTANCE (m)
constexpr float CARROT_FWD_NORM = 8.0f;
constexpr float CARROT_LAT_NORM = 2.5f;
constexpr float VEL_LIN_NORM = 2.0f;
constexpr float VEL_YAW_NORM = 3.0f;
constexpr float STEER_ANGLE_NORM = 0.60f;
constexpr float STEER_RATE_NORM = 10.0f;
constexpr float MAX_APPROACH = 1.0f;             // m/s, same temporal-diff scaling as v1
constexpr float TEMPORAL_EMA = 0.8f;
constexpr float STEP_DT = 1.0f / 30.0f;          // 30 Hz control

constexpr double STEER_SCALE = 0.60;             // rad at action = 1
constexpr double WHEELBASE = 0.213;              // m (JETACKER_WHEELBASE)
constexpr double LOOKAHEAD_DIST = 2.0;           // m; carrot distance ahead along the path
constexpr double CARROT_CAPTURE = 1.2;           // m; closest-index only advances within this

double yawFromQuat(const geometry_msgs::msg::Quaternion & q)
{
  return std::atan2(2.0 * (q.w * q.z + q.x * q.y),
                    1.0 - 2.0 * (q.y * q.y + q.z * q.z));
}

template <typename T>
T clampf(T v, T lo, T hi) { return std::max(lo, std::min(hi, v)); }

struct Vec2 { double x, y; };
double norm2(const Vec2 & a) { return std::hypot(a.x, a.y); }
}  // namespace

class RLControllerV3 : public nav2_core::Controller
{
public:
  RLControllerV3() = default;

  void configure(
    const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
    std::string name,
    std::shared_ptr<tf2_ros::Buffer>,
    std::shared_ptr<nav2_costmap_2d::Costmap2DROS>) override
  {
    node_ = parent.lock();
    logger_ = node_->get_logger();
    plugin_name_ = name;

    declareParam("model_path", "/home/ubuntu/ros2_ws/src/rl_nav_cpp/carrot_l6.onnx");
    declareParam("goal_change_thresh", 0.5);     // m endpoint move that resets the GRU (new goal vs replan)
    declareParam("lidar_angle_offset", 3.14159265);  // rad; this lidar is mounted yawed 180 deg
    declareParam("lidar_ccw", true);
    // DEPLOY SPEED: spec's target is a fixed 0.60 m/s ("no governor"), but that's untested
    // on real hardware with THIS model. Start low and ramp up during field testing, same
    // as v2's rollout. Also note navigation/config/nav2_params.yaml velocity_smoother
    // currently caps linear.x at 0.26 regardless of this param.
    declareParam("speed_scale", 0.25);
    // empirical knob: if the robot consistently steers the WRONG way on a straight path,
    // flip this to -1.0 (no rebuild needed) rather than editing code. This can happen if
    // the model's own steer-sign training convention doesn't match this decode.
    declareParam("steer_sign", 1.0);
    declareParam("estop_distance", 0.30);
    declareParam("estop_release", 0.40);
    declareParam("estop_half_width", 0.17);
    declareParam("estop_fail_sec", 1.5);
    declareParam("reverse_speed", 0.10);
    declareParam("reverse_dist", 0.80);
    declareParam("reverse_clear", 0.55);
    declareParam("reverse_rear_stop", 0.22);
    declareParam("debug", true);

    std::string model_path = node_->get_parameter(prefix("model_path")).as_string();

    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "rl_v3");
    session_options_ = std::make_unique<Ort::SessionOptions>();
    session_options_->SetIntraOpNumThreads(1);
    session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), *session_options_);

    h_.assign(HIDDEN, 0.0f);
    temporal_ema_.assign(N_SECTORS, 0.0f);

    // /scan_raw, not /scan -- see rl_controller.cpp v1 header for why (filter chain keeps
    // the physically-blind rear sector and drops the valid front on this yaw-180 mount).
    scan_sub_ = node_->create_subscription<sensor_msgs::msg::LaserScan>(
      "/scan_raw", rclcpp::SensorDataQoS(),
      [this](sensor_msgs::msg::LaserScan::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(scan_mtx_);
        last_scan_ = msg;
        last_scan_time_ = node_->now();
      });

    RCLCPP_INFO(logger_,
      "RL Ackermann controller v3 configured (1098-dim carrot obs, GRU, forward-only). Model: %s",
      model_path.c_str());
  }

  void cleanup() override {}
  void activate() override { resetEpisode(); }
  void deactivate() override {}
  void setSpeedLimit(const double &, const bool &) override {}

  void setPlan(const nav_msgs::msg::Path & path) override
  {
    path_.clear();
    for (const auto & p : path.poses) path_.push_back({p.pose.position.x, p.pose.position.y});
    if (path_.empty()) return;

    Vec2 new_goal = path_.back();
    bool new_goal_started =
      !have_goal_ || norm2({new_goal.x - last_goal_.x, new_goal.y - last_goal_.y})
                       > node_->get_parameter(prefix("goal_change_thresh")).as_double();

    if (new_goal_started) {
      resetEpisode();
      closest_idx_ = 0;
    } else {
      closest_idx_ = nearestIndex(last_robot_, 0);   // re-localize on replan, full search
    }
    last_goal_ = new_goal;
    have_goal_ = true;
  }

  geometry_msgs::msg::TwistStamped computeVelocityCommands(
    const geometry_msgs::msg::PoseStamped & pose,
    const geometry_msgs::msg::Twist & velocity,
    nav2_core::GoalChecker *) override
  {
    geometry_msgs::msg::TwistStamped cmd;
    cmd.header.frame_id = "base_link";
    cmd.header.stamp = node_->now();
    if (path_.empty()) return cmd;

    Vec2 robot{pose.pose.position.x, pose.pose.position.y};
    double ryaw = yawFromQuat(pose.pose.orientation);
    last_robot_ = robot;

    rclcpp::Time now = node_->now();
    double dt = have_last_time_ ? (now - last_time_).seconds() : STEP_DT;
    if (dt <= 1e-4 || dt > 1.0) dt = STEP_DT;
    last_time_ = now;
    have_last_time_ = true;

    // --- sensor-outage guard (lidar intermittently stalls on this robot) ---
    double scan_age;
    {
      std::lock_guard<std::mutex> lock(scan_mtx_);
      scan_age = last_scan_ ? (now - last_scan_time_).seconds() : 1e9;
    }
    if (scan_age > 0.5) {
      estop_timing_ = false;
      nomove_timing_ = false;
      if (reversing_) {
        reversing_ = false;
        RCLCPP_WARN(logger_, "scan went stale mid-reverse -> aborting reverse");
      }
      RCLCPP_WARN_THROTTLE(logger_, *node_->get_clock(), 2000,
        "scan is %.1f s old (lidar outage?) - holding position", scan_age);
      prev_action_[0] = 0.0f;
      prev_action_[1] = 0.0f;
      steer_prev_ = steer_last_;
      steer_last_ = 0.0;
      return cmd;
    }

    // --- lidar frame -> ring buffer (one frame per control step, 30 Hz) ---
    pushFrame(sampleScan());

    std::array<float, OBS_DIM> obs{};
    int o = 0;

    // [0..1079] stacked frames, newest-first: offsets 0,5,10,15,20,25
    for (int f = 0; f < N_FRAMES; ++f) {
      const std::vector<float> & fr = frameAtOffset(f * FRAME_STRIDE);
      for (int k = 0; k < N_RAYS; ++k) obs[o++] = fr[k];
    }

    // [1080..1087] temporal sector diff
    std::array<float, N_SECTORS> temporal = computeTemporal();
    for (int s = 0; s < N_SECTORS; ++s) obs[o++] = temporal[s];

    // carrot: closest point on path (monotonic forward, captures within CARROT_CAPTURE),
    // then walk forward LOOKAHEAD_DIST along the path for the actual carrot + tangent
    advanceClosest(robot);
    Vec2 carrot; double tangent;
    computeCarrot(robot, carrot, tangent);

    double cyaw = std::cos(ryaw), syaw = std::sin(ryaw);
    Vec2 to_carrot{carrot.x - robot.x, carrot.y - robot.y};
    double bx = to_carrot.x * cyaw + to_carrot.y * syaw;      // world -> body: R(-yaw)
    double by = -to_carrot.x * syaw + to_carrot.y * cyaw;
    double theta = std::atan2(std::sin(tangent - ryaw), std::cos(tangent - ryaw));

    // [1088..1089] carrot_rel
    obs[o++] = clampf(static_cast<float>(bx / CARROT_FWD_NORM), -1.0f, 1.0f);
    obs[o++] = clampf(static_cast<float>(by / CARROT_LAT_NORM), -1.0f, 1.0f);
    // [1090..1091] carrot_heading -- ORDER is [sin, cos]
    obs[o++] = static_cast<float>(std::sin(theta));
    obs[o++] = static_cast<float>(std::cos(theta));

    // [1092..1093] robot_velocity
    obs[o++] = clampf(static_cast<float>(velocity.linear.x) / VEL_LIN_NORM, -1.0f, 1.0f);
    obs[o++] = clampf(static_cast<float>(velocity.angular.z) / VEL_YAW_NORM, -1.0f, 1.0f);

    // [1094..1095] previous actions (raw clamped)
    obs[o++] = prev_action_[0];
    obs[o++] = prev_action_[1];

    // [1096] steering_angle, [1097] steering_rate (reconstructed from last command; no
    // verified servo feedback topic on this robot yet -- same proxy v1/v2 use)
    obs[o++] = clampf(static_cast<float>(steer_last_ / STEER_ANGLE_NORM), -1.0f, 1.0f);
    float steer_rate = static_cast<float>((steer_last_ - steer_prev_) / dt);
    obs[o++] = clampf(steer_rate / STEER_RATE_NORM, -1.0f, 1.0f);

    // --- inference: obs + h_in -> action + h_out ---
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::array<int64_t, 2> obs_shape{1, OBS_DIM};
    std::array<int64_t, 3> h_shape{1, 1, HIDDEN};
    std::array<Ort::Value, 2> inputs{
      Ort::Value::CreateTensor<float>(mem, obs.data(), obs.size(), obs_shape.data(), obs_shape.size()),
      Ort::Value::CreateTensor<float>(mem, h_.data(), h_.size(), h_shape.data(), h_shape.size())};

    const char * in_names[] = {"obs", "h_in"};
    const char * out_names[] = {"action", "h_out"};
    auto outputs = session_->Run(
      Ort::RunOptions{nullptr}, in_names, inputs.data(), 2, out_names, 2);

    const float * action = outputs[0].GetTensorData<float>();
    const float * h_out = outputs[1].GetTensorData<float>();
    std::copy(h_out, h_out + HIDDEN, h_.begin());

    float throttle = clampf(action[0], -1.0f, 1.0f);
    float steer = clampf(action[1], -1.0f, 1.0f);

    double v = std::max(0.0f, throttle) * node_->get_parameter(prefix("speed_scale")).as_double();
    double steer_sign = node_->get_parameter(prefix("steer_sign")).as_double();
    double d = steer_sign * static_cast<double>(steer) * STEER_SCALE;

    // --- stuck handling: lidar e-stop + hardcoded steered-reverse recovery (ported from
    // v1 unchanged -- this model, like v1, was never trained to reverse itself) ---
    bool est = checkEmergencyStop();
    double fail_after = node_->get_parameter(prefix("estop_fail_sec")).as_double();

    if (!reversing_ && fail_after > 0.0) {
      if (est) {
        if (!estop_timing_) { estop_timing_ = true; estop_since_ = now; }
      } else {
        estop_timing_ = false;
      }
      bool moving = std::abs(velocity.linear.x) > 0.02 || std::abs(velocity.angular.z) > 0.05;
      if (moving) failed_reverses_ = 0;
      if (v > 0.06 && !moving) {
        if (!nomove_timing_) { nomove_timing_ = true; nomove_since_ = now; }
      } else {
        nomove_timing_ = false;
      }

      bool est_stuck = estop_timing_ && (now - estop_since_).seconds() > fail_after;
      bool push_stuck = nomove_timing_ && (now - nomove_since_).seconds() > fail_after;
      if (est_stuck || push_stuck) {
        reversing_ = true;
        reverse_start_ = robot;
        reverse_begin_ = now;
        RCLCPP_WARN(logger_, "STUCK (%s) -> reversing out",
                    est_stuck ? "wall in forward arc" : "pushing but not moving");
      }
    }

    if (reversing_) {
      double rev_speed = node_->get_parameter(prefix("reverse_speed")).as_double();
      double rev_dist = node_->get_parameter(prefix("reverse_dist")).as_double();
      double rear_stop = node_->get_parameter(prefix("reverse_rear_stop")).as_double();
      float rear_min = rearArcMin();

      double travelled = norm2({robot.x - reverse_start_.x, robot.y - reverse_start_.y});
      double elapsed = (now - reverse_begin_).seconds();
      double clear_x = node_->get_parameter(prefix("reverse_clear")).as_double();
      double half_w = node_->get_parameter(prefix("estop_half_width")).as_double();
      bool clear_to_turn = travelled >= 0.15 && frontCorridorMin(half_w) > clear_x;
      bool done = clear_to_turn || travelled >= rev_dist;
      bool rear_blocked = rear_min < rear_stop;
      bool rev_stalled = elapsed > 0.8 &&
                         std::abs(velocity.linear.x) < 0.02 && travelled < 0.05;
      bool timed_out = elapsed > 3.0 * rev_dist / std::max(rev_speed, 0.01) || rev_stalled;

      if (done || rear_blocked || timed_out) {
        reversing_ = false;
        estop_timing_ = false;
        nomove_timing_ = false;
        estopped_ = false;
        if (travelled < 0.10) {
          double release_d = node_->get_parameter(prefix("estop_release")).as_double();
          bool front_open = frontCorridorMin(half_w) > release_d;
          ++failed_reverses_;
          if (!front_open || failed_reverses_ >= 3) {
            failed_reverses_ = 0;
            throw nav2_core::PlannerException(
              "RL controller v3: boxed in (reverse " +
              std::string(rear_blocked ? "rear-blocked" : "timed out") + ", moved " +
              std::to_string(travelled) + " m)");
          }
        } else {
          failed_reverses_ = 0;
          RCLCPP_INFO(logger_, "reverse recovery done (%.2f m) -> resuming policy", travelled);
        }
        v = 0.0; d = 0.0; throttle = 0.0f; steer = 0.0f;
      } else {
        v = -rev_speed;
        d = (theta > 0.0 ? -1.0 : 1.0) * 0.40;   // steer nose toward the carrot while backing
        throttle = 0.0f; steer = 0.0f;
      }
    } else if (est) {
      v = 0.0; d = 0.0; throttle = 0.0f; steer = 0.0f;
    }

    cmd.twist.linear.x = v;
    cmd.twist.angular.z = (std::abs(v) > 1e-6) ? v * std::tan(d) / WHEELBASE : 0.0;

    prev_action_[0] = throttle;
    prev_action_[1] = steer;
    steer_prev_ = steer_last_;
    steer_last_ = d;

    if (node_->get_parameter(prefix("debug")).as_bool()) {
      RCLCPP_INFO(logger_,
        "idx %zu/%zu act[%.2f %.2f] carrot[fwd=%.2f lat=%.2f th=%.2f] -> v=%.3f d=%.3f w=%.3f",
        closest_idx_, path_.size(), throttle, steer, bx, by, theta,
        v, d, cmd.twist.angular.z);
    }
    return cmd;
  }

private:
  std::string prefix(const std::string & p) const { return plugin_name_ + "." + p; }
  template <typename T>
  void declareParam(const std::string & p, const T & def)
  {
    if (!node_->has_parameter(prefix(p))) node_->declare_parameter(prefix(p), def);
  }

  void resetEpisode()
  {
    h_.assign(HIDDEN, 0.0f);
    prev_action_[0] = prev_action_[1] = 0.0f;
    steer_last_ = steer_prev_ = 0.0;
    temporal_ema_.assign(N_SECTORS, 0.0f);
    ring_.clear();
    estopped_ = false;
    estop_timing_ = false;
    nomove_timing_ = false;
    reversing_ = false;
    failed_reverses_ = 0;
  }

  float frontCorridorMin(double half_width) const
  {
    const std::vector<float> & fr = frameAtOffset(0);
    float best = LIDAR_CAP;
    for (int i = 0; i < N_RAYS; ++i) {
      double b = 2.0 * M_PI * i / N_RAYS;
      double cb = std::cos(b);
      if (cb <= 0.0) continue;
      float r = fr[i] * LIDAR_CAP;
      if (r < 0.05f) continue;
      if (std::abs(r * std::sin(b)) <= half_width) {
        best = std::min(best, static_cast<float>(r * cb));
      }
    }
    return best;
  }

  bool checkEmergencyStop()
  {
    double stop_d = node_->get_parameter(prefix("estop_distance")).as_double();
    if (stop_d <= 0.0 || ring_.empty()) return false;

    {
      std::lock_guard<std::mutex> lock(scan_mtx_);
      if (!last_scan_ || (node_->now() - last_scan_time_).seconds() > 1.0) {
        if (!estopped_) RCLCPP_WARN(logger_, "E-STOP: laser scan stale or missing");
        estopped_ = true;
        return true;
      }
    }

    double release_d = std::max(node_->get_parameter(prefix("estop_release")).as_double(), stop_d);
    float min_x = frontCorridorMin(node_->get_parameter(prefix("estop_half_width")).as_double());

    bool was = estopped_;
    if (estopped_) estopped_ = (min_x < release_d);
    else           estopped_ = (min_x < stop_d);
    if (estopped_ && !was) {
      RCLCPP_WARN(logger_, "E-STOP: blocker %.2f m ahead in the driving corridor", min_x);
    } else if (!estopped_ && was) {
      RCLCPP_INFO(logger_, "E-STOP released (corridor clear, %.2f m ahead)", min_x);
    }
    return estopped_;
  }

  // rear +/-54 deg arc (inner +/-36 deg is physically blind, see v1 for the measurement)
  float rearArcMin() const
  {
    const std::vector<float> & fr = frameAtOffset(0);
    float rear_min = LIDAR_CAP;
    for (int i = 63; i <= 117; ++i) {
      float r = fr[i] * LIDAR_CAP;
      if (r >= 0.05f) rear_min = std::min(rear_min, r);
    }
    return rear_min;
  }

  // ---- lidar (identical to v1: 180 rays at training bearings, i*2deg CCW from +X) ----
  std::vector<float> sampleScan()
  {
    std::vector<float> out(N_RAYS, 1.0f);
    sensor_msgs::msg::LaserScan::SharedPtr s;
    {
      std::lock_guard<std::mutex> lock(scan_mtx_);
      s = last_scan_;
    }
    if (!s || s->ranges.empty() || s->angle_increment == 0.0f) return out;

    double offset = node_->get_parameter(prefix("lidar_angle_offset")).as_double();
    double dir = node_->get_parameter(prefix("lidar_ccw")).as_bool() ? 1.0 : -1.0;
    int n = static_cast<int>(s->ranges.size());

    for (int i = 0; i < N_RAYS; ++i) {
      double ang = offset + dir * (2.0 * M_PI * i / N_RAYS);
      ang = std::atan2(std::sin(ang), std::cos(ang));
      int idx = static_cast<int>(std::lround((ang - s->angle_min) / s->angle_increment));
      idx = ((idx % n) + n) % n;
      float r = s->ranges[idx];
      out[i] = (!std::isfinite(r) || r >= LIDAR_CAP) ? 1.0f
               : clampf(r / LIDAR_CAP, 0.0f, 1.0f);
    }
    return out;
  }

  void pushFrame(const std::vector<float> & frame)
  {
    ring_.push_front(frame);
    const size_t need = static_cast<size_t>((N_FRAMES - 1) * FRAME_STRIDE + 1);
    while (ring_.size() > need) ring_.pop_back();
  }

  const std::vector<float> & frameAtOffset(int off) const
  {
    static const std::vector<float> kFar(N_RAYS, 1.0f);
    if (ring_.empty()) return kFar;
    size_t idx = std::min(static_cast<size_t>(off), ring_.size() - 1);
    return ring_[idx];
  }

  std::array<float, N_SECTORS> computeTemporal()
  {
    std::array<float, N_SECTORS> result{};
    const std::vector<float> & cur = frameAtOffset(0);
    const std::vector<float> & old = frameAtOffset(FRAME_STRIDE);

    const float max_diff = MAX_APPROACH * (FRAME_STRIDE * STEP_DT) / LIDAR_CAP;
    const float scale = 0.05f / std::max(max_diff, 1e-9f);

    int base = N_RAYS / N_SECTORS;
    int rem = N_RAYS % N_SECTORS;
    int start = 0;
    for (int s = 0; s < N_SECTORS; ++s) {
      int len = base + (s < rem ? 1 : 0);
      float mn = std::numeric_limits<float>::infinity();
      for (int k = start; k < start + len; ++k) mn = std::min(mn, cur[k] - old[k]);
      float scaled = mn * scale;
      temporal_ema_[s] = TEMPORAL_EMA * temporal_ema_[s] + (1.0f - TEMPORAL_EMA) * scaled;
      result[s] = clampf(temporal_ema_[s] / 0.05f, -1.0f, 1.0f);
      start += len;
    }
    return result;
  }

  // ---- carrot / lookahead point (Regulated-Pure-Pursuit style) ----
  size_t nearestIndex(const Vec2 & p, size_t from) const
  {
    size_t best = from;
    double bd = std::numeric_limits<double>::max();
    for (size_t i = from; i < path_.size(); ++i) {
      double d = norm2({path_[i].x - p.x, path_[i].y - p.y});
      if (d < bd) { bd = d; best = i; }
    }
    return best;
  }

  // closest-index only moves forward, and only "captures" (accepts) a further point once
  // the robot is within CARROT_CAPTURE of it -- prevents jumping backward or ahead on a
  // self-intersecting path from scan noise.
  void advanceClosest(const Vec2 & robot)
  {
    if (path_.empty()) return;
    size_t n = path_.size();
    size_t i = std::min(closest_idx_, n - 1);
    double dist = norm2({robot.x - path_[i].x, robot.y - path_[i].y});
    while (i + 1 < n) {
      double dist_next = norm2({robot.x - path_[i + 1].x, robot.y - path_[i + 1].y});
      bool captured = dist < CARROT_CAPTURE;
      bool passed = dist_next < dist;
      if (!(captured || passed)) break;
      ++i;
      dist = dist_next;
    }
    closest_idx_ = i;
  }

  // walk forward from closest_idx_ accumulating arc length until LOOKAHEAD_DIST; interpolate
  // the carrot point on the segment where it's crossed, tangent = that segment's direction.
  // If the remaining path is shorter than the lookahead, carrot = path endpoint.
  void computeCarrot(const Vec2 & robot, Vec2 & carrot, double & tangent) const
  {
    size_t n = path_.size();
    if (n == 0) { carrot = robot; tangent = 0.0; return; }
    if (n == 1) { carrot = path_[0]; tangent = 0.0; return; }

    size_t i = std::min(closest_idx_, n - 1);
    double acc = norm2({robot.x - path_[i].x, robot.y - path_[i].y});
    if (acc >= LOOKAHEAD_DIST || i + 1 >= n) {
      // already at/past lookahead from the very first segment, or on the last point
      size_t j = std::min(i + 1, n - 1);
      Vec2 seg{path_[j].x - path_[i].x, path_[j].y - path_[i].y};
      tangent = (norm2(seg) > 1e-6) ? std::atan2(seg.y, seg.x)
                                    : std::atan2(robot.y - path_[i].y, robot.x - path_[i].x);
      carrot = path_[j];
      return;
    }
    for (size_t k = i; k + 1 < n; ++k) {
      double seg_len = norm2({path_[k + 1].x - path_[k].x, path_[k + 1].y - path_[k].y});
      if (acc + seg_len >= LOOKAHEAD_DIST || k + 2 == n) {
        double remain = LOOKAHEAD_DIST - acc;
        double t = (seg_len > 1e-6) ? clampf(remain / seg_len, 0.0, 1.0) : 1.0;
        carrot.x = path_[k].x + t * (path_[k + 1].x - path_[k].x);
        carrot.y = path_[k].y + t * (path_[k + 1].y - path_[k].y);
        tangent = std::atan2(path_[k + 1].y - path_[k].y, path_[k + 1].x - path_[k].x);
        return;
      }
      acc += seg_len;
    }
    // fell through (shouldn't happen given the k+2==n check above) -- fall back to endpoint
    carrot = path_.back();
    tangent = std::atan2(path_.back().y - path_[n - 2].y, path_.back().x - path_[n - 2].x);
  }

  rclcpp::Logger logger_{rclcpp::get_logger("rl_controller_v3")};
  rclcpp_lifecycle::LifecycleNode::SharedPtr node_;
  std::string plugin_name_;

  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
  sensor_msgs::msg::LaserScan::SharedPtr last_scan_;
  std::mutex scan_mtx_;
  rclcpp::Time last_scan_time_;

  std::vector<Vec2> path_;
  size_t closest_idx_{0};
  Vec2 last_goal_{0, 0};
  Vec2 last_robot_{0, 0};
  bool have_goal_{false};

  std::deque<std::vector<float>> ring_;
  std::vector<float> temporal_ema_;

  float prev_action_[2]{0.0f, 0.0f};
  double steer_last_{0.0}, steer_prev_{0.0};
  bool estopped_{false};
  bool estop_timing_{false};
  rclcpp::Time estop_since_;
  bool nomove_timing_{false};
  rclcpp::Time nomove_since_;
  bool reversing_{false};
  Vec2 reverse_start_{0, 0};
  rclcpp::Time reverse_begin_;
  int failed_reverses_{0};
  rclcpp::Time last_time_;
  bool have_last_time_{false};

  std::vector<float> h_;

  std::unique_ptr<Ort::Env> env_;
  std::unique_ptr<Ort::Session> session_;
  std::unique_ptr<Ort::SessionOptions> session_options_;
};

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(RLControllerV3, nav2_core::Controller)
