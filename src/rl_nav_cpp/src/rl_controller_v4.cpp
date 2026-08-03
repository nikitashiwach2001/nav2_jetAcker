// RL Ackermann controller v4 for Nav2 (parallel plugin, does NOT replace v1/v2/v3).
//
// Model: policy_v4_model6998.onnx -- exported from model_6998.pt (a new checkpoint the
// user provided). model_6998.pt's actor-path weight shapes are IDENTICAL to
// model_189933.pt (the checkpoint v3's carrot_l6.onnx was exported from): same CNN lidar
// encoder (conv 6x180 -> proj 2880->256), same 18-dim proprio branch, same 320-dim GRU
// input / 256 hidden. It does NOT match v1's 1102-dim/22-proprio observation layout, so
// this file reuses v3's obs-building/carrot-lookahead code (proven correct for this exact
// network shape) rather than v1's. See rl_nav_cpp/scripts/export_model_6998_onnx.py for
// the export (architecture reconstructed by inspecting carrot_l6.onnx's own ONNX graph
// node-for-node, then verified to match the reconstructed PyTorch module output to ~1e-6).
//
// nav2-side behavior (docking, reverse-recovery escalation) is ported from v1's
// field-tested version, NOT v3's original simpler recovery -- only the observation/
// inference path had to change for this model; everything else matches v1:
//   - terminal docking near the goal (dock_dist/dock_speed/dock_kp)
//   - reverse direction locked once per attempt + steer angle escalates with each
//     failed attempt (reverse_dir_, failed_reverses_)
//   - minimum reverse distance escalates with each failed attempt (3-point-turn logic)
//   - failed_reverses_ only clears after RESUME_SETTLE_SEC of sustained (not momentary)
//     forward motion
//   - reverse_stall_sec-gated rear-contact stall detection (default 0.5 s)
// The steer_ema_alpha filter is v4's own (not ported from v1's alpha/rate-cap combo).
//
// Model: policy_v4_model6998.onnx  (CnnGruActorCritic family, carrot/lookahead-point config)
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
// forward along the path by carrot_lookahead_dist (param, default 1.0 m -- UNVERIFIED
// against this model's actual training spec, tune live if steering feels systematically
// wrong) accumulating arc length, interpolating the exact point and using the local path
// segment as the tangent -- i.e. Regulated Pure Pursuit's lookahead-point computation,
// just feeding the result into the observation instead of RPP's own steering law. The
// closest-point index only moves forward and only advances once the robot is within
// CARROT_CAPTURE (1.2 m) of it, so it can't jump backward or chatter on a self-intersecting
// path.
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
constexpr double CARROT_CAPTURE = 1.2;           // m; closest-index only advances within this
constexpr double RESUME_SETTLE_SEC = 5.0;        // sustained (not momentary) motion needed
                                                  // to clear the boxed-in failure streak

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

class RLControllerV4 : public nav2_core::Controller
{
public:
  RLControllerV4() = default;

  void configure(
    const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
    std::string name,
    std::shared_ptr<tf2_ros::Buffer>,
    std::shared_ptr<nav2_costmap_2d::Costmap2DROS>) override
  {
    node_ = parent.lock();
    logger_ = node_->get_logger();
    plugin_name_ = name;

    declareParam("model_path", "/home/ubuntu/ros2_ws/src/rl_nav_cpp/policies/policy_v4_model6998.onnx");
    declareParam("goal_change_thresh", 0.5);     // m endpoint move that resets the GRU (new goal vs replan)
    // carrot/lookahead distance along the path (was hardcoded to 2.0 m). This MUST match
    // whatever the training/sim environment used -- unverified against this model's actual
    // training spec (we only confirmed the ONNX graph's computation, not the semantic
    // meaning/scale of the carrot inputs). Tune this live to test the mismatch theory:
    //   ros2 param set /controller_server FollowPath.carrot_lookahead_dist 1.0
    declareParam("carrot_lookahead_dist", 1.0);
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
    // contact detector: rear arc is physically blind, so touching an obstacle behind is
    // only visible as "commanding reverse but not moving" -- ported from v1.
    declareParam("reverse_stall_sec", 0.5);
    // terminal docking (ported from v1): within dock_dist of the path end, hand off from
    // the policy to a plain pure-pursuit approach + back-and-fill, since the policy never
    // saw the final close-in approach during training and an Ackermann base can otherwise
    // orbit the goal on its turning circle forever.
    declareParam("dock_dist", 1.0);
    declareParam("dock_speed", 0.12);
    declareParam("dock_kp", 1.5);
    // EMA on the final commanded steering angle (after policy AND any recovery/estop
    // override) to kill the ~3.6 Hz weave jitter seen on this model while passing real
    // steering intent (<1 Hz turns). alpha = weight on the PREVIOUS command: 0.7 -> ~1.7 Hz
    // cutoff. Raise toward 0.8 for smoother/more lag, lower toward 0.6 for more responsive.
    declareParam("steer_ema_alpha", 0.7);
    declareParam("debug", true);

    std::string model_path = node_->get_parameter(prefix("model_path")).as_string();

    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "rl_v4");
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
      "RL Ackermann controller v4 configured (1098-dim carrot obs, GRU, forward-only, model_6998). Model: %s",
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
    double lookahead_dist = node_->get_parameter(prefix("carrot_lookahead_dist")).as_double();
    computeCarrot(robot, lookahead_dist, carrot, tangent);

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

    // --- terminal docking (ported from v1): the policy never saw the final close-in
    // approach during training, and an Ackermann base can otherwise orbit the goal on its
    // turning circle forever. Within dock_dist of the path endpoint, hand off to a plain
    // pure-pursuit approach + back-and-fill. v1 also required being near the end of its
    // discrete waypoint list; there's no such index here, so this uses the raw path
    // endpoint distance alone (the carrot computation already saturates at the endpoint
    // once close, so this doesn't fight the carrot).
    double dock_dist = node_->get_parameter(prefix("dock_dist")).as_double();
    Vec2 fin = path_.back();
    double fin_dist = norm2({fin.x - robot.x, fin.y - robot.y});
    bool docking = dock_dist > 0.0 && fin_dist < dock_dist;
    if (docking) {
      double fb = std::atan2(fin.y - robot.y, fin.x - robot.x);
      double e = std::atan2(std::sin(fb - ryaw), std::cos(fb - ryaw));
      // back-and-fill only when the goal is genuinely behind (min turn radius is
      // wheelbase/tan(0.6) ~= 0.31 m, so forward steering handles large offsets)
      if (dock_reversing_) {
        if (std::abs(e) < 0.4) dock_reversing_ = false;
      } else if (std::abs(e) > 1.6 && fin_dist > 0.35) {
        dock_reversing_ = true;
      }
      if (fin_dist <= 0.35) dock_reversing_ = false;
      // never back-and-fill into something behind us
      if (dock_reversing_ &&
          rearArcMin() < node_->get_parameter(prefix("reverse_rear_stop")).as_double()) {
        dock_reversing_ = false;
      }
      double dock_speed = node_->get_parameter(prefix("dock_speed")).as_double();
      if (dock_reversing_) {
        v = -dock_speed;                       // reversing: yaw rate = v*tan(d)/L flips sign,
        d = (e > 0.0 ? -1.0 : 1.0) * 0.45;     // so opposite steer swings the nose toward the goal
      } else {
        v = std::min(dock_speed, 0.3 * fin_dist + 0.06);  // ease in as the goal closes
        d = clampf(node_->get_parameter(prefix("dock_kp")).as_double() * e,
                   -STEER_SCALE, STEER_SCALE);
      }
      throttle = 0.0f;
      steer = 0.0f;
      if (!was_docking_) {
        RCLCPP_INFO(logger_, "docking: %.2f m to goal, heading err %.2f rad", fin_dist, e);
      }
    } else {
      dock_reversing_ = false;
    }
    was_docking_ = docking;

    // --- stuck handling: lidar e-stop + hardcoded steered-reverse recovery (escalation
    // logic ported from v1's field-tested version -- this model, like v1, was never
    // trained to reverse itself) ---
    bool est = checkEmergencyStop();
    double fail_after = node_->get_parameter(prefix("estop_fail_sec")).as_double();

    if (!reversing_ && fail_after > 0.0) {
      if (est) {
        if (!estop_timing_) { estop_timing_ = true; estop_since_ = now; }
      } else {
        estop_timing_ = false;
      }
      bool moving = std::abs(velocity.linear.x) > 0.02 || std::abs(velocity.angular.z) > 0.05;
      // clear the failure streak only after SUSTAINED progress, not a momentary blip (a
      // robot boxed in front+rear moves a real ~0.2-0.3 m on every reverse then drives
      // straight back into the same wall a couple seconds later)
      if (moving) {
        if (!have_settled_since_) { settled_since_ = now; have_settled_since_ = true; }
        if (failed_reverses_ > 0 && (now - settled_since_).seconds() > RESUME_SETTLE_SEC) {
          failed_reverses_ = 0;
        }
      } else {
        have_settled_since_ = false;
      }
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
        // decide the turn direction ONCE and hold it for the whole maneuver -- theta
        // recomputed fresh every frame can flip sign near zero while the robot's own
        // heading changes mid-reverse, which would zig-zag steering within one attempt
        reverse_dir_ = (theta > 0.0) ? -1.0 : 1.0;
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
      // minimum distance before the "front clear, safe to turn" exit escalates with each
      // failed attempt: the reverse steers at a fixed nonzero angle, so more distance along
      // that arc means more accumulated heading change (same principle as a 3-point car
      // park) -- each retry ends up genuinely more rotated, not just re-trying the same
      // geometry. rear_blocked/rev_stalled (contact detection) below are fully independent
      // and still cut the reverse short immediately regardless of this target.
      double min_travel = std::min(rev_dist, 0.15 + failed_reverses_ * 0.15);
      double clear_x = node_->get_parameter(prefix("reverse_clear")).as_double();
      double half_w = node_->get_parameter(prefix("estop_half_width")).as_double();
      bool clear_to_turn = travelled >= min_travel && frontCorridorMin(half_w) > clear_x;
      bool done = clear_to_turn || travelled >= rev_dist;
      bool rear_blocked = rear_min < rear_stop;
      double stall_sec = node_->get_parameter(prefix("reverse_stall_sec")).as_double();
      bool rev_stalled = elapsed > stall_sec &&
                         std::abs(velocity.linear.x) < 0.02 && travelled < 0.05;
      if (rev_stalled) {
        RCLCPP_WARN(logger_, "reverse produced no motion for %.1f s -> rear contact "
                    "likely, stopping reverse", elapsed);
      }
      bool timed_out = elapsed > 3.0 * rev_dist / std::max(rev_speed, 0.01) || rev_stalled;

      if (done || rear_blocked || timed_out) {
        reversing_ = false;
        estop_timing_ = false;
        nomove_timing_ = false;
        estopped_ = false;
        double release_d = node_->get_parameter(prefix("estop_release")).as_double();
        bool front_open = frontCorridorMin(half_w) > release_d;
        // count EVERY completed reverse attempt against the streak, whether it stalled or
        // nominally "succeeded" by distance -- only sustained forward progress (checked
        // above) clears this, distance alone can't be the only signal
        ++failed_reverses_;
        if (failed_reverses_ >= 3 || (travelled < 0.10 && !front_open)) {
          int streak = failed_reverses_;
          failed_reverses_ = 0;
          throw nav2_core::PlannerException(
            "RL controller v4: boxed in (reverse " +
            std::string(rear_blocked ? "rear-blocked" : "timed out") + ", moved " +
            std::to_string(travelled) + " m; front " +
            (front_open ? "open but oscillating" : "blocked") + "; streak " +
            std::to_string(streak) + ")");
        }
        if (travelled < 0.10) {
          RCLCPP_WARN(logger_,
            "reverse blocked (rear %.2f m) but front corridor open -> resuming policy forward (%d/3)",
            rear_min, failed_reverses_);
        } else {
          // resume WARM: same GRU state, same carrot tracking -- resetting here made the
          // policy reorient with wide arcs after every recovery
          RCLCPP_INFO(logger_,
            "reverse recovery done (%.2f m, rear min %.2f m) -> resuming policy (%d/3 if it re-sticks fast)",
            travelled, rear_min, failed_reverses_);
        }
        v = 0.0; d = 0.0; throttle = 0.0f; steer = 0.0f;
      } else {
        v = -rev_speed;
        // 3-point-turn reverse: steer angle escalates with each consecutive failed attempt
        // (reset only via the sustained-motion gate above) so every retry turns sharper
        // than the last; direction uses reverse_dir_ locked in when this reverse started.
        double reverse_steer = std::min(STEER_SCALE, 0.40 + failed_reverses_ * 0.10);
        d = reverse_dir_ * reverse_steer;
        throttle = 0.0f; steer = 0.0f;
      }
    } else if (est) {
      v = 0.0; d = 0.0; throttle = 0.0f; steer = 0.0f;
    }

    // steering EMA, applied last (after policy AND any recovery/estop override) so state
    // transitions into/out of reverse-recovery are smoothed too, not just normal driving
    {
      double alpha = node_->get_parameter(prefix("steer_ema_alpha")).as_double();
      d = alpha * steer_cmd_prev_ + (1.0 - alpha) * d;
      steer_cmd_prev_ = d;
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
    steer_cmd_prev_ = 0.0;
    temporal_ema_.assign(N_SECTORS, 0.0f);
    ring_.clear();
    estopped_ = false;
    estop_timing_ = false;
    nomove_timing_ = false;
    reversing_ = false;
    failed_reverses_ = 0;
    have_settled_since_ = false;
    dock_reversing_ = false;
    was_docking_ = false;
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

  // walk forward from closest_idx_ accumulating arc length until lookahead_dist; interpolate
  // the carrot point on the segment where it's crossed, tangent = that segment's direction.
  // If the remaining path is shorter than the lookahead, carrot = path endpoint.
  void computeCarrot(const Vec2 & robot, double lookahead_dist, Vec2 & carrot, double & tangent) const
  {
    size_t n = path_.size();
    if (n == 0) { carrot = robot; tangent = 0.0; return; }
    if (n == 1) { carrot = path_[0]; tangent = 0.0; return; }

    size_t i = std::min(closest_idx_, n - 1);
    double acc = norm2({robot.x - path_[i].x, robot.y - path_[i].y});
    if (acc >= lookahead_dist || i + 1 >= n) {
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
      if (acc + seg_len >= lookahead_dist || k + 2 == n) {
        double remain = lookahead_dist - acc;
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

  rclcpp::Logger logger_{rclcpp::get_logger("rl_controller_v4")};
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
  double steer_cmd_prev_{0.0};
  bool estopped_{false};
  bool estop_timing_{false};
  rclcpp::Time estop_since_;
  bool nomove_timing_{false};
  rclcpp::Time nomove_since_;
  bool reversing_{false};
  double reverse_dir_{1.0};   // decided once when the reverse starts, held for its duration
  Vec2 reverse_start_{0, 0};
  rclcpp::Time reverse_begin_;
  int failed_reverses_{0};
  rclcpp::Time settled_since_;
  bool have_settled_since_{false};
  bool dock_reversing_{false};
  bool was_docking_{false};
  rclcpp::Time last_time_;
  bool have_last_time_{false};

  std::vector<float> h_;

  std::unique_ptr<Ort::Env> env_;
  std::unique_ptr<Ort::Session> session_;
  std::unique_ptr<Ort::SessionOptions> session_options_;
};

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(RLControllerV4, nav2_core::Controller)
