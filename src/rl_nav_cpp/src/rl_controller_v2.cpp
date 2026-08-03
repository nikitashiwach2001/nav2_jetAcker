// RL Ackermann controller v2 for Nav2 (parallel plugin, does NOT replace rl_controller.cpp / v1).
//
// Model: policy_reverse_15995.onnx  (stateless MLP, no recurrence)
//   input : obs     [1, 62] float32  (raw -- normalizer baked into the graph)
//   output: actions [1, 2]  float32  = [throttle, steer], UNBOUNDED -> clamp to [-1,1] here
//
// Observation layout (62), exact order given by the model's training spec:
//   [0..39]  lidar_sectors    40 sectors x 9 deg; value = min(range in sector)/4.0, clip[0,1];
//            no-hit/inf -> 1.0. sector 0 = robot forward (0 deg), index increases CCW
//            (left ~90deg -> sector ~10, back ~180deg -> sector ~20, right -90deg -> sector ~30)
//   [40]     goal_dist        clip(||goal-robot|| / 8.0, 0, 1)
//   [41]     goal cos(theta)  theta = bearing_to_goal - yaw, robot body frame, CCW+
//   [42]     goal sin(theta)
//   [43..57] reserved (second_goal / waypoints) -- ALWAYS ZERO, this deployment doesn't use them
//   [58]     prev throttle    last cycle's RAW clamped onnx output[0], [-1,1]
//   [59]     prev steer       last cycle's RAW clamped onnx output[1], [-1,1]
//   [60]     steer angle      current_steer_angle / 0.60, clip[-1,1]
//            NOTE: no verified servo position-feedback topic on this robot yet -- reconstructed
//            from our own last commanded steer (same proxy rl_controller.cpp/v1 uses). Replace
//            with real feedback once a topic is confirmed (see integration notes).
//   [61]     steer rate       steer_angular_vel / 10.0, clip[-1,1] (finite diff of the above)
//
// Action -> Ackermann (throttle may be NEGATIVE: this model decides its own reverse, unlike v1
// which always zeroed reverse and used hardcoded recovery code instead):
//   throttle = clamp(out[0],-1,1) ; steer = clamp(out[1],-1,1)
//   v = throttle * v_cap   (m/s, v_cap is a ROS param -- START LOW, ramp up during field testing)
//   d = steer * 0.60 (rad)
//   Twist: linear.x = v ; angular.z = v * tan(d) / 0.213
//
// Safety design: this controller does NOT reimplement a stuck/recovery state machine like v1 --
// the model is meant to decide forward/reverse/avoidance itself. What stays independent of the
// model, always, regardless of how well it drives:
//   1. Lidar staleness guard (this robot's lidar intermittently stalls -- hold position, don't
//      trust a frozen scan)
//   2. Front-corridor e-stop VETO: if the model commands forward into a lidar-visible blocker
//      inside estop_distance, clamp v to 0 for that cycle (never overrides steering, just speed)
//   3. Reverse-contact VETO: the robot's rear ~54 deg arc is PHYSICALLY blind (own structure
//      blocks the lidar -- confirmed by raw-serial probe, not a software gap). A model trained in
//      sim with full 360 deg visibility will see "clear" there even when something's touching the
//      robot. If reverse is commanded but real measured velocity stays ~0, clamp v to 0 -- this is
//      a MOTION-based check (odometry), independent of what the lidar sectors say about the rear.
// Both vetoes only clamp linear.x for the current cycle; they never take over steering or attempt
// their own recovery maneuver -- that's left to Nav2's behavior tree (progress_checker + BackUp).

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
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace
{
constexpr int OBS_DIM = 62;
constexpr int N_SECTORS = 40;
constexpr double SECTOR_WIDTH = 2.0 * M_PI / N_SECTORS;   // 9 deg

// training constants (exact, from the model spec)
constexpr float LIDAR_CAP = 4.0f;               // LIDAR_MAX_DISTANCE (m)
constexpr float GOAL_DIST_NORM = 8.0f;
constexpr float GOAL_RADIUS = 0.40f;            // waypoint capture radius
constexpr float STEER_ANGLE_NORM = 0.60f;       // JETACKER_MAX_STEER (rad)
constexpr float STEER_RATE_NORM = 10.0f;        // rad/s
constexpr float STEP_DT = 1.0f / 30.0f;         // 30 Hz control

constexpr double STEER_SCALE = 0.60;            // rad at action = 1
constexpr double WHEELBASE = 0.213;             // m (JETACKER_WHEELBASE)

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

class RLControllerV2 : public nav2_core::Controller
{
public:
  RLControllerV2() = default;

  void configure(
    const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
    std::string name,
    std::shared_ptr<tf2_ros::Buffer>,
    std::shared_ptr<nav2_costmap_2d::Costmap2DROS>) override
  {
    node_ = parent.lock();
    logger_ = node_->get_logger();
    plugin_name_ = name;

    declareParam("model_path", "/home/ubuntu/ros2_ws/src/rl_nav_cpp/policies/policy_reverse_15995.onnx");
    // With no lookahead in the observation (idx 43-57 always zero) the model can't see the
    // path's shape beyond the current target point -- keep waypoints closer together than v1's
    // 2.0m so the carrot still roughly traces the Nav2 plan. Tune during field testing.
    declareParam("waypoint_spacing", 1.0);
    // testing switch: skip the carrot/waypoint-advance system entirely and always feed the
    // model Nav2's FINAL goal point (path endpoint) as obs[40..42], nothing in between. Use
    // this to test the raw policy's own point-to-point + lidar-avoidance behavior without any
    // waypoint guidance shaping it.
    declareParam("final_goal_only", false);
    declareParam("goal_change_thresh", 0.5);
    declareParam("lidar_angle_offset", M_PI);    // rad; this lidar is mounted yawed 180 deg
    declareParam("lidar_ccw", true);
    // deploy speed cap: START LOW. Model spec says 1.0-1.2 m/s deploy target, but that requires
    // navigation/config/nav2_params.yaml velocity_smoother max_velocity raised from its current
    // 0.26 first, AND collision_monitor stop/slowdown distances retuned for the higher stopping
    // distance (v^2). Ramp this param up in stages during testing, don't jump straight to 1.0.
    declareParam("v_cap", 0.4);
    declareParam("estop_distance", 0.30);        // m; forward corridor veto threshold (0 disables)
    declareParam("estop_half_width", 0.17);      // m; corridor half-width = robot half-width + margin
    declareParam("reverse_stall_sec", 0.8);      // s; commanding reverse w/ no real motion this long -> veto
    declareParam("debug", true);

    std::string model_path = node_->get_parameter(prefix("model_path")).as_string();

    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "rl_v2");
    session_options_ = std::make_unique<Ort::SessionOptions>();
    session_options_->SetIntraOpNumThreads(1);
    session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), *session_options_);

    Ort::AllocatorWithDefaultOptions alloc;
    size_t n_in = session_->GetInputCount();
    size_t n_out = session_->GetOutputCount();
    if (n_in != 1 || n_out != 1) {
      throw std::runtime_error(
        "RLControllerV2: expected a stateless 1-input/1-output ONNX graph, got " +
        std::to_string(n_in) + " inputs / " + std::to_string(n_out) + " outputs. "
        "This plugin does not carry recurrent state -- if the model needs a hidden-state "
        "input/output, this file needs updating.");
    }
    input_name_ = session_->GetInputNameAllocated(0, alloc).get();
    output_name_ = session_->GetOutputNameAllocated(0, alloc).get();

    // /scan_raw, not /scan: this robot's /scan filter chain's angular bounds are wrong for the
    // yaw-180 lidar mount (confirmed 2026-07-08 on v1) -- keeps the physically-blind rear sector
    // and drops the valid front. Same fix applies here.
    scan_sub_ = node_->create_subscription<sensor_msgs::msg::LaserScan>(
      "/scan_raw", rclcpp::SensorDataQoS(),
      [this](sensor_msgs::msg::LaserScan::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(scan_mtx_);
        last_scan_ = msg;
        last_scan_time_ = node_->now();
      });

    RCLCPP_INFO(logger_,
      "RL Ackermann controller v2 configured (62-dim goal+lidar obs, stateless, reverse-capable). "
      "Model: %s (input '%s' -> output '%s')",
      model_path.c_str(), input_name_.c_str(), output_name_.c_str());
  }

  void cleanup() override {}
  void activate() override { resetEpisode(); }
  void deactivate() override {}
  void setSpeedLimit(const double &, const bool &) override {}

  void setPlan(const nav_msgs::msg::Path & path) override
  {
    buildWaypoints(path);
    if (waypoints_.empty()) return;

    Vec2 new_goal = waypoints_.back();
    bool new_goal_started =
      !have_goal_ || norm2({new_goal.x - last_goal_.x, new_goal.y - last_goal_.y})
                       > node_->get_parameter(prefix("goal_change_thresh")).as_double();

    if (new_goal_started) {
      resetEpisode();
      wp_index_ = 0;
    } else {
      wp_index_ = nearestWaypoint(last_robot_);
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
    if (waypoints_.empty()) return cmd;

    Vec2 robot{pose.pose.position.x, pose.pose.position.y};
    double ryaw = yawFromQuat(pose.pose.orientation);
    last_robot_ = robot;

    rclcpp::Time now = node_->now();
    double dt = have_last_time_ ? (now - last_time_).seconds() : STEP_DT;
    if (dt <= 1e-4 || dt > 1.0) dt = STEP_DT;
    last_time_ = now;
    have_last_time_ = true;

    // --- sensor-outage guard (see file header) ---
    sensor_msgs::msg::LaserScan::SharedPtr scan;
    double scan_age;
    {
      std::lock_guard<std::mutex> lock(scan_mtx_);
      scan = last_scan_;
      scan_age = last_scan_ ? (now - last_scan_time_).seconds() : 1e9;
    }
    if (scan_age > 0.5 || !scan || scan->ranges.empty()) {
      RCLCPP_WARN_THROTTLE(logger_, *node_->get_clock(), 2000,
        "scan is %.1f s old / missing (lidar outage?) - holding position", scan_age);
      prev_action_[0] = 0.0f;
      prev_action_[1] = 0.0f;
      steer_prev_ = steer_last_;
      steer_last_ = 0.0;
      reverse_cmd_timing_ = false;
      return cmd;   // zero twist
    }

    double offset = node_->get_parameter(prefix("lidar_angle_offset")).as_double();
    double dir = node_->get_parameter(prefix("lidar_ccw")).as_bool() ? 1.0 : -1.0;

    std::array<float, OBS_DIM> obs{};
    int o = 0;

    // [0..39] lidar sectors
    std::array<float, N_SECTORS> sectors = computeLidarSectors(*scan, offset, dir);
    for (int s = 0; s < N_SECTORS; ++s) obs[o++] = sectors[s];

    // waypoint advance, then goal bearing in body frame
    bool final_goal_only = node_->get_parameter(prefix("final_goal_only")).as_bool();
    size_t n_wp = waypoints_.size();
    size_t cur_i;
    if (final_goal_only) {
      cur_i = n_wp - 1;             // always the path endpoint -- no carrot advancing at all
    } else {
      advanceWaypoint(robot);
      cur_i = std::min(wp_index_, n_wp - 1);
    }
    double cyaw = std::cos(ryaw), syaw = std::sin(ryaw);

    Vec2 to_goal{waypoints_[cur_i].x - robot.x, waypoints_[cur_i].y - robot.y};
    double bx = to_goal.x * cyaw + to_goal.y * syaw;     // world -> body: R(-yaw)
    double by = -to_goal.x * syaw + to_goal.y * cyaw;
    double dist = std::hypot(bx, by);
    double cos_t = dist > 1e-6 ? bx / dist : 1.0;
    double sin_t = dist > 1e-6 ? by / dist : 0.0;

    // [40] goal dist, [41] cos theta, [42] sin theta
    obs[o++] = clampf(static_cast<float>(dist / GOAL_DIST_NORM), 0.0f, 1.0f);
    obs[o++] = static_cast<float>(cos_t);
    obs[o++] = static_cast<float>(sin_t);

    // [43..57] reserved -- always zero
    for (int k = 0; k < 15; ++k) obs[o++] = 0.0f;

    // [58..59] previous raw actions
    obs[o++] = prev_action_[0];
    obs[o++] = prev_action_[1];

    // [60..61] steering angle / rate -- reconstructed proxy, see file header
    obs[o++] = clampf(static_cast<float>(steer_last_ / STEER_ANGLE_NORM), -1.0f, 1.0f);
    float steer_rate = static_cast<float>((steer_last_ - steer_prev_) / dt);
    obs[o++] = clampf(steer_rate / STEER_RATE_NORM, -1.0f, 1.0f);

    // --- inference: single obs -> single actions, no recurrent state ---
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::array<int64_t, 2> obs_shape{1, OBS_DIM};
    Ort::Value input = Ort::Value::CreateTensor<float>(
      mem, obs.data(), obs.size(), obs_shape.data(), obs_shape.size());

    const char * in_names[] = {input_name_.c_str()};
    const char * out_names[] = {output_name_.c_str()};
    auto outputs = session_->Run(
      Ort::RunOptions{nullptr}, in_names, &input, 1, out_names, 1);

    const float * action = outputs[0].GetTensorData<float>();
    float throttle = clampf(action[0], -1.0f, 1.0f);
    float steer = clampf(action[1], -1.0f, 1.0f);

    double v_cap = node_->get_parameter(prefix("v_cap")).as_double();
    double v = static_cast<double>(throttle) * v_cap;
    double d = static_cast<double>(steer) * STEER_SCALE;

    // --- safety veto 1: forward into a lidar-visible blocker ---
    double estop_d = node_->get_parameter(prefix("estop_distance")).as_double();
    if (v > 0.02 && estop_d > 0.0) {
      double half_w = node_->get_parameter(prefix("estop_half_width")).as_double();
      float front_min = frontCorridorMin(*scan, offset, dir, half_w);
      if (front_min < estop_d) {
        RCLCPP_WARN_THROTTLE(logger_, *node_->get_clock(), 1000,
          "VETO forward: blocker %.2f m ahead in corridor (policy wanted v=%.2f)",
          front_min, v);
        v = 0.0;
      }
    }

    // --- safety veto 2: reverse into something the (physically blind) rear can't see ---
    // Don't just zero the twist and let the SAME policy try again next cycle -- with an
    // unchanged observation it's likely to command reverse again immediately, which looks
    // like continuous grinding against the obstacle instead of a clean stop. Hand off to
    // Nav2 instead: throwing here fails this FollowPath attempt, and the behavior tree's
    // existing BackUp recovery (0.40 m @ 0.08 m/s, navigate_to_pose_w_replanning_and_recovery
    // _ackermann.xml) backs the robot out under its own control, then the policy gets a
    // fresh approach. That recovery action publishes /cmd_vel directly (bypasses this
    // controller entirely), so it isn't subject to the same blind-rear problem repeating.
    double stall_sec = node_->get_parameter(prefix("reverse_stall_sec")).as_double();
    bool moving = std::abs(velocity.linear.x) > 0.02;
    if (v < -0.02) {
      if (!reverse_cmd_timing_) { reverse_cmd_timing_ = true; reverse_cmd_since_ = now; }
      if (moving) {
        reverse_cmd_timing_ = false;   // genuinely backing up -- fine, reset the timer
      } else if ((now - reverse_cmd_since_).seconds() > stall_sec) {
        reverse_cmd_timing_ = false;
        RCLCPP_WARN(logger_,
          "reverse commanded (v=%.2f) for %.1fs with no real motion -> rear contact likely, "
          "handing off to Nav2 recovery (BackUp)", v, stall_sec);
        throw nav2_core::PlannerException(
          "RL controller v2: reverse stalled, suspected rear contact (physically blind arc)");
      }
    } else {
      reverse_cmd_timing_ = false;
    }

    cmd.twist.linear.x = v;
    cmd.twist.angular.z = (std::abs(v) > 1e-6) ? v * std::tan(d) / WHEELBASE : 0.0;

    // prev_action stores the model's RAW output (pre-veto), per training spec
    prev_action_[0] = throttle;
    prev_action_[1] = steer;
    steer_prev_ = steer_last_;
    steer_last_ = d;

    if (node_->get_parameter(prefix("debug")).as_bool()) {
      RCLCPP_INFO(logger_,
        "wp %zu/%zu act[%.2f %.2f] goal[d=%.2f cos=%.2f sin=%.2f] -> v=%.3f d=%.3f w=%.3f",
        cur_i, n_wp, throttle, steer, dist, cos_t, sin_t,
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
    prev_action_[0] = prev_action_[1] = 0.0f;
    steer_last_ = steer_prev_ = 0.0;
    reverse_cmd_timing_ = false;
  }

  // min forward distance of any return whose lateral offset is inside the robot's swept
  // corridor (same geometric definition as rl_controller.cpp v1's frontCorridorMin, computed
  // directly off the raw scan instead of a resampled ring buffer -- v2 keeps no frame history).
  float frontCorridorMin(
    const sensor_msgs::msg::LaserScan & s, double offset, double dir, double half_width) const
  {
    float best = LIDAR_CAP;
    int n = static_cast<int>(s.ranges.size());
    for (int k = 0; k < n; ++k) {
      float r = s.ranges[k];
      if (!std::isfinite(r) || r < 0.05f) continue;
      double raw_ang = s.angle_min + k * s.angle_increment;
      double b = dir * (raw_ang - offset);
      b = std::atan2(std::sin(b), std::cos(b));   // wrap to [-pi,pi], robot body frame
      double cb = std::cos(b);
      if (cb <= 0.0) continue;                    // rear half-plane
      if (std::abs(r * std::sin(b)) <= half_width) {
        best = std::min(best, static_cast<float>(r * cb));
      }
    }
    return best;
  }

  // 40 sectors x 9 deg, sector 0 = robot forward, index increases CCW. min range per sector,
  // normalized /4.0, no-hit -> 1.0 (far/clear).
  std::array<float, N_SECTORS> computeLidarSectors(
    const sensor_msgs::msg::LaserScan & s, double offset, double dir) const
  {
    std::array<float, N_SECTORS> sector_min;
    sector_min.fill(std::numeric_limits<float>::infinity());
    int n = static_cast<int>(s.ranges.size());
    for (int k = 0; k < n; ++k) {
      float r = s.ranges[k];
      if (!std::isfinite(r) || r < 0.05f) continue;
      double raw_ang = s.angle_min + k * s.angle_increment;
      double b = dir * (raw_ang - offset);
      b = std::atan2(std::sin(b), std::cos(b));
      if (b < 0.0) b += 2.0 * M_PI;                // wrap to [0, 2pi)
      int sec = static_cast<int>(b / SECTOR_WIDTH);
      if (sec >= N_SECTORS) sec = N_SECTORS - 1;
      sector_min[sec] = std::min(sector_min[sec], r);
    }
    std::array<float, N_SECTORS> out;
    for (int i = 0; i < N_SECTORS; ++i) {
      out[i] = std::isinf(sector_min[i]) ? 1.0f : clampf(sector_min[i] / LIDAR_CAP, 0.0f, 1.0f);
    }
    return out;
  }

  // ---- carrot (discrete waypoints), same scheme as rl_controller.cpp v1 ----
  void buildWaypoints(const nav_msgs::msg::Path & path)
  {
    waypoints_.clear();
    if (path.poses.empty()) return;
    double spacing = node_->get_parameter(prefix("waypoint_spacing")).as_double();
    Vec2 last{path.poses[0].pose.position.x, path.poses[0].pose.position.y};
    waypoints_.push_back(last);
    for (size_t k = 1; k < path.poses.size(); ++k) {
      Vec2 p{path.poses[k].pose.position.x, path.poses[k].pose.position.y};
      if (norm2({p.x - last.x, p.y - last.y}) >= spacing) {
        waypoints_.push_back(p);
        last = p;
      }
    }
    Vec2 back{path.poses.back().pose.position.x, path.poses.back().pose.position.y};
    if (norm2({back.x - last.x, back.y - last.y}) > 1e-3) waypoints_.push_back(back);
  }

  size_t nearestWaypoint(const Vec2 & p) const
  {
    size_t best = 0;
    double bd = std::numeric_limits<double>::max();
    for (size_t i = 0; i < waypoints_.size(); ++i) {
      double d = norm2({waypoints_[i].x - p.x, waypoints_[i].y - p.y});
      if (d < bd) { bd = d; best = i; }
    }
    return best;
  }

  void advanceWaypoint(const Vec2 & robot)
  {
    size_t n = waypoints_.size();
    if (n == 0) return;
    size_t i = std::min(wp_index_, n - 1);
    if (i + 1 >= n) return;   // on the final waypoint: hold, Nav2's goal_checker ends the run
    double dist = norm2({robot.x - waypoints_[i].x, robot.y - waypoints_[i].y});
    double dist_next = norm2({robot.x - waypoints_[i + 1].x, robot.y - waypoints_[i + 1].y});
    bool captured = dist < GOAL_RADIUS;
    bool passed = dist_next < dist;
    if (captured || passed) wp_index_ = i + 1;
  }

  rclcpp::Logger logger_{rclcpp::get_logger("rl_controller_v2")};
  rclcpp_lifecycle::LifecycleNode::SharedPtr node_;
  std::string plugin_name_;

  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
  sensor_msgs::msg::LaserScan::SharedPtr last_scan_;
  std::mutex scan_mtx_;
  rclcpp::Time last_scan_time_;

  std::vector<Vec2> waypoints_;
  size_t wp_index_{0};
  Vec2 last_goal_{0, 0};
  Vec2 last_robot_{0, 0};
  bool have_goal_{false};

  float prev_action_[2]{0.0f, 0.0f};
  double steer_last_{0.0}, steer_prev_{0.0};
  bool reverse_cmd_timing_{false};
  rclcpp::Time reverse_cmd_since_;
  rclcpp::Time last_time_;
  bool have_last_time_{false};

  std::unique_ptr<Ort::Env> env_;
  std::unique_ptr<Ort::Session> session_;
  std::unique_ptr<Ort::SessionOptions> session_options_;
  std::string input_name_, output_name_;
};

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(RLControllerV2, nav2_core::Controller)
