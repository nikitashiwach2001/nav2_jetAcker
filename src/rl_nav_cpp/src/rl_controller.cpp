// RL Ackermann controller for Nav2 (replacement for FollowPath).
//
// Model: policy_full_new.onnx  (CnnGruActorCritic, pure-pursuit waypoint config)
//   inputs : obs   [1, 1102]   (raw, no extra normalization)
//            h_in  [1, 1, 256] (GRU hidden state, carried across steps)
//   outputs: actions [1, 2]    (raw, UNBOUNDED -> clamp to [-1,1] here)
//            h_out   [1, 1, 256]
//
// Observation layout (1102), exact training order (waypoint config, 180 rays):
//   [0   .. 1079] lidar_stacked   6 frames x 180 rays, newest-first, frames 5 steps apart
//                 [t, t-5, t-10, t-15, t-20, t-25]; ray i at bearing i*2deg CCW from robot +X;
//                 value = clamp(range,0,4)/4 ; no-return/nan/inf -> 1.0
//   [1080..1087] lidar_temporal_sector_diff  8 sectors, [-1,1], -1 = approaching fast
//   [1088]       goal_distance   ||cur_waypoint - robot|| / 33.3048, clamped [0,1]
//   [1089]       goal_angle      wrap(bearing_to_cur_waypoint - yaw) / pi, [-1,1]
//   [1090..1095] next_waypoint   body-frame XY of the next 3 waypoints (wp+1,wp+2,wp+3),
//                 each / 8.0, clamped [-1,1]; end-of-path duplicates the final waypoint
//   [1096..1097] robot_velocity  [fwd / 2.0, yaw_rate / 3.0], clamped
//   [1098..1099] previous_actions [throttle, steering] raw [-1,1]
//   [1100]       steering_angle  actual front-wheel angle / 0.60, clamped
//   [1101]       steering_rate   front-wheel angular velocity / 10.0, clamped
//
// Waypoints (pure-pursuit): the Nav2 path is resampled into ~2.0 m waypoints; wp_index advances by 1
// when the robot captures the current waypoint (dist < 0.7) or passes it (closer to wp_index+1).
// goal_distance/goal_angle target waypoints[wp_index]; next_waypoint looks 1..3 ahead.
//
// Action -> Ackermann:
//   a0=clamp(out[0],-1,1) ; a1=clamp(out[1],-1,1)
//   v = max(0,a0) * 0.6   (m/s, forward only) ; d = a1 * 0.60 (rad)
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
constexpr int OBS_DIM = 1102;
constexpr int HIDDEN = 256;

// confirmed training constants
constexpr float LIDAR_CAP = 4.0f;               // LIDAR_MAX_DISTANCE (m)
constexpr float GOAL_DIST_NORM = 33.3048f;      // sqrt(2) * ARENA_SIZE(23.55): goal_distance normalizer
constexpr float LOOKAHEAD_NORM = 8.0f;          // next_waypoint body-frame XY normalizer (m)
constexpr int   N_LOOKAHEAD = 3;                // waypoints ahead exposed to the policy
constexpr float WP_CAPTURE_RADIUS = 0.7f;       // GOAL_RADIUS: via-point captured when dist < 0.7 m
constexpr size_t WP_RESYNC_WINDOW = 3;          // forward-only re-sync search depth (waypoints)
constexpr float VEL_LIN_NORM = 2.0f;            // JETACKER_MAX_LIN_VEL
constexpr float VEL_YAW_NORM = 3.0f;            // JETACKER_MAX_ANG_VEL
constexpr float STEER_ANGLE_NORM = 0.60f;       // JETACKER_MAX_STEER
constexpr float STEER_RATE_NORM = 10.0f;        // MAX_STEER_RATE
constexpr float MAX_APPROACH = 1.0f;            // MAX_OBSTACLE_APPROACH_SPEED (m/s)
constexpr float TEMPORAL_EMA = 0.8f;
constexpr float STEP_DT = 0.0333333f;           // 30 Hz control (decimation 4 x 1/120)
constexpr double RESUME_SETTLE_SEC = 5.0;       // sustained (not momentary) motion needed
                                                 // to clear the boxed-in failure streak

// action scalings (Ackermann)
// NOTE: throttle scale is the `throttle_scale` parameter now. It MUST stay <= the velocity
// smoother's max_velocity: the smoother clips linear.x but passes angular.z through
// (scale_velocities: False), so any clipping more than doubles the executed curvature and
// the robot arcs far tighter than the policy intended.
constexpr double STEER_SCALE = 0.60;            // rad at action = 1 (== JETACKER_MAX_STEER)
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

class RLController : public nav2_core::Controller
{
public:
  RLController() = default;

  void configure(
    const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
    std::string name,
    std::shared_ptr<tf2_ros::Buffer>,
    std::shared_ptr<nav2_costmap_2d::Costmap2DROS>) override
  {
    node_ = parent.lock();
    logger_ = node_->get_logger();
    plugin_name_ = name;

    declareParam("model_path", "/home/ubuntu/ros2_ws/src/rl_nav_cpp/policies/policy_full_new.onnx");
    declareParam("waypoint_spacing", 2.0);       // m between resampled carrot waypoints (training: 2.0)
    declareParam("goal_change_thresh", 0.5);     // m endpoint move that counts as a new goal -> reset GRU
    declareParam("lidar_angle_offset", 0.0);     // rad; bearing of training ray 0 in the /scan frame
    declareParam("lidar_ccw", true);             // increasing ray index = CCW (REP-103)
    // empirical knob: if the robot consistently, confidently steers the WRONG way (not
    // oscillating -- committing hard to one direction and never correcting, e.g. path is
    // clearly to one side and the robot just keeps heading its own current way), flip this
    // to -1.0 live (no rebuild): ros2 param set /controller_server FollowPath.steer_sign -1.0
    // v2/v3/v4 always had this knob for exactly this risk; v1 never did until now.
    declareParam("steer_sign", 1.0);
    declareParam("estop_distance", 0.30);        // m; blocker in the forward corridor closer -> stop (0 disables)
    declareParam("estop_release", 0.40);         // m; hysteresis: resume once the corridor clears past this
    declareParam("estop_half_width", 0.17);      // m; corridor half-width = robot half-width + margin
    declareParam("estop_fail_sec", 1.5);         // s; stuck this long -> reverse recovery (0 disables)
    declareParam("reverse_speed", 0.10);         // m/s; built-in reverse recovery speed
    declareParam("reverse_dist", 0.80);          // m; hard cap on how far a recovery may back up
    declareParam("reverse_stall_sec", 0.5);      // s; no real motion this long while reversing
                                                 // -> rear contact likely, stop grinding (was 0.8)
    declareParam("reverse_clear", 0.55);         // m; stop reversing once the corridor ahead is clear past this
    declareParam("reverse_rear_stop", 0.22);     // m; rear-arc ray closer than this stops the reverse
    declareParam("throttle_scale", 0.25);        // m/s at action=1; keep <= velocity smoother max
    declareParam("dock_dist", 1.0);              // m; within this of the path end -> docking mode (0 disables)
    declareParam("dock_speed", 0.12);            // m/s; docking approach speed cap
    declareParam("dock_kp", 1.5);                // steering gain on heading error while docking
    // EMA on the final commanded steering angle (weight on the PREVIOUS command): kills the
    // ~3.6 Hz weave jitter while passing real steering intent (<1 Hz turns).
    // steer_cmd = alpha*prev + (1-alpha)*policy ; alpha=0.7 -> ~1.7 Hz cutoff. Raise toward
    // 0.8 for smoother/more lag, lower toward 0.6 for more responsive. Replaces the old
    // alpha-on-new + hard rate-cap combo: the rate cap limited how much the angle could
    // move per 33 ms cycle (4.0 rad/s * 0.033 s =~ 0.13 rad), which could take several
    // cycles to reach full lock (0.60 rad) on a sharp turn -- likely why steering wasn't
    // reaching full deflection in time and the robot got stuck. A plain EMA has no such
    // hard per-cycle ceiling.
    declareParam("steer_ema_alpha", 0.7);
    // oscillation detector (supervisory safety layer, NOT a filter): if the commanded
    // steering direction flips this many times within the window while actively driving
    // forward, hand off to TEB rather than let the policy keep weaving indefinitely.
    // Tune based on real behavior: a genuine winding path legitimately flips sign every
    // 1-2 s, so these defaults require flips faster than that to trigger.
    declareParam("oscillation_flip_count", 5);
    declareParam("oscillation_window_sec", 4.0);
    declareParam("oscillation_min_speed", 0.05);   // m/s; below this, don't count flips
    // heading-divergence detector (supervisory safety layer): if the heading error to the
    // target waypoint stays above this magnitude (rad) for this long while actively
    // driving, hand off to TEB -- catches "confidently committing to the wrong direction
    // and never correcting" (e.g. steer-sign mismatch), which the oscillation detector
    // (needs sign flips) never sees. ~2.0 rad =~ 115 deg: past this the target is behind
    // the robot, which briefly happens on legitimate sharp turns too, so this only fires
    // if it STAYS this bad for heading_diverge_sec straight.
    declareParam("heading_diverge_thresh", 2.0);
    declareParam("heading_diverge_sec", 3.0);
    declareParam("debug", true);

    std::string model_path = node_->get_parameter(prefix("model_path")).as_string();

    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "rl");
    session_options_ = std::make_unique<Ort::SessionOptions>();
    session_options_->SetIntraOpNumThreads(1);
    session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), *session_options_);

    h_.assign(HIDDEN, 0.0f);
    temporal_ema_.assign(N_SECTORS, 0.0f);

    // /scan_raw, NOT /scan: the filter chain's angular bounds are wrong for this yaw-180
    // lidar mount - measured 2026-07-08, /scan keeps only the robot's REAR sector (which
    // is physically blind, 0 valid rays) and drops the fully-valid front. On /scan the
    // corridor e-stop and rear check were both reading "no data = 4 m clear".
    scan_sub_ = node_->create_subscription<sensor_msgs::msg::LaserScan>(
      "/scan_raw", rclcpp::SensorDataQoS(),
      [this](sensor_msgs::msg::LaserScan::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(scan_mtx_);
        last_scan_ = msg;
        last_scan_time_ = node_->now();
      });

    RCLCPP_INFO(logger_, "RL Ackermann controller configured (1102-dim waypoint obs, GRU). Model: %s",
                model_path.c_str());
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
      resetEpisode();          // genuinely new goal -> reset recurrent state + carrot
      wp_index_ = 0;
    } else {
      // replan to the same goal -> keep GRU state, re-localize the carrot to the nearest waypoint
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

    // dt for the steering-rate finite difference
    rclcpp::Time now = node_->now();
    double dt = have_last_time_ ? (now - last_time_).seconds() : STEP_DT;
    if (dt <= 1e-4 || dt > 1.0) dt = STEP_DT;
    last_time_ = now;
    have_last_time_ = true;

    // --- sensor-outage guard: this lidar intermittently stops streaming. A frozen scan
    // also freezes rf2o (measured velocity -> 0) and the collision monitor (robot stopped),
    // so the stuck detectors below would mis-read the outage as wall contact and reverse
    // BLIND (the rear check reads the same frozen scan). Hold still until data returns.
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
      return cmd;   // zero twist
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

    // waypoints: advance the discrete index, then read the goal + lookahead terms
    advanceWaypoint(robot);
    size_t n_wp = waypoints_.size();
    size_t cur_i = std::min(wp_index_, n_wp - 1);
    double cyaw = std::cos(ryaw), syaw = std::sin(ryaw);

    // current target waypoint -> goal_distance / goal_angle relative to it
    Vec2 to_cur{waypoints_[cur_i].x - robot.x, waypoints_[cur_i].y - robot.y};
    double goal_dist = norm2(to_cur);
    double bearing = std::atan2(to_cur.y, to_cur.x);
    double gangle = std::atan2(std::sin(bearing - ryaw), std::cos(bearing - ryaw));

    // [1088] goal_distance, [1089] goal_angle
    obs[o++] = clampf(static_cast<float>(goal_dist / GOAL_DIST_NORM), 0.0f, 1.0f);
    obs[o++] = clampf(static_cast<float>(gangle / M_PI), -1.0f, 1.0f);

    // [1090..1095] next_waypoint: body-frame XY of the next N_LOOKAHEAD waypoints, clamped to last
    for (int k = 1; k <= N_LOOKAHEAD; ++k) {
      size_t idx = std::min(wp_index_ + static_cast<size_t>(k), n_wp - 1);
      Vec2 d{waypoints_[idx].x - robot.x, waypoints_[idx].y - robot.y};
      double bx = d.x * cyaw + d.y * syaw;      // world -> body: R(-yaw)
      double by = -d.x * syaw + d.y * cyaw;
      obs[o++] = clampf(static_cast<float>(bx / LOOKAHEAD_NORM), -1.0f, 1.0f);
      obs[o++] = clampf(static_cast<float>(by / LOOKAHEAD_NORM), -1.0f, 1.0f);
    }

    // [1096..1097] robot_velocity (body frame)
    obs[o++] = clampf(static_cast<float>(velocity.linear.x) / VEL_LIN_NORM, -1.0f, 1.0f);
    obs[o++] = clampf(static_cast<float>(velocity.angular.z) / VEL_YAW_NORM, -1.0f, 1.0f);

    // [1098..1099] previous actions (raw clamped)
    obs[o++] = prev_action_[0];
    obs[o++] = prev_action_[1];

    // [1100] steering_angle, [1101] steering_rate  (reconstructed from last command)
    obs[o++] = clampf(static_cast<float>(steer_last_ / STEER_ANGLE_NORM), -1.0f, 1.0f);
    float steer_rate = static_cast<float>((steer_last_ - steer_prev_) / dt);
    obs[o++] = clampf(steer_rate / STEER_RATE_NORM, -1.0f, 1.0f);

    // --- inference: obs + h_in -> actions + h_out ---
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::array<int64_t, 2> obs_shape{1, OBS_DIM};
    std::array<int64_t, 3> h_shape{1, 1, HIDDEN};
    std::array<Ort::Value, 2> inputs{
      Ort::Value::CreateTensor<float>(mem, obs.data(), obs.size(), obs_shape.data(), obs_shape.size()),
      Ort::Value::CreateTensor<float>(mem, h_.data(), h_.size(), h_shape.data(), h_shape.size())};

    const char * in_names[] = {"obs", "h_in"};
    const char * out_names[] = {"actions", "h_out"};
    auto outputs = session_->Run(
      Ort::RunOptions{nullptr}, in_names, inputs.data(), 2, out_names, 2);

    const float * action = outputs[0].GetTensorData<float>();
    const float * h_out = outputs[1].GetTensorData<float>();
    std::copy(h_out, h_out + HIDDEN, h_.begin());   // carry recurrent state

    float a0 = clampf(action[0], -1.0f, 1.0f);
    float a1 = clampf(action[1], -1.0f, 1.0f);

    // --- action -> Ackermann twist ---
    double v = std::max(0.0f, a0) * node_->get_parameter(prefix("throttle_scale")).as_double();
    double steer_sign = node_->get_parameter(prefix("steer_sign")).as_double();
    double d = steer_sign * static_cast<double>(a1) * STEER_SCALE;

    // --- terminal docking ---
    // Training "captured" waypoints at 0.7 m (WP_CAPTURE_RADIUS); the policy never saw the
    // final 0.7 -> 0.25 m approach the goal checker demands, and once misaligned that close
    // an Ackermann base orbits the goal on its turning circle forever. Inside dock_dist of
    // the path end a plain pure-pursuit takes over: drive at the goal point; if the goal
    // drifts too far off-axis, back-and-fill (reverse with opposite steer) to realign.
    double dock_dist = node_->get_parameter(prefix("dock_dist")).as_double();
    Vec2 fin = waypoints_.back();
    double fin_dist = norm2({fin.x - robot.x, fin.y - robot.y});
    bool docking = dock_dist > 0.0 && cur_i + 2 >= n_wp && fin_dist < dock_dist;
    if (docking) {
      double fb = std::atan2(fin.y - robot.y, fin.x - robot.x);
      double e = std::atan2(std::sin(fb - ryaw), std::cos(fb - ryaw));
      // back-and-fill only when the goal is genuinely behind (min turn radius is
      // wheelbase/tan(0.6) ~= 0.31 m, so forward steering handles large offsets);
      // eager thresholds here caused a jerky forward/reverse churn near the goal
      if (dock_reversing_) {
        if (std::abs(e) < 0.4) dock_reversing_ = false;
      } else if (std::abs(e) > 1.6 && fin_dist > 0.35) {
        dock_reversing_ = true;
      }
      // when already close, just creep in -- the goal checker fires at the tolerance
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
      a0 = 0.0f;
      a1 = 0.0f;
      if (!was_docking_) {
        RCLCPP_INFO(logger_, "docking: %.2f m to goal, heading err %.2f rad", fin_dist, e);
      }
    } else {
      dock_reversing_ = false;
    }
    was_docking_ = docking;

    // --- stuck handling: lidar e-stop + built-in reverse recovery ---
    // The reverse is done HERE, not via the BT BackUp behavior: once the robot is touching
    // a wall its footprint overlaps lethal costmap cells and nav2's BackUp collision check
    // vetoes the motion, so the behavior server can never back out of a real contact.
    bool est = checkEmergencyStop();
    double fail_after = node_->get_parameter(prefix("estop_fail_sec")).as_double();

    if (!reversing_ && fail_after > 0.0) {
      // detector 1: obstacle latched in the forward e-stop arc
      if (est) {
        if (!estop_timing_) { estop_timing_ = true; estop_since_ = now; }
      } else {
        estop_timing_ = false;
      }
      // detector 2: commanding forward but the robot is not moving -- oblique/side wall
      // contact outside the e-stop arc (measured velocity is real: rf2o -> EKF).
      // Threshold above the motor deadband: tiny commands that don't move the robot are
      // normal, and phantom triggers here caused pointless reverses + reorientation arcs.
      bool moving = std::abs(velocity.linear.x) > 0.02 || std::abs(velocity.angular.z) > 0.05;
      // Clear the failure streak only after SUSTAINED progress, not a momentary blip. A
      // robot boxed in front+rear moves a real ~0.2-0.3 m on every single reverse (looks
      // like "recovery done" by distance alone) then drives straight back into the same
      // wall a couple seconds later -- resetting the streak on any instantaneous motion
      // meant the escalation below never engaged, so it oscillated forever instead of
      // escaping or giving up to Nav2.
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
        // Decide the turn direction ONCE, here, and hold it for the whole maneuver.
        // gangle is recomputed every cycle from the live heading error, and while backing
        // up the robot's own yaw is changing (that's the point) -- if gangle happens to
        // sit near zero, its sign can flip mid-reverse, which used to flip the steering
        // direction every single 30 Hz frame (a real zig-zag inside ONE reverse attempt,
        // not just between separate attempts).
        //
        // Prefer whichever side is ACTUALLY physically open, not just goal-oriented: pure
        // gangle-based choice always tries to turn toward the goal/waypoint even when
        // that exact side is the one blocked (e.g. wall on the goal side, open space on
        // the other) -- this repeatedly re-attempts the SAME blocked direction and never
        // makes progress. Only fall back to the goal-oriented choice when both sides are
        // roughly equally clear (no strong physical reason to prefer either).
        {
          float left_clear = sideClearance(true);
          float right_clear = sideClearance(false);
          const float SIDE_MARGIN = 0.15f;  // m; how much clearer one side must be to override goal preference
          if (left_clear > right_clear + SIDE_MARGIN) {
            reverse_dir_ = -1.0;   // left is clearly more open -> turn left regardless of goal side
          } else if (right_clear > left_clear + SIDE_MARGIN) {
            reverse_dir_ = 1.0;    // right is clearly more open -> turn right regardless of goal side
          } else {
            reverse_dir_ = (gangle > 0.0) ? -1.0 : 1.0;   // similarly clear -> goal-oriented as before
          }
        }
        RCLCPP_WARN(logger_, "STUCK (%s) -> reversing out", est_stuck ? "wall in forward arc" : "pushing but not moving");
      }
    }

    if (reversing_) {
      double rev_speed = node_->get_parameter(prefix("reverse_speed")).as_double();
      double rev_dist = node_->get_parameter(prefix("reverse_dist")).as_double();
      double rear_stop = node_->get_parameter(prefix("reverse_rear_stop")).as_double();
      float rear_min = rearArcMin();

      double travelled = norm2({robot.x - reverse_start_.x, robot.y - reverse_start_.y});
      double elapsed = (now - reverse_begin_).seconds();
      // back up until the corridor ahead is open enough to steer away toward the waypoint
      // (>= reverse_clear), with rev_dist as a hard cap. Minimum distance escalates with
      // each failed attempt (same failed_reverses_ streak as the steer-angle escalation
      // above): backing up the SAME short distance every retry just re-attempts the turn
      // from nearly the same spot at nearly the same heading (same wall in the way). Since
      // the reverse steers at a fixed nonzero angle, more distance along that arc means
      // more accumulated heading change (same principle as a 3-point car park) -- each
      // retry ends up genuinely more rotated, not just re-trying the same geometry. This
      // ONLY affects the "front looks clear, safe to turn" exit path -- rear_blocked and
      // rev_stalled (contact detection) below are fully independent and still cut the
      // reverse short immediately (within reverse_stall_sec) regardless of this target.
      double min_travel = std::min(rev_dist, 0.15 + failed_reverses_ * 0.15);
      double clear_x = node_->get_parameter(prefix("reverse_clear")).as_double();
      double half_w = node_->get_parameter(prefix("estop_half_width")).as_double();
      bool clear_to_turn = travelled >= min_travel && frontCorridorMin(half_w) > clear_x;
      bool done = clear_to_turn || travelled >= rev_dist;
      bool rear_blocked = rear_min < rear_stop;
      // contact detector: the rear arc is PHYSICALLY blind (own structure blocks the
      // lidar), so touching an obstacle behind is only visible as "commanding reverse
      // but not moving". Bail fast instead of grinding against it (was 3.0 s).
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
        estopped_ = false;   // re-evaluated next cycle from the new standoff
        double release_d = node_->get_parameter(prefix("estop_release")).as_double();
        bool front_open = frontCorridorMin(half_w) > release_d;
        // count EVERY completed reverse attempt against the streak, whether it stalled
        // (travelled<0.10) or nominally "succeeded" by distance -- a boxed-in robot backs
        // up a real ~0.2-0.3 m each time and still oscillates, so distance alone can't be
        // the only signal. Only sustained forward progress (checked above) clears this.
        ++failed_reverses_;
        if (failed_reverses_ >= 3 || (travelled < 0.10 && !front_open)) {
          int streak = failed_reverses_;
          failed_reverses_ = 0;
          throw nav2_core::PlannerException(
            "RL controller: boxed in (reverse " +
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
          // resume WARM: same GRU state, same carrot waypoint. Resetting the GRU and
          // re-targeting the nearest waypoint here made the policy reorient with wide
          // arcs after every recovery (nearest waypoint can be behind the robot).
          RCLCPP_INFO(logger_,
            "reverse recovery done (%.2f m, rear min %.2f m) -> resuming policy (%d/3 if it re-sticks fast)",
            travelled, rear_min, failed_reverses_);
        }
        v = 0.0; d = 0.0; a0 = 0.0f; a1 = 0.0f;
      } else {
        v = -rev_speed;
        // 3-point-turn reverse: steer so the nose swings TOWARD the current waypoint
        // while backing (with v<0 the yaw rate flips sign, hence the opposite steer).
        // A straight-back reverse (d=0) never changes heading, so a robot e-stopped
        // against a wall it must turn away from re-approached the same wall forever
        // (forward -> e-stop -> straight reverse -> forward ... oscillation).
        //
        // Boxed in (obstacle in front AND behind, e.g. a tight corridor): a FIXED steer
        // angle repeats the exact same short shuffle every attempt -- no net heading
        // change accumulates, so the robot oscillates forward/back forever instead of
        // escaping at an angle like a 3-point car park. Escalate the steer angle with each
        // consecutive failed attempt (failed_reverses_, reset once a reverse makes real
        // progress) so every retry turns sharper than the last.
        double reverse_steer = std::min(STEER_SCALE, 0.40 + failed_reverses_ * 0.10);
        // use the direction locked in when this reverse started (reverse_dir_), NOT a
        // fresh gangle sign every frame -- see the comment where reverse_dir_ is set.
        d = reverse_dir_ * reverse_steer;
        a0 = 0.0f;
        a1 = 0.0f;         // previous_actions reflect "no forward command"
      }
    } else if (est) {
      // plain hold while under fail_after (or fail-fast disabled)
      v = 0.0; d = 0.0; a0 = 0.0f; a1 = 0.0f;
    }

    // Steering EMA (weight on the PREVIOUS command) to cut frame-to-frame jitter/zig-zag:
    // the raw policy output isn't a smooth analytic signal like a PID's, and the
    // steering_angle/steering_rate fed back to it (obs[1100..1101]) are reconstructed from
    // our OWN last command, not real servo feedback -- any mismatch between commanded and
    // true servo angle can otherwise induce visible wobble. Applied to the FINAL d (after
    // docking/estop/reverse overrides) so every mode benefits and steer_last_ (what gets
    // fed back next cycle) reflects what was actually smoothed/sent. Throttle is left
    // unfiltered -- only steering weaves.
    {
      double alpha = node_->get_parameter(prefix("steer_ema_alpha")).as_double();
      d = alpha * steer_smoothed_ + (1.0 - alpha) * d;
      steer_smoothed_ = d;
    }

    // --- oscillation detector: sustained left-right steering direction flips while
    // actively driving forward under the RL policy. This is NOT a frequency-domain noise
    // problem (steer_ema_alpha already passes real <1 Hz turns through on purpose) -- it
    // catches the case where the POLICY ITSELF keeps alternating its decision. No output
    // filter can fix a wrong decision, so instead of trying to smooth this away, hand off
    // to TEB (deterministic, no such failure mode) via the same BT fallback mechanism the
    // stuck-detector uses. This is distinct from stuck-detection: a robot that's weaving
    // but still net-moving forward never trips the (near-zero-motion) stuck detector above.
    if (!reversing_ && !docking && !est &&
        v > node_->get_parameter(prefix("oscillation_min_speed")).as_double()) {
      double sign = (d > 0.02) ? 1.0 : (d < -0.02) ? -1.0 : 0.0;
      if (sign != 0.0) {
        if (last_steer_sign_ != 0.0 && sign != last_steer_sign_) {
          flip_times_.push_back(now);
        }
        last_steer_sign_ = sign;
      }
      double window = node_->get_parameter(prefix("oscillation_window_sec")).as_double();
      while (!flip_times_.empty() && (now - flip_times_.front()).seconds() > window) {
        flip_times_.pop_front();
      }
      int flip_thresh = static_cast<int>(node_->get_parameter(prefix("oscillation_flip_count")).as_int());
      if (static_cast<int>(flip_times_.size()) >= flip_thresh) {
        flip_times_.clear();
        last_steer_sign_ = 0.0;
        throw nav2_core::PlannerException(
          "RL controller: sustained steering oscillation (" + std::to_string(flip_thresh) +
          "+ direction flips in " + std::to_string(window) + " s) -> handing off to TEB");
      }
    } else {
      // don't let a mode-transition edge (e.g. docking -> normal) count as a flip
      last_steer_sign_ = 0.0;
    }

    // --- heading-divergence detector: sustained large heading error to the target
    // waypoint that never shrinks, while actively driving forward. Distinct from the
    // oscillation detector above (that needs sign FLIPS; this catches the opposite
    // failure mode -- committing hard to one direction and never correcting, e.g. the
    // path is clearly to one side and the robot just keeps heading its own current way).
    // Automatically detecting and correcting a genuine steering-sign-convention mismatch
    // is NOT safe to do at runtime (it's a fixed hardware/model property, not something
    // that's "sometimes" wrong -- toggling it mid-episode would invert every subsequent
    // decision). What IS safe: recognize "not correcting toward the target" as a
    // controller failure regardless of root cause, and hand off to TEB, same as above.
    if (!reversing_ && !docking && !est &&
        v > node_->get_parameter(prefix("oscillation_min_speed")).as_double()) {
      double thresh = node_->get_parameter(prefix("heading_diverge_thresh")).as_double();
      if (std::abs(gangle) > thresh) {
        if (!heading_bad_timing_) { heading_bad_timing_ = true; heading_bad_since_ = now; }
        double diverge_sec = node_->get_parameter(prefix("heading_diverge_sec")).as_double();
        if ((now - heading_bad_since_).seconds() > diverge_sec) {
          heading_bad_timing_ = false;
          throw nav2_core::PlannerException(
            "RL controller: heading error stuck at " + std::to_string(gangle) +
            " rad for " + std::to_string(diverge_sec) + "+ s -> handing off to TEB");
        }
      } else {
        heading_bad_timing_ = false;
      }
    } else {
      heading_bad_timing_ = false;
    }

    cmd.twist.linear.x = v;
    cmd.twist.angular.z = (std::abs(v) > 1e-6) ? v * std::tan(d) / WHEELBASE : 0.0;

    prev_action_[0] = a0;
    prev_action_[1] = a1;
    steer_prev_ = steer_last_;
    steer_last_ = d;

    if (node_->get_parameter(prefix("debug")).as_bool()) {
      RCLCPP_INFO(logger_,
        "wp %zu/%zu act[%.2f %.2f] goal[d=%.2f a=%.2f] -> v=%.3f d=%.3f w=%.3f",
        wp_index_, waypoints_.size(), a0, a1, goal_dist, gangle, v, d, cmd.twist.angular.z);
    }
    return cmd;
  }

private:
  // ---- params ----
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
    steer_smoothed_ = 0.0;
    temporal_ema_.assign(N_SECTORS, 0.0f);
    ring_.clear();
    estopped_ = false;   // re-evaluated on the next control step from the fresh scan
    estop_timing_ = false;
    nomove_timing_ = false;
    reversing_ = false;
    failed_reverses_ = 0;
    have_settled_since_ = false;
    dock_reversing_ = false;
    was_docking_ = false;
    flip_times_.clear();
    last_steer_sign_ = 0.0;
    heading_bad_timing_ = false;
  }

  // min forward distance (x = r*cos b) of any return whose lateral offset |y| = r*|sin b|
  // is inside the robot's swept corridor. A frontal CONE latches on side walls the robot is
  // merely passing (a return at 0.28 m / 55 deg is not in the way of a 0.22 m-wide robot);
  // only points inside the corridor actually block forward motion.
  float frontCorridorMin(double half_width) const
  {
    const std::vector<float> & fr = frameAtOffset(0);
    float best = LIDAR_CAP;
    for (int i = 0; i < N_RAYS; ++i) {
      double b = 2.0 * M_PI * i / N_RAYS;          // ray bearing, CCW from +X
      double cb = std::cos(b);
      if (cb <= 0.0) continue;                     // rear half-plane
      float r = fr[i] * LIDAR_CAP;
      if (r < 0.05f) continue;                     // below lidar range_min -> spurious zero
      if (std::abs(r * std::sin(b)) <= half_width) {
        best = std::min(best, static_cast<float>(r * cb));
      }
    }
    return best;
  }

  // lidar e-stop with hysteresis on the forward corridor: latch when a blocker is inside
  // estop_distance, release once the corridor clears past estop_release. estop_distance <= 0
  // disables.
  bool checkEmergencyStop()
  {
    double stop_d = node_->get_parameter(prefix("estop_distance")).as_double();
    if (stop_d <= 0.0 || ring_.empty()) return false;

    // fail safe: don't drive blind on a stale/missing scan (the collision monitor polygon
    // is disabled, so this controller is the only pre-contact safety layer)
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

  // min range in the rear +/-54 deg arc of the newest resampled frame (ray 90 = straight
  // back). The inner +/-36 deg is physically blind (robot structure); the widened arc at
  // least catches diagonal-rear returns that the raw scan does see.
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

  // min range in the front-left (0..90 deg CCW) or front-right (270..360 deg, i.e. rays
  // 135..179) quadrant -- used to pick which way is actually physically open to turn into,
  // independent of which way the goal happens to be.
  float sideClearance(bool left) const
  {
    const std::vector<float> & fr = frameAtOffset(0);
    float best = LIDAR_CAP;
    int lo = left ? 0 : 135;
    int hi = left ? 45 : 179;
    for (int i = lo; i <= hi; ++i) {
      float r = fr[i] * LIDAR_CAP;
      if (r < 0.05f) continue;
      best = std::min(best, r);
    }
    return best;
  }

  // ---- lidar ----
  // Resample latest /scan to 180 rays at training bearings (ray i at i*2deg CCW from +X).
  std::vector<float> sampleScan()
  {
    std::vector<float> out(N_RAYS, 1.0f);   // default = no hit / far
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
      double ang = offset + dir * (2.0 * M_PI * i / N_RAYS);   // i*2deg
      ang = std::atan2(std::sin(ang), std::cos(ang));          // wrap [-pi,pi]
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
    const size_t need = static_cast<size_t>((N_FRAMES - 1) * FRAME_STRIDE + 1);  // 26
    while (ring_.size() > need) ring_.pop_back();
  }

  const std::vector<float> & frameAtOffset(int off) const
  {
    static const std::vector<float> kFar(N_RAYS, 1.0f);
    if (ring_.empty()) return kFar;
    size_t idx = std::min(static_cast<size_t>(off), ring_.size() - 1);
    return ring_[idx];
  }

  // diff(now, 5-steps-ago) -> per-sector min -> scale -> EMA(0.8) -> clamp(/0.05)
  // tensor_split(180,8) sectoring: first 4 sectors 23 rays, last 4 sectors 22.
  std::array<float, N_SECTORS> computeTemporal()
  {
    std::array<float, N_SECTORS> result{};
    const std::vector<float> & cur = frameAtOffset(0);
    const std::vector<float> & old = frameAtOffset(FRAME_STRIDE);

    const float max_diff = MAX_APPROACH * (FRAME_STRIDE * STEP_DT) / LIDAR_CAP;  // 0.041667
    const float scale = 0.05f / std::max(max_diff, 1e-9f);                        // = 1.2

    int base = N_RAYS / N_SECTORS;   // 22
    int rem = N_RAYS % N_SECTORS;    // 4
    int start = 0;
    for (int s = 0; s < N_SECTORS; ++s) {
      int len = base + (s < rem ? 1 : 0);   // 23,23,23,23,22,22,22,22
      float mn = std::numeric_limits<float>::infinity();
      for (int k = start; k < start + len; ++k) mn = std::min(mn, cur[k] - old[k]);
      float scaled = mn * scale;
      temporal_ema_[s] = TEMPORAL_EMA * temporal_ema_[s] + (1.0f - TEMPORAL_EMA) * scaled;
      result[s] = clampf(temporal_ema_[s] / 0.05f, -1.0f, 1.0f);
      start += len;
    }
    return result;
  }

  // ---- carrot (discrete waypoints) ----
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

  // pure-pursuit advance (matches training rewards.waypoint_reward): a via-point clears when the
  // robot captures it (dist < 0.7) or passes it (closer to the next waypoint). At most one advance
  // per control step. The final waypoint is never advanced past -> Nav2's goal checker ends the run.
  void advanceWaypoint(const Vec2 & robot)
  {
    size_t n = waypoints_.size();
    if (n == 0) return;
    size_t i = std::min(wp_index_, n - 1);
    if (i + 1 >= n) return;                        // on the final waypoint: hold

    // Re-sync to the truly closest waypoint within a small FORWARD-ONLY window (never
    // behind, so it can't un-advance or oscillate backward). The path itself stays fixed
    // (no replanning -- see the BT comment on KeepPathOrReplan), but during a turn the
    // robot's actual position can end up geometrically closer to a slightly different
    // point than the rigid "current or next only" check assumes; searching a few
    // waypoints ahead keeps the tracked target locked to where the robot really is
    // without ever touching/regenerating the global path.
    size_t window_end = std::min(i + WP_RESYNC_WINDOW, n - 1);
    size_t best = i;
    double best_dist = norm2({robot.x - waypoints_[i].x, robot.y - waypoints_[i].y});
    for (size_t k = i + 1; k <= window_end; ++k) {
      double d = norm2({robot.x - waypoints_[k].x, robot.y - waypoints_[k].y});
      if (d < best_dist) { best_dist = d; best = k; }
    }
    wp_index_ = best;
    i = wp_index_;
    if (i + 1 >= n) return;

    double dist = norm2({robot.x - waypoints_[i].x, robot.y - waypoints_[i].y});
    double dist_next = norm2({robot.x - waypoints_[i + 1].x, robot.y - waypoints_[i + 1].y});
    bool captured = dist < WP_CAPTURE_RADIUS;
    bool passed = dist_next < dist;
    if (captured || passed) wp_index_ = i + 1;
  }

  rclcpp::Logger logger_{rclcpp::get_logger("rl_controller")};
  rclcpp_lifecycle::LifecycleNode::SharedPtr node_;
  std::string plugin_name_;

  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
  sensor_msgs::msg::LaserScan::SharedPtr last_scan_;
  std::mutex scan_mtx_;

  std::vector<Vec2> waypoints_;
  size_t wp_index_{0};
  Vec2 last_goal_{0, 0};
  Vec2 last_robot_{0, 0};
  bool have_goal_{false};

  std::deque<std::vector<float>> ring_;
  std::vector<float> temporal_ema_;

  float prev_action_[2]{0.0f, 0.0f};
  double steer_last_{0.0}, steer_prev_{0.0};
  double steer_smoothed_{0.0};
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
  rclcpp::Time last_scan_time_;
  bool dock_reversing_{false};
  bool was_docking_{false};
  rclcpp::Time last_time_;
  bool have_last_time_{false};
  std::deque<rclcpp::Time> flip_times_;
  double last_steer_sign_{0.0};
  bool heading_bad_timing_{false};
  rclcpp::Time heading_bad_since_;

  std::vector<float> h_;

  std::unique_ptr<Ort::Env> env_;
  std::unique_ptr<Ort::Session> session_;
  std::unique_ptr<Ort::SessionOptions> session_options_;
};

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(RLController, nav2_core::Controller)
