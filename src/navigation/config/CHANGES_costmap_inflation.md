# Change: costmap inflation increased

File: `navigation/config/nav2_params.yaml`

## Global costmap (line 214-216)
| Param | Before | After |
|---|---|---|
| `cost_scaling_factor` | 10.0 | 5.5 |
| `inflation_radius` | 0.25 | 0.35 |

## Local costmap (line 147-149)
| Param | Before | After |
|---|---|---|
| `cost_scaling_factor` | 5.0 | 5.0 (unchanged) |
| `inflation_radius` | 0.25 | 0.35 |

## Reason
Planned paths were cutting closer to obstacles (0.25 m buffer) than the RL
controller's own safety margin (0.30 m `estop_distance`). Path looked "valid"
to the planner but tripped the controller's independent check, so the robot
hesitated. Widening the inflation buffer to 0.35 m and softening the global
cost falloff makes the planner keep more clearance, matching what the
controller already expects.

Rebuilt: `colcon build --packages-select navigation`

cd /home/ubuntu/ros2_ws && source /opt/ros/humble/setup.zsh 2>/dev/null; colcon build --packages-select navigation 2>&1 | tail -10

cd /home/ubuntu/ros2_ws && source /opt/ros/humble/setup.zsh 2>/dev/null; colcon build --packages-select rl_nav_cpp 2>&1 | tail -3