#!/usr/bin/env python3
"""Send navigation goals WITHOUT RViz -- runs headless on the robot.

Why this exists: RViz is only a GUI viewed over NoMachine/SSH. When the robot drives
far and the WiFi gets weak, that GUI freezes (and if the whole SSH session drops, a
launch started from that terminal dies with it). The navigation itself runs fine on the
robot -- only the remote GUI is fragile. This script lets you drive the robot to preset
map coordinates from a plain terminal (ideally inside tmux, so it survives disconnects),
with no RViz and no GUI at all.

Goals are published on /goal_pose, i.e. through scripts/goal_point_relay.py, so they get
the same loop-free heading search RViz goals get (see that file). Progress is then tracked
by watching the map->base_footprint transform, so it works with no action-client plumbing.

Usage:
  send_goal.py <name> [name ...]   send preset goals, one after another (waits for each)
  send_goal.py --xy X Y [YAW]      send raw map coordinates (yaw in degrees, optional)
  send_goal.py --list              list saved presets
  send_goal.py --save <name>       save the robot's CURRENT pose as a preset
  send_goal.py --delete <name>     remove a preset
  send_goal.py --where             just print where the robot thinks it is
  send_goal.py --cancel            STOP navigation right now (cancels the active goal)

Options:
  --tolerance M    arrival radius for the progress report (default 0.4, matches
                   nav2_controller_rl_v5.yaml's xy_goal_tolerance)
  --timeout S      give up reporting after this long (default 300); navigation itself
                   is NOT cancelled, only this script stops watching
  --no-wait        send the goal and exit immediately, don't track progress

Presets live in navigation/config/goal_presets.yaml and are plain map-frame coordinates,
so they stay valid as long as you keep using the same map.
"""
import argparse
import math
import os
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from geometry_msgs.msg import PoseStamped
from action_msgs.srv import CancelGoal
import tf2_ros
import yaml

PRESETS_FILE = '/home/ubuntu/ros2_ws/src/navigation/config/goal_presets.yaml'
MAP_FRAME = 'map'
ROBOT_FRAME = 'base_footprint'
GOAL_TOPIC = 'goal_pose'          # -> goal_point_relay -> /goal_pose_nav -> bt_navigator


def load_presets():
    if not os.path.exists(PRESETS_FILE):
        return {}
    with open(PRESETS_FILE) as f:
        data = yaml.safe_load(f) or {}
    return data.get('goals', {}) or {}


def save_presets(goals):
    os.makedirs(os.path.dirname(PRESETS_FILE), exist_ok=True)
    with open(PRESETS_FILE, 'w') as f:
        yaml.safe_dump({'goals': goals}, f, default_flow_style=False, sort_keys=True)


class GoalSender(Node):

    def __init__(self):
        super().__init__('send_goal')
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.pub = self.create_publisher(PoseStamped, GOAL_TOPIC, 1)

    def robot_pose(self, timeout=5.0):
        """Return (x, y, yaw_rad) of the robot in the map frame, or None."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            try:
                tf = self.tf_buffer.lookup_transform(
                    MAP_FRAME, ROBOT_FRAME, rclpy.time.Time(),
                    timeout=Duration(seconds=0.2))
            except Exception:
                continue
            q = tf.transform.rotation
            yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                             1.0 - 2.0 * (q.y * q.y + q.z * q.z))
            return (tf.transform.translation.x, tf.transform.translation.y, yaw)
        return None

    def send(self, x, y, yaw_rad):
        # the relay subscribes with depth 1; publishing before it has matched us would
        # silently drop the goal, so wait for the subscription to appear first
        deadline = time.time() + 5.0
        while self.pub.get_subscription_count() == 0 and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        if self.pub.get_subscription_count() == 0:
            self.get_logger().warn(
                f'nobody is subscribed to /{GOAL_TOPIC} -- is goal_point_relay.py running? '
                'Sending anyway, but the goal will most likely be dropped.')

        msg = PoseStamped()
        msg.header.frame_id = MAP_FRAME
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.orientation.z = math.sin(yaw_rad / 2.0)
        msg.pose.orientation.w = math.cos(yaw_rad / 2.0)
        self.pub.publish(msg)
        # give the message a moment to actually leave before we (possibly) exit
        for _ in range(10):
            rclpy.spin_once(self, timeout_sec=0.05)

    def cancel_all(self):
        """Cancel whatever goal bt_navigator is currently running.

        We are not the client that sent the goal, so we can't use a goal handle. Instead
        we call the action's own cancel service with an all-zero goal_id + zero stamp,
        which the action server treats as "cancel everything".
        """
        cli = self.create_client(CancelGoal, '/navigate_to_pose/_action/cancel_goal')
        if not cli.wait_for_service(timeout_sec=5.0):
            print('navigate_to_pose cancel service not available -- '
                  'is the navigation stack running?')
            return False
        req = CancelGoal.Request()   # zeroed uuid + zero stamp == cancel all
        fut = cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=5.0)
        res = fut.result()
        if res is None:
            print('cancel request timed out')
            return False
        n = len(res.goals_canceling)
        if n:
            print(f'cancelled {n} active goal(s) -- robot should stop')
        else:
            print('no active goal was running (nothing to cancel)')
        return True

    def watch(self, x, y, tolerance, timeout):
        """Print distance-to-goal until arrival or timeout. Returns True if arrived."""
        start = time.time()
        last_print = 0.0
        print(f'watching progress (tolerance {tolerance:.2f} m, timeout {timeout:.0f} s) '
              '-- Ctrl+C to stop watching, navigation keeps running')
        while time.time() - start < timeout:
            rclpy.spin_once(self, timeout_sec=0.2)
            pose = self.robot_pose(timeout=1.0)
            if pose is None:
                if time.time() - last_print > 2.0:
                    print('  ... no map->base_footprint transform yet '
                          '(is localization up?)')
                    last_print = time.time()
                continue
            d = math.hypot(x - pose[0], y - pose[1])
            if d <= tolerance:
                print(f'ARRIVED: {d:.2f} m from goal '
                      f'(took {time.time() - start:.0f} s)')
                return True
            if time.time() - last_print > 2.0:
                print(f'  {d:5.2f} m to go   (robot at {pose[0]:.2f}, {pose[1]:.2f})')
                last_print = time.time()
        print(f'stopped watching after {timeout:.0f} s -- navigation is still running')
        return False


def main():
    ap = argparse.ArgumentParser(
        description='Send Nav2 goals from a terminal, no RViz needed.')
    ap.add_argument('name', nargs='*',
                    help='preset goal name(s); several run in sequence, '
                         'each waiting for arrival before the next')
    ap.add_argument('--xy', nargs='+', type=float, metavar=('X', 'Y'),
                    help='raw map coordinates: --xy X Y [YAW_DEGREES]')
    ap.add_argument('--list', action='store_true', help='list saved presets')
    ap.add_argument('--save', metavar='NAME', help="save robot's current pose as a preset")
    ap.add_argument('--delete', metavar='NAME', help='delete a preset')
    ap.add_argument('--where', action='store_true', help='print current robot pose')
    ap.add_argument('--cancel', action='store_true',
                    help='cancel the goal the robot is currently driving to')
    ap.add_argument('--tolerance', type=float, default=0.4)
    ap.add_argument('--timeout', type=float, default=300.0)
    ap.add_argument('--no-wait', action='store_true')
    args = ap.parse_args()

    goals = load_presets()

    if args.list:
        if not goals:
            print(f'no presets yet ({PRESETS_FILE})')
            print('drive the robot somewhere, then:  send_goal.py --save <name>')
        else:
            print(f'presets in {PRESETS_FILE}:')
            for n, g in sorted(goals.items()):
                yaw_txt = f", yaw {math.degrees(g.get('yaw', 0.0)):.0f} deg"
                print(f"  {n:<20} x={g['x']:.2f}  y={g['y']:.2f}{yaw_txt}")
        return 0

    if args.delete:
        if args.delete not in goals:
            print(f"no preset named '{args.delete}'")
            return 1
        del goals[args.delete]
        save_presets(goals)
        print(f"deleted '{args.delete}'")
        return 0

    rclpy.init()
    node = GoalSender()
    try:
        if args.where or args.save:
            pose = node.robot_pose()
            if pose is None:
                print(f'could not read {MAP_FRAME} -> {ROBOT_FRAME} transform. '
                      'Is the navigation stack (and localization) running?')
                return 1
            x, y, yaw = pose
            print(f'robot is at x={x:.3f}  y={y:.3f}  yaw={math.degrees(yaw):.1f} deg')
            if args.save:
                goals[args.save] = {'x': round(x, 3), 'y': round(y, 3),
                                    'yaw': round(yaw, 4)}
                save_presets(goals)
                print(f"saved as preset '{args.save}'")
            return 0

        if args.cancel:
            node.cancel_all()
            return 0

        # build the list of (label, x, y, yaw) to run, in order
        queue = []
        if args.xy:
            if len(args.xy) < 2:
                print('--xy needs at least X and Y')
                return 1
            queue.append(('--xy', args.xy[0], args.xy[1],
                          math.radians(args.xy[2]) if len(args.xy) > 2 else None))
        elif args.name:
            for nm in args.name:
                if nm not in goals:
                    print(f"no preset named '{nm}'. Known: "
                          f"{', '.join(sorted(goals)) or '(none)'}")
                    return 1
            for nm in args.name:
                g = goals[nm]
                queue.append((nm, g['x'], g['y'], g.get('yaw')))
        else:
            ap.print_help()
            return 1

        total = len(queue)
        for i, (label, x, y, yaw) in enumerate(queue, 1):
            if total > 1:
                print(f'\n----- goal {i}/{total}: {label} -----')
            # No heading given -> aim from the robot's current position toward the goal.
            # goal_point_relay re-tests and replaces this if it plans a loopy path anyway.
            if yaw is None:
                pose = node.robot_pose(timeout=3.0)
                yaw = math.atan2(y - pose[1], x - pose[0]) if pose else 0.0

            print(f'sending goal: x={x:.2f}  y={y:.2f}  yaw={math.degrees(yaw):.0f} deg')
            node.send(x, y, yaw)
            print('goal sent.')

            if args.no_wait:
                continue
            arrived = node.watch(x, y, args.tolerance, args.timeout)
            if not arrived and i < total:
                print('did not arrive in time -- stopping the sequence here. '
                      'Re-run with the remaining goals when ready.')
                return 1
        if total > 1 and not args.no_wait:
            print(f'\nall {total} goals done.')
        return 0
    except KeyboardInterrupt:
        print('\nstopped watching. Navigation is STILL RUNNING -- '
              'use  send_goal.py --cancel  to actually stop the robot.')
        return 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    sys.exit(main())
