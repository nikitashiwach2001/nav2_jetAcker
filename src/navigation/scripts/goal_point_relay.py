#!/usr/bin/env python3
"""Heading-free goal sending: click with RViz "Publish Point" instead of "Nav2 Goal".

Humble's SmacPlannerHybrid always plans to the goal's EXACT heading (no
goal_heading_mode), and with the forward-only DUBIN motion model a heading that
doesn't match the natural approach direction forces a full minimum-radius loop
(diameter ~0.7 m) into the path - even though the goal checker accepts any
final heading (yaw_goal_tolerance 3.15).

This node listens to /clicked_point (RViz "Publish Point" tool) and picks the
goal heading FOR you: it asks the planner (ComputePathToPose) to test-plan the
straight-line robot->goal heading first, and if that path is loopy (much longer
than the beeline) it tries rotated candidates and keeps the shortest plan. The
winning heading is then sent as a normal /goal_pose for bt_navigator.

The Nav2 Goal arrow tool still works unchanged when a specific heading is wanted.
"""
import math

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.action import ActionClient
from geometry_msgs.msg import PointStamped, PoseStamped
from nav2_msgs.action import ComputePathToPose

import tf2_ros

ROBOT_FRAME = 'base_footprint'
# candidate heading offsets (rad) relative to the straight robot->goal bearing,
# in preference order; only tried when the straight heading plans badly
OFFSETS = [0.0, 0.52, -0.52, 1.05, -1.05, 1.57, -1.57, 2.36, -2.36, 3.14]
GOOD_RATIO = 1.25    # plan <= this x beeline distance -> accept immediately
PLAN_TIMEOUT = 4.0   # s per candidate (max_planning_time is 3.0)


def path_length(path):
    pts = path.poses
    return sum(math.hypot(pts[i+1].pose.position.x - pts[i].pose.position.x,
                          pts[i+1].pose.position.y - pts[i].pose.position.y)
               for i in range(len(pts) - 1))


class GoalPointRelay(Node):
    def __init__(self):
        super().__init__('goal_point_relay')
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.goal_pub = self.create_publisher(PoseStamped, 'goal_pose', 1)
        self.planner = ActionClient(self, ComputePathToPose, 'compute_path_to_pose')
        self.create_subscription(PointStamped, 'clicked_point', self.on_click, 1)
        self.get_logger().info('goal_point_relay ready: use the RViz "Publish Point" tool '
                               'to send heading-free navigation goals')

    def make_goal(self, msg, yaw):
        goal = PoseStamped()
        goal.header.frame_id = msg.header.frame_id or 'map'
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.position.x = msg.point.x
        goal.pose.position.y = msg.point.y
        goal.pose.orientation.z = math.sin(yaw / 2.0)
        goal.pose.orientation.w = math.cos(yaw / 2.0)
        return goal

    def try_plan(self, goal_pose):
        """Plan to goal_pose; return path length or None."""
        req = ComputePathToPose.Goal()
        req.goal = goal_pose
        req.use_start = False   # plan from current robot pose
        fut = self.planner.send_goal_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=PLAN_TIMEOUT)
        gh = fut.result()
        if gh is None or not gh.accepted:
            return None
        rfut = gh.get_result_async()
        rclpy.spin_until_future_complete(self, rfut, timeout_sec=PLAN_TIMEOUT)
        res = rfut.result()
        if res is None or not res.result.path.poses:
            return None
        return path_length(res.result.path)

    def on_click(self, msg: PointStamped):
        frame = msg.header.frame_id or 'map'
        try:
            tf = self.tf_buffer.lookup_transform(frame, ROBOT_FRAME,
                                                 rclpy.time.Time(),
                                                 timeout=Duration(seconds=0.5))
        except Exception as e:
            self.get_logger().error(f'cannot get robot pose in "{frame}": {e} - goal ignored')
            return
        rx = tf.transform.translation.x
        ry = tf.transform.translation.y
        beeline = math.hypot(msg.point.x - rx, msg.point.y - ry)
        bearing = math.atan2(msg.point.y - ry, msg.point.x - rx)

        best_yaw, best_len = bearing, None
        if self.planner.wait_for_server(timeout_sec=1.0):
            for off in OFFSETS:
                yaw = bearing + off
                length = self.try_plan(self.make_goal(msg, yaw))
                if length is None:
                    continue
                if best_len is None or length < best_len:
                    best_yaw, best_len = yaw, length
                if length <= beeline * GOOD_RATIO:
                    break   # short enough - no loop, stop searching
            if best_len is not None:
                self.get_logger().info(
                    f'best heading {math.degrees(best_yaw):.0f} deg: path {best_len:.2f} m '
                    f'(beeline {beeline:.2f} m)')
            else:
                self.get_logger().warn('no candidate heading produced a plan - '
                                       'sending straight-line heading anyway')
        else:
            self.get_logger().warn('planner action server unavailable - '
                                   'sending straight-line heading without testing')

        self.goal_pub.publish(self.make_goal(msg, best_yaw))
        self.get_logger().info(f'goal ({msg.point.x:.2f}, {msg.point.y:.2f}) sent with '
                               f'heading {math.degrees(best_yaw):.0f} deg')


def main():
    rclpy.init()
    rclpy.spin(GoalPointRelay())


if __name__ == '__main__':
    main()
