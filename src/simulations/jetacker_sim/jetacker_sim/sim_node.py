"""Lightweight 2D kinematic + raycast simulator, standing in for the real robot's
lidar/odometry/servo hardware chain so the Nav2 stack (AMCL, costmaps, planner,
bt_navigator, our RL controller plugins, BT recovery structure) can run completely
unchanged against a synthetic sensor/actuation source.

Subscribes /cmd_vel (the final topic the real servo hardware also consumes, after
controller_server -> velocity_smoother -> collision_monitor), integrates plain unicycle
kinematics, and publishes /odom + TF (odom -> base_footprint) + a synthetic LaserScan
(via 2D raycasting against a nav2_map_server-format map) on both /scan_raw (read raw by
the RL controllers) and /scan (read by AMCL).
"""
import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Twist, TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from tf2_ros import TransformBroadcaster

from jetacker_sim.map_loader import load_map


def wrap_to_pi(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def yaw_to_quat(yaw):
    return math.sin(yaw / 2.0), math.cos(yaw / 2.0)  # (z, w) for a pure-yaw rotation


class SimNode(Node):

    def __init__(self):
        super().__init__('jetacker_sim_node')

        self.declare_parameter('map_yaml', '/home/ubuntu/ros2_ws/src/slam/maps/map_01.yaml')
        self.declare_parameter('init_x', 0.0)
        self.declare_parameter('init_y', 0.0)
        self.declare_parameter('init_yaw', 0.0)
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('lidar_frame', 'lidar_sim_frame')
        # base_footprint -> lidar_link translation, from jetacker.urdf.xacro's lidar_joint
        self.declare_parameter('lidar_offset_x', 0.10414)
        self.declare_parameter('lidar_offset_y', 0.000685)
        self.declare_parameter('lidar_yaw_offset', math.pi)  # lidar_sim_frame_joint rpy
        self.declare_parameter('odom_rate_hz', 50.0)
        self.declare_parameter('scan_rate_hz', 10.0)
        self.declare_parameter('range_min', 0.15)
        self.declare_parameter('range_max', 4.0)
        self.declare_parameter('num_samples', 400)
        self.declare_parameter('cmd_vel_timeout', 0.5)
        self.declare_parameter('unknown_is_occupied', False)

        map_yaml = self.get_parameter('map_yaml').value
        self.occ, self.resolution, self.origin = load_map(map_yaml)
        self.map_h, self.map_w = self.occ.shape
        self.get_logger().info(
            f'loaded map {map_yaml}: {self.map_w}x{self.map_h} px @ {self.resolution} m/px')

        self.odom_frame = self.get_parameter('odom_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.lidar_frame = self.get_parameter('lidar_frame').value
        self.lidar_dx = self.get_parameter('lidar_offset_x').value
        self.lidar_dy = self.get_parameter('lidar_offset_y').value
        self.lidar_yaw_offset = self.get_parameter('lidar_yaw_offset').value
        self.range_min = self.get_parameter('range_min').value
        self.range_max = self.get_parameter('range_max').value
        self.num_samples = int(self.get_parameter('num_samples').value)
        self.cmd_vel_timeout = self.get_parameter('cmd_vel_timeout').value
        self.unknown_is_occupied = self.get_parameter('unknown_is_occupied').value

        self.x = self.get_parameter('init_x').value
        self.y = self.get_parameter('init_y').value
        self.yaw = self.get_parameter('init_yaw').value
        self.v = 0.0
        self.omega = 0.0

        now = self.get_clock().now()
        self.last_cmd_time = now
        self._last_odom_time = now

        # precompute ray geometry once
        self.ray_angles_local = np.linspace(-math.pi, math.pi, self.num_samples, endpoint=False)
        self.ray_steps = np.arange(self.range_min, self.range_max, self.resolution * 0.5)
        self.angle_increment = 2.0 * math.pi / self.num_samples

        # 2026-08-18: nav2's collision_monitor now outputs to /controller/cmd_vel instead of
        # /cmd_vel (the real driver hard-clamps /cmd_vel to 0.2 m/s -- see cmd_vel_out_topic
        # in nav2_params.yaml). Subscribe to both so the sim works either way.
        # OLD: self.cmd_sub = self.create_subscription(Twist, '/cmd_vel', self.on_cmd_vel, 10)
        self.cmd_sub = self.create_subscription(Twist, '/cmd_vel', self.on_cmd_vel, 10)
        self.cmd_sub_ctrl = self.create_subscription(
            Twist, '/controller/cmd_vel', self.on_cmd_vel, 10)
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.scan_raw_pub = self.create_publisher(LaserScan, '/scan_raw', qos_profile_sensor_data)
        self.scan_pub = self.create_publisher(LaserScan, '/scan', qos_profile_sensor_data)
        self.tf_broadcaster = TransformBroadcaster(self)

        odom_rate = self.get_parameter('odom_rate_hz').value
        scan_rate = self.get_parameter('scan_rate_hz').value
        self.odom_timer = self.create_timer(1.0 / odom_rate, self.on_odom_timer)
        self.scan_timer = self.create_timer(1.0 / scan_rate, self.on_scan_timer)

        self.get_logger().info(
            f'jetacker_sim ready: start pose ({self.x:.2f}, {self.y:.2f}, {self.yaw:.2f})')

    def on_cmd_vel(self, msg: Twist):
        self.v = msg.linear.x
        self.omega = msg.angular.z
        self.last_cmd_time = self.get_clock().now()

    def on_odom_timer(self):
        now = self.get_clock().now()
        dt = (now - self._last_odom_time).nanoseconds * 1e-9
        self._last_odom_time = now
        if dt <= 0.0:
            return

        stale = (now - self.last_cmd_time).nanoseconds * 1e-9 > self.cmd_vel_timeout
        v, omega = (0.0, 0.0) if stale else (self.v, self.omega)

        self.x += v * math.cos(self.yaw) * dt
        self.y += v * math.sin(self.yaw) * dt
        self.yaw = wrap_to_pi(self.yaw + omega * dt)

        qz, qw = yaw_to_quat(self.yaw)

        odom = Odometry()
        odom.header.stamp = now.to_msg()
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id = self.base_frame
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        odom.twist.twist.linear.x = v
        odom.twist.twist.angular.z = omega
        self.odom_pub.publish(odom)

        tf = TransformStamped()
        tf.header.stamp = now.to_msg()
        tf.header.frame_id = self.odom_frame
        tf.child_frame_id = self.base_frame
        tf.transform.translation.x = self.x
        tf.transform.translation.y = self.y
        tf.transform.rotation.z = qz
        tf.transform.rotation.w = qw
        self.tf_broadcaster.sendTransform(tf)

    def on_scan_timer(self):
        ranges = self._raycast()
        now = self.get_clock().now().to_msg()

        scan = LaserScan()
        scan.header.frame_id = self.lidar_frame
        scan.header.stamp = now
        scan.angle_min = -math.pi
        scan.angle_max = math.pi - self.angle_increment
        scan.angle_increment = self.angle_increment
        scan.range_min = self.range_min
        scan.range_max = self.range_max
        scan.ranges = ranges.tolist()

        self.scan_raw_pub.publish(scan)

        scan2 = LaserScan()
        scan2.header.frame_id = scan.header.frame_id
        scan2.header.stamp = now
        scan2.angle_min = scan.angle_min
        scan2.angle_max = scan.angle_max
        scan2.angle_increment = scan.angle_increment
        scan2.range_min = scan.range_min
        scan2.range_max = scan.range_max
        scan2.ranges = list(scan.ranges)
        self.scan_pub.publish(scan2)

    def _raycast(self):
        """Vectorized 2D raycast from the current pose against the loaded occupancy grid.
        Returns a (num_samples,) float array; +inf where no return (REP-117)."""
        lx = self.x + self.lidar_dx * math.cos(self.yaw) - self.lidar_dy * math.sin(self.yaw)
        ly = self.y + self.lidar_dx * math.sin(self.yaw) + self.lidar_dy * math.cos(self.yaw)

        world_angles = self.yaw + self.lidar_yaw_offset + self.ray_angles_local  # (N,)
        cos_a = np.cos(world_angles)[:, None]
        sin_a = np.sin(world_angles)[:, None]
        xs = lx + cos_a * self.ray_steps[None, :]   # (N, K)
        ys = ly + sin_a * self.ray_steps[None, :]

        cols = ((xs - self.origin[0]) / self.resolution).astype(np.int64)
        rows = (self.map_h - 1 - (ys - self.origin[1]) / self.resolution).astype(np.int64)
        in_bounds = (rows >= 0) & (rows < self.map_h) & (cols >= 0) & (cols < self.map_w)

        occupied = np.zeros_like(in_bounds)
        occupied[in_bounds] = self.occ[rows[in_bounds], cols[in_bounds]]
        if self.unknown_is_occupied:
            occupied = occupied | (~in_bounds)

        hit = occupied.any(axis=1)
        first_idx = np.argmax(occupied, axis=1)
        ranges = np.where(hit, self.ray_steps[first_idx], np.inf)
        return ranges


def main(args=None):
    rclpy.init(args=args)
    node = SimNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
