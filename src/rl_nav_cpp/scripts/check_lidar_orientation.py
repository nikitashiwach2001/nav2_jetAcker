#!/usr/bin/env python3
"""Work out what lidar_angle_offset the RL controller should be using.

Why: the controller maps its training rays onto the raw scan itself --
    scan_angle(i) = lidar_angle_offset + dir * (2*pi*i/180)
with training ray 0 defined as the robot's FORWARD (+X) direction. If the lidar is
remounted at a different rotation and that offset is not updated, the policy sees the
whole world rotated: the forward corridor e-stop watches somewhere else, rearArcMin()
watches somewhere else, and obstacle avoidance steers the wrong way. This has bitten this
robot before (the 180-deg yaw mount was found only after a lot of confusing behaviour).

HOW TO USE
  1. Put a single object (box, chair, wall) roughly 1 m DIRECTLY IN FRONT of the robot,
     with nothing else that close in any other direction.
  2. Run this script.
  3. It prints the lidar_angle_offset that makes training ray 0 point at that object,
     i.e. forward.

    python3 check_lidar_orientation.py

It also reports any blind wedge (a run of inf/no-return), which is how you confirm the
new high mounting really does see all 360 deg.
"""
import math
import sys

import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan

CURRENT_OFFSET = 3.14159265   # what nav2_controller_rl_v5.yaml has today


def main():
    rclpy.init()
    node = rclpy.create_node('lidar_orientation_check')
    got = {}

    def cb(msg):
        if 'msg' not in got:
            got['msg'] = msg

    node.create_subscription(LaserScan, '/scan_raw', cb, qos_profile_sensor_data)
    print('waiting for /scan_raw ...')
    import time
    end = time.time() + 10
    while time.time() < end and 'msg' not in got:
        rclpy.spin_once(node, timeout_sec=0.2)
    if 'msg' not in got:
        print('no scan received. Is the lidar publishing?  ros2 topic hz /scan_raw')
        rclpy.shutdown()
        return 1

    m = got['msg']
    n = len(m.ranges)
    print(f'\nscan: {n} points, angle_min={m.angle_min:.3f}, '
          f'increment={m.angle_increment:.5f} rad, frame={m.header.frame_id}')

    # --- blind wedge: longest run of non-finite returns ---
    valid = [math.isfinite(r) and r > 0.05 for r in m.ranges]
    best_len = best_start = 0
    cur_len = cur_start = 0
    for i in range(2 * n):            # wrap around
        if not valid[i % n]:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
            if cur_len > best_len:
                best_len, best_start = cur_len, cur_start
        else:
            cur_len = 0
    print(f'valid returns: {sum(valid)}/{n}')
    if best_len > n * 0.05:
        a0 = m.angle_min + (best_start % n) * m.angle_increment
        a1 = a0 + best_len * m.angle_increment
        print(f'BLIND WEDGE: {best_len} rays = {math.degrees(best_len*m.angle_increment):.0f} deg, '
              f'scan angles {math.degrees(a0):.0f}..{math.degrees(a1):.0f} deg')
        print('  -> something still occludes the lidar over that arc.')
    else:
        print('no significant blind wedge -> full 360 deg view.')

    # --- nearest return = the object you placed in front ---
    best_i, best_r = None, float('inf')
    for i, r in enumerate(m.ranges):
        if math.isfinite(r) and 0.05 < r < best_r:
            best_r, best_i = r, i
    if best_i is None:
        print('\nno valid return at all -- cannot determine orientation.')
        rclpy.shutdown()
        return 1

    scan_angle = m.angle_min + best_i * m.angle_increment
    scan_angle = math.atan2(math.sin(scan_angle), math.cos(scan_angle))  # wrap to +/-pi

    print(f'\nnearest object: {best_r:.2f} m at scan index {best_i} '
          f'(scan angle {math.degrees(scan_angle):.1f} deg)')
    print('\n' + '=' * 62)
    print('If that object is DIRECTLY IN FRONT of the robot, then:')
    print(f'   lidar_angle_offset should be:  {scan_angle:.6f}   '
          f'({math.degrees(scan_angle):.1f} deg)')
    print(f'   config currently has:          {CURRENT_OFFSET:.6f}   '
          f'({math.degrees(CURRENT_OFFSET):.1f} deg)')
    diff = abs(math.atan2(math.sin(scan_angle - CURRENT_OFFSET),
                          math.cos(scan_angle - CURRENT_OFFSET)))
    if diff < math.radians(15):
        print('   -> MATCHES. No change needed.')
    else:
        print(f'   -> OFF BY {math.degrees(diff):.0f} deg. The config MUST be updated,')
        print('      otherwise the policy sees the world rotated.')
    print('=' * 62)
    rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
