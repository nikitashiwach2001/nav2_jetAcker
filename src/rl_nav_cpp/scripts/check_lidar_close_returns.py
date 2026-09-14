#!/usr/bin/env python3
"""Decide whether lidar_min_range (currently 0.22 m) is still needed.

Why: the RL controller masks every return closer than lidar_min_range to "far", so it cannot
see anything in that band at all. That was added when the lidar sat low enough to see the
robot's own structure. Since the lidar was remounted high for a 360 deg view the mask may no
longer be masking anything real -- but it DOES create a blind band 0.024 m past the front
bumper and 0.09 m past the sides, which is exactly where an object that "almost touches" the
robot sits. In that band the controller sees a clear corridor, keeps commanding forward, and
the collision monitor silently zeroes the command: the robot freezes without ever e-stopping.

Lowering the mask blind is NOT safe: if the lidar really does see the chassis, a lower mask
makes the robot e-stop permanently. This tells you which it is.

HOW TO USE
  1. Park the robot in OPEN space -- nothing within 1 m in any direction.
  2. Leave it completely stationary.
  3. python3 check_lidar_close_returns.py

A self-return shows up as the same ray reporting the same short range in every single scan
(near-zero spread). A real object that happens to be nearby varies by at least a few mm.
"""
import math
import statistics
import sys
import time
from collections import defaultdict

import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan

CURRENT_MIN_RANGE = 0.22
N_SCANS = 40
BAND = 0.35            # only care about returns closer than this


def main():
    rclpy.init()
    node = rclpy.create_node('lidar_close_return_check')
    scans = []
    node.create_subscription(LaserScan, '/scan_raw', lambda m: scans.append(m),
                             qos_profile_sensor_data)
    print(f'collecting {N_SCANS} scans from /scan_raw (robot must be STILL) ...')
    end = time.time() + 20
    while time.time() < end and len(scans) < N_SCANS:
        rclpy.spin_once(node, timeout_sec=0.2)
    if len(scans) < 5:
        print('not enough scans. Is the lidar publishing?  ros2 topic hz /scan_raw')
        rclpy.shutdown()
        return 1

    n = len(scans[0].ranges)
    per_ray = defaultdict(list)
    for m in scans:
        for i, r in enumerate(m.ranges):
            if math.isfinite(r) and 0.02 < r < BAND:
                per_ray[i].append(r)

    a0, inc = scans[0].angle_min, scans[0].angle_increment
    print(f'\n{len(scans)} scans, {n} rays each\n')
    if not per_ray:
        print(f'NO returns closer than {BAND} m at all.')
        print(f'  -> the lidar sees none of its own structure.')
        print(f'  -> lidar_min_range {CURRENT_MIN_RANGE} is masking nothing real, and is')
        print(f'     costing you a blind band right where near-contact objects sit.')
        print(f'  -> SAFE to lower it (0.12 suggested).')
        rclpy.shutdown()
        return 0

    persistent, occasional = [], []
    for i, vals in sorted(per_ray.items()):
        seen = len(vals) / len(scans)
        spread = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        ang = math.degrees(a0 + i * inc)
        row = (i, ang, statistics.median(vals), spread, seen)
        (persistent if seen > 0.9 and spread < 0.01 else occasional).append(row)

    if persistent:
        print('PERSISTENT close returns  (same ray, same range, every scan)')
        print('  -> this is the robot\'s own structure; the mask IS doing a job\n')
        print(f'  {"ray":>5} {"angle":>8} {"range":>7} {"spread":>7}')
        for i, ang, med, sd, _ in persistent:
            print(f'  {i:5d} {ang:7.1f}d {med:6.3f}m {sd:6.4f}m')
        need = max(m for _, _, m, _, _ in persistent) + 0.02
        print(f'\n  -> lidar_min_range must stay at least {need:.2f}')
        print(f'  -> currently {CURRENT_MIN_RANGE}: ' +
              ('fine, leave it.' if CURRENT_MIN_RANGE >= need
               else f'TOO LOW, raise to {need:.2f}'))
        if need < CURRENT_MIN_RANGE - 0.02:
            print(f'  -> but {CURRENT_MIN_RANGE} is higher than needed; {need:.2f} would')
            print(f'     recover {CURRENT_MIN_RANGE - need:.3f} m of blind band.')
    else:
        print(f'No PERSISTENT close returns ({len(occasional)} occasional ones, which are')
        print('real nearby objects or noise, not structure).')
        print(f'  -> lidar_min_range {CURRENT_MIN_RANGE} is masking nothing structural.')
        print(f'  -> SAFE to lower it (0.12 suggested).')

    rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
