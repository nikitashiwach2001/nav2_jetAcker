#!/usr/bin/env python3
"""Measure how much this robot's odometry drifts, independently of SLAM.

Why this matters here: there are no wheel encoders. /odom comes from rf2o (laser scan
matching) fused by the EKF, and slam_toolbox then does scan matching of its own on top. So
both layers share a single failure mode -- in a repetitive space they can drift together with
nothing to contradict them. Every slam_toolbox parameter assumes the odometry prior is roughly
right; if it is not, no amount of tuning fixes the map.

TWO TESTS

  straight   Drive the robot in a straight line, stop, and compare what odometry thinks it
             travelled against what you measured with a tape.
                 python3 check_odom_drift.py straight
             Good: within ~2%. Bad: >5% -- scale error, every map will be stretched.

  loop       Drive the robot in a loop and return it to EXACTLY the same spot and heading
             (mark the floor first). Odometry should come back to near zero.
                 python3 check_odom_drift.py loop
             Good: <2% of distance travelled. Bad: >5% -- that error is what loop closure has
             to absorb, and it is why loop_search_maximum_distance had to keep changing.

Press Ctrl-C when the robot is back / stopped.
"""
import math
import sys

import rclpy
from nav_msgs.msg import Odometry
from rclpy.qos import qos_profile_sensor_data


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'loop'
    if mode not in ('straight', 'loop'):
        print(__doc__)
        return 1

    rclpy.init()
    node = rclpy.create_node('odom_drift_check')
    st = {'first': None, 'last': None, 'path': 0.0, 'prev': None}

    def cb(m):
        p = m.pose.pose
        cur = (p.position.x, p.position.y, yaw_of(p.orientation))
        if st['first'] is None:
            st['first'] = cur
            print('start latched -- now drive. Ctrl-C when done.')
        if st['prev'] is not None:
            st['path'] += math.hypot(cur[0] - st['prev'][0], cur[1] - st['prev'][1])
        st['prev'] = cur
        st['last'] = cur

    node.create_subscription(Odometry, '/odom', cb, qos_profile_sensor_data)
    print(f'mode: {mode}   waiting for /odom ...')
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    if st['first'] is None or st['last'] is None:
        print('\nno odometry received. Is the robot bringup running?  ros2 topic hz /odom')
        rclpy.shutdown()
        return 1

    x0, y0, a0 = st['first']
    x1, y1, a1 = st['last']
    dx, dy = x1 - x0, y1 - y0
    straight = math.hypot(dx, dy)
    da = math.degrees(abs(math.atan2(math.sin(a1 - a0), math.cos(a1 - a0))))

    print(f'\n  path length driven (odom) : {st["path"]:.3f} m')
    print(f'  start -> end displacement : {straight:.3f} m')
    print(f'  heading change            : {da:.1f} deg')

    if mode == 'straight':
        print('\n  Measure the real distance with a tape and compare to the displacement above.')
        print('  error % = (odom - real) / real * 100')
        print('    within 2%  : fine')
        print('    over  5%   : scale error -- maps will come out stretched or squashed')
    else:
        err = 100.0 * straight / st['path'] if st['path'] > 0.1 else float('nan')
        print(f'\n  closure error : {straight:.3f} m over {st["path"]:.2f} m driven = {err:.1f}%')
        print(f'  heading error : {da:.1f} deg')
        print('    under 2%   : odometry is fine, the map problem is elsewhere')
        print('    2-5%       : borderline; keep mapping loops short')
        print('    over 5%    : odometry is the bottleneck -- no slam_toolbox setting fixes this')
        print('\n  (this assumes you returned the robot to the SAME spot and heading)')
    rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
