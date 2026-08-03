import os
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription,
                            OpaqueFunction, ExecuteProcess, TimerAction)

# Standalone desktop debug tool: boots the lightweight jetacker_sim simulator (no Gazebo,
# no real hardware) + the SAME Nav2 stack used on the real robot (bringup_v5.launch.py,
# unchanged) against the same map (map_01) already used for real-robot v5 testing. Exists
# so BT/recovery/controller LOGIC bugs can be reproduced and iterated on without the real
# robot -- no HOST/MASTER env vars or namespace push needed, unlike navigation_v5.launch.py,
# since this never runs on the actual fleet.
#
# Usage: ros2 launch navigation navigation_sim.launch.py use_rl_v5:=true

def launch_setup(context):
    compiled = os.environ['need_compile']
    if compiled == 'True':
        slam_package_path = get_package_share_directory('slam')
        navigation_package_path = get_package_share_directory('navigation')
        jetacker_sim_path = get_package_share_directory('jetacker_sim')
    else:
        slam_package_path = '/home/ubuntu/ros2_ws/src/slam'
        navigation_package_path = '/home/ubuntu/ros2_ws/src/navigation'
        jetacker_sim_path = '/home/ubuntu/ros2_ws/src/simulations/jetacker_sim'

    map_name = LaunchConfiguration('map', default='map_01').perform(context)
    use_rl_v5 = LaunchConfiguration('use_rl_v5', default='true').perform(context)
    init_x = LaunchConfiguration('init_x', default='0.0')
    init_y = LaunchConfiguration('init_y', default='0.0')
    init_yaw = LaunchConfiguration('init_yaw', default='0.0')
    use_rviz = LaunchConfiguration('use_rviz', default='true')

    map_name_arg = DeclareLaunchArgument('map', default_value=map_name)
    use_rl_v5_arg = DeclareLaunchArgument('use_rl_v5', default_value=use_rl_v5)
    init_x_arg = DeclareLaunchArgument('init_x', default_value=init_x)
    init_y_arg = DeclareLaunchArgument('init_y', default_value=init_y)
    init_yaw_arg = DeclareLaunchArgument('init_yaw', default_value=init_yaw)
    use_rviz_arg = DeclareLaunchArgument('use_rviz', default_value=use_rviz)

    map_yaml_path = os.path.join(slam_package_path, 'maps', map_name + '.yaml')

    sim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(jetacker_sim_path, 'launch', 'sim_bringup.launch.py')),
        launch_arguments={
            'map': map_yaml_path,
            'init_x': init_x,
            'init_y': init_y,
            'init_yaw': init_yaw,
            'use_rviz': use_rviz,
        }.items())

    goal_point_relay = ExecuteProcess(
        cmd=['python3', os.path.join(navigation_package_path, 'scripts/goal_point_relay.py')],
        output='screen',
    )

    # bringup_v5.launch.py declares sensible defaults for everything else (namespace='',
    # use_sim_time='false', autostart='true', params_file=navigation/config/nav2_params.yaml)
    # -- only override what actually needs to differ from the real-robot launch: which map,
    # and which controller. use_sim_time stays 'false': the sim runs on wall-clock time.
    nav2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(navigation_package_path, 'launch/include/bringup_v5.launch.py')),
        launch_arguments={
            'map': map_yaml_path,
            'use_rl_v5': use_rl_v5,
        }.items())

    return [map_name_arg, use_rl_v5_arg, init_x_arg, init_y_arg, init_yaw_arg, use_rviz_arg,
            sim_launch, goal_point_relay,
            # short delay so sim_node/robot_state_publisher are up (TF, /odom, /scan)
            # before AMCL/costmaps do their first TF lookups
            TimerAction(period=3.0, actions=[nav2_launch])]


def generate_launch_description():
    return LaunchDescription([
        OpaqueFunction(function=launch_setup)
    ])
