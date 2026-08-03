import os
from ament_index_python.packages import get_package_share_directory

from launch_ros.actions import PushRosNamespace
from launch import LaunchDescription, LaunchService
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, GroupAction, OpaqueFunction, TimerAction, ExecuteProcess

# Parallel copy of navigation_v4.launch.py that boots the v5 RL controller stack
# (rl_nav_cpp/RLControllerV5, config/nav2_controller_rl_v5.yaml,
# policy_v5_model10496.onnx -- literal copy of v1, only the model differs) via
# bringup_v5.launch.py. Does not touch the v1/v2/v3/v4 files.
#
# Usage: ros2 launch navigation navigation_v5.launch.py use_rl_v5:=true

def launch_setup(context):
    compiled = os.environ['need_compile']
    if compiled == 'True':
        slam_package_path = get_package_share_directory('slam')
        navigation_package_path = get_package_share_directory('navigation')
    else:
        slam_package_path = '/home/ubuntu/ros2_ws/src/slam'
        navigation_package_path = '/home/ubuntu/ros2_ws/src/navigation'

    sim = LaunchConfiguration('sim', default='false').perform(context)
    map_name = LaunchConfiguration('map', default='map_01').perform(context)
    robot_name = LaunchConfiguration('robot_name', default=os.environ['HOST']).perform(context)
    master_name = LaunchConfiguration('master_name', default=os.environ['MASTER']).perform(context)
    use_teb = LaunchConfiguration('use_teb', default='true').perform(context)
    use_rl = LaunchConfiguration('use_rl', default='false').perform(context)
    use_rl_v2 = LaunchConfiguration('use_rl_v2', default='false').perform(context)
    use_rl_v3 = LaunchConfiguration('use_rl_v3', default='false').perform(context)
    use_rl_v4 = LaunchConfiguration('use_rl_v4', default='false').perform(context)
    use_rl_v5 = LaunchConfiguration('use_rl_v5', default='true').perform(context)

    sim_arg = DeclareLaunchArgument('sim', default_value=sim)
    map_name_arg = DeclareLaunchArgument('map', default_value=map_name)
    master_name_arg = DeclareLaunchArgument('master_name', default_value=master_name)
    robot_name_arg = DeclareLaunchArgument('robot_name', default_value=robot_name)
    use_teb_arg = DeclareLaunchArgument('use_teb', default_value=use_teb)
    use_rl_arg = DeclareLaunchArgument('use_rl', default_value=use_rl)
    use_rl_v2_arg = DeclareLaunchArgument('use_rl_v2', default_value=use_rl_v2)
    use_rl_v3_arg = DeclareLaunchArgument('use_rl_v3', default_value=use_rl_v3)
    use_rl_v4_arg = DeclareLaunchArgument('use_rl_v4', default_value=use_rl_v4)
    use_rl_v5_arg = DeclareLaunchArgument('use_rl_v5', default_value=use_rl_v5)

    use_sim_time = 'true' if sim == 'true' else 'false'
    use_namespace = 'true' if robot_name != '/' else 'false'

    base_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(slam_package_path, 'launch/include/robot.launch.py')),
        launch_arguments={
            'sim': sim,
            'master_name': master_name,
            'robot_name': robot_name
        }.items(),
    )

    # same heading-free goal relay as v1/v2 -- unrelated to controller choice, reused as-is
    goal_point_relay = ExecuteProcess(
        cmd=['python3', os.path.join(navigation_package_path, 'scripts/goal_point_relay.py')],
        output='screen',
    )

    navigation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(navigation_package_path, 'launch/include/bringup_v5.launch.py')),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'map': os.path.join(slam_package_path, 'maps', map_name + '.yaml'),
            'params_file': os.path.join(navigation_package_path, 'config', 'nav2_params.yaml'),
            'namespace': robot_name,
            'use_namespace': use_namespace,
            'autostart': 'true',
            'use_teb': use_teb,
            'use_rl': use_rl,
            'use_rl_v2': use_rl_v2,
            'use_rl_v3': use_rl_v3,
            'use_rl_v4': use_rl_v4,
            'use_rl_v5': use_rl_v5,
        }.items(),
    )

    bringup_launch = GroupAction(
     actions=[
         PushRosNamespace(robot_name),
         base_launch,
         goal_point_relay,
         TimerAction(
             period=10.0,
             actions=[navigation_launch],
         ),
      ]
    )

    return [sim_arg, map_name_arg, master_name_arg, robot_name_arg, use_teb_arg, use_rl_arg,
            use_rl_v2_arg, use_rl_v3_arg, use_rl_v4_arg, use_rl_v5_arg, bringup_launch]

def generate_launch_description():
    return LaunchDescription([
        OpaqueFunction(function = launch_setup)
    ])

if __name__ == '__main__':
    ld = generate_launch_description()

    ls = LaunchService()
    ls.include_launch_description(ld)
    ls.run()
