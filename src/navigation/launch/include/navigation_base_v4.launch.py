import os
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch_ros.actions import LoadComposableNodes
from launch_ros.descriptions import ComposableNode
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable, OpaqueFunction

# Parallel copy of navigation_base_v3.launch.py that can select the v4 RL controller
# plugin (rl_nav_cpp/RLControllerV4, config/nav2_controller_rl_v4.yaml, model_6998
# checkpoint) via use_rl_v4:=true, without touching the v1/v2/v3 files.

def launch_setup(context):
    compiled = os.environ['need_compile']
    if compiled == 'True':
        navigation_package_path = get_package_share_directory('navigation')
    else:
        navigation_package_path = '/home/ubuntu/ros2_ws/src/navigation'

    use_namespace = LaunchConfiguration('use_namespace').perform(context)
    use_teb = LaunchConfiguration('use_teb', default='false').perform(context)
    use_rl = LaunchConfiguration('use_rl', default='false').perform(context)
    use_rl_v2 = LaunchConfiguration('use_rl_v2', default='false').perform(context)
    use_rl_v3 = LaunchConfiguration('use_rl_v3', default='false').perform(context)
    use_rl_v4 = LaunchConfiguration('use_rl_v4', default='false').perform(context)
    namespace = LaunchConfiguration('namespace')
    use_sim_time = LaunchConfiguration('use_sim_time')
    autostart = LaunchConfiguration('autostart')
    params_file = LaunchConfiguration('params_file')
    container_name = LaunchConfiguration('container_name')
    container_name_full = (namespace, container_name)

    lifecycle_nodes = ['controller_server',
                       'smoother_server',
                       'planner_server',
                       'behavior_server',
                       'bt_navigator',
                       'waypoint_follower',
                       'velocity_smoother',
                       'collision_monitor']

    remappings = [('/tf', 'tf'),
                  ('/tf_static', 'tf_static')]

    stdout_linebuf_envvar = SetEnvironmentVariable(
        'RCUTILS_LOGGING_BUFFERED_STREAM', '1')

    declare_namespace_cmd = DeclareLaunchArgument(
        'namespace',
        default_value='',
        description='Top-level namespace')

    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation (Gazebo) clock if true')

    declare_params_file_cmd = DeclareLaunchArgument(
        'params_file',
        default_value=os.path.join(navigation_package_path, 'config/nav2_params.yaml'),
        description='Full path to the ROS2 parameters file to use for all launched nodes')

    declare_autostart_cmd = DeclareLaunchArgument(
        'autostart', default_value='true',
        description='Automatically startup the nav2 stack')

    declare_container_name_cmd = DeclareLaunchArgument(
        'container_name', default_value='nav2_container',
        description='the name of conatiner that nodes will load in if use composition')

    declare_use_namespace_cmd = DeclareLaunchArgument(
        'use_namespace', default_value='false')

    declare_use_teb_cmd = DeclareLaunchArgument(
        'use_teb', default_value='true')

    declare_use_rl_cmd = DeclareLaunchArgument(
        'use_rl', default_value='false',
        description='Use the RL v1 (ONNX) Ackermann controller instead of TEB/DWB')

    declare_use_rl_v2_cmd = DeclareLaunchArgument(
        'use_rl_v2', default_value='false',
        description='Use the RL v2 (ONNX, reverse-capable) Ackermann controller')

    declare_use_rl_v3_cmd = DeclareLaunchArgument(
        'use_rl_v3', default_value='false',
        description='Use the RL v3 (ONNX, carrot lookahead) Ackermann controller. '
                     'Takes priority over use_rl_v2/use_rl/use_teb if true.')

    declare_use_rl_v4_cmd = DeclareLaunchArgument(
        'use_rl_v4', default_value='false',
        description='Use the RL v4 (ONNX, model_6998 checkpoint, carrot lookahead) '
                     'Ackermann controller. Takes priority over use_rl_v3/v2/use_rl/use_teb if true.')

    if use_rl_v4 == 'true':
        controller_param = os.path.join(navigation_package_path, 'config/nav2_controller_rl_v4.yaml')
    elif use_rl_v3 == 'true':
        controller_param = os.path.join(navigation_package_path, 'config/nav2_controller_rl_v3.yaml')
    elif use_rl_v2 == 'true':
        controller_param = os.path.join(navigation_package_path, 'config/nav2_controller_rl_v2.yaml')
    elif use_rl == 'true':
        controller_param = os.path.join(navigation_package_path, 'config/nav2_controller_rl.yaml')
    elif use_teb == 'true':
        controller_param = os.path.join(navigation_package_path, 'config/nav2_controller_teb.yaml')
    else:
        controller_param = os.path.join(navigation_package_path, 'config/nav2_controller_dwb.yaml')


    load_composable_nodes = LoadComposableNodes(
        target_container=container_name_full,
        composable_node_descriptions=[
            ComposableNode(
                package='nav2_controller',
                plugin='nav2_controller::ControllerServer',
                name='controller_server',
                parameters=[controller_param],
                remappings=remappings + [('cmd_vel', 'cmd_vel_nav')]),
            ComposableNode(
                package='nav2_smoother',
                plugin='nav2_smoother::SmootherServer',
                name='smoother_server',
                parameters=[params_file],
                remappings=remappings),
            ComposableNode(
                package='nav2_planner',
                plugin='nav2_planner::PlannerServer',
                name='planner_server',
                parameters=[params_file],
                remappings=remappings),
            ComposableNode(
                package='nav2_behaviors',
                plugin='behavior_server::BehaviorServer',
                name='behavior_server',
                parameters=[params_file],
                remappings=remappings),
            ComposableNode(
                package='nav2_bt_navigator',
                plugin='nav2_bt_navigator::BtNavigator',
                name='bt_navigator',
                parameters=[params_file],
                remappings=remappings),
            ComposableNode(
                package='nav2_waypoint_follower',
                plugin='nav2_waypoint_follower::WaypointFollower',
                name='waypoint_follower',
                parameters=[params_file],
                remappings=remappings),
            ComposableNode(
                package='nav2_velocity_smoother',
                plugin='nav2_velocity_smoother::VelocitySmoother',
                name='velocity_smoother',
                parameters=[params_file],
                remappings=remappings + [('cmd_vel', 'cmd_vel_nav')]),
            ComposableNode(
                package='nav2_collision_monitor',
                plugin='nav2_collision_monitor::CollisionMonitor',
                name='collision_monitor',
                parameters=[params_file],
                remappings=remappings),
            ComposableNode(
                package='nav2_lifecycle_manager',
                plugin='nav2_lifecycle_manager::LifecycleManager',
                name='lifecycle_manager_navigation',
                parameters=[{'use_sim_time': use_sim_time,
                             'autostart': autostart,
                             'node_names': lifecycle_nodes}]),
        ],
    )

    return [stdout_linebuf_envvar,
            declare_namespace_cmd,
            declare_use_namespace_cmd,
            declare_use_sim_time_cmd,
            declare_params_file_cmd,
            declare_autostart_cmd,
            declare_container_name_cmd,
            declare_use_teb_cmd,
            declare_use_rl_cmd,
            declare_use_rl_v2_cmd,
            declare_use_rl_v3_cmd,
            declare_use_rl_v4_cmd,
            load_composable_nodes]

def generate_launch_description():
    return LaunchDescription([
        OpaqueFunction(function = launch_setup)
    ])

if __name__ == '__main__':
    ld = generate_launch_description()

    ls = LaunchService()
    ls.include_launch_description(ld)
    ls.run()
