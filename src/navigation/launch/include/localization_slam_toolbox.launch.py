import os
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable, OpaqueFunction

# Parallel copy of localization.launch.py that uses slam_toolbox's LocalizationSlamToolbox
# (deserializing the re-mapped slam/maps/map_01_v2.data/.posegraph pose-graph) instead of
# map_server + amcl. Same launch-argument interface as localization.launch.py (namespace,
# use_sim_time, autostart, container_name, map, params_file) so it's a drop-in swap in the
# bringup chain -- the `map` arg is accepted but unused (slam_toolbox uses map_file_name
# from slam_toolbox_localization.yaml instead of a plain .yaml/.pgm).
#
# NOT lifecycle-managed: slam_toolbox::SlamToolbox (the base class behind
# LocalizationSlamToolbox) is a plain rclcpp::Node, not a LifecycleNode -- it has no
# ~/change_state service for nav2_lifecycle_manager to call. So there is no
# lifecycle_manager_localization here (nothing to manage).
#
# STANDALONE NODE, NOT A COMPOSABLE NODE (2026-08-03): loading
# slam_toolbox::LocalizationSlamToolbox into nav2_container silently did nothing --
# the node came up with only the 5 default rclcpp params (qos_overrides/use_sim_time),
# no mode/map_file_name/frames, no /map, no map->odom TF. Reason: SlamToolbox does its
# real init in configure() + loadPoseGraphByParams() (see
# /opt/ros/humble/include/slam_toolbox/slam_toolbox_common.hpp), which the standalone
# localization_slam_toolbox_node main() calls after construction but a component
# container never does -- it only constructs. So it must run as its own process.

def launch_setup(context):
    use_sim_time = LaunchConfiguration('use_sim_time')
    params_file = LaunchConfiguration(
        'params_file',
        default=os.path.join(get_package_share_directory('navigation'),
                             'config/slam_toolbox_localization.yaml')).perform(context)
    # container_name is still declared below (interface parity with localization.launch.py)
    # but unused: this runs as its own process, not inside nav2_container -- see header.

    remappings = [('/tf', 'tf'),
                  ('/tf_static', 'tf_static')]

    stdout_linebuf_envvar = SetEnvironmentVariable(
        'RCUTILS_LOGGING_BUFFERED_STREAM', '1')

    declare_namespace_cmd = DeclareLaunchArgument(
        'namespace', default_value='', description='Top-level namespace')

    declare_map_yaml_cmd = DeclareLaunchArgument(
        'map', default_value='',
        description='Unused here (slam_toolbox uses map_file_name from params_file) -- '
                     'accepted only so this is a drop-in swap for localization.launch.py')

    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time', default_value='false',
        description='Use simulation (Gazebo) clock if true')

    declare_params_file_cmd = DeclareLaunchArgument(
        'params_file',
        default_value=os.path.join(get_package_share_directory('navigation'),
                                   'config/slam_toolbox_localization.yaml'),
        description='Full path to the slam_toolbox localization params file')

    declare_autostart_cmd = DeclareLaunchArgument(
        'autostart', default_value='true',
        description='Unused here (no lifecycle manager) -- accepted for interface parity')

    declare_container_name_cmd = DeclareLaunchArgument(
        'container_name', default_value='nav2_container',
        description='the name of container that nodes will load in if use composition')

    declare_use_namespace_cmd = DeclareLaunchArgument(
        'use_namespace', default_value='false')

    slam_toolbox_node = Node(
        package='slam_toolbox',
        executable='localization_slam_toolbox_node',
        name='slam_toolbox',
        output='screen',
        parameters=[params_file, {'use_sim_time': use_sim_time}],
        remappings=remappings)

    return [
        stdout_linebuf_envvar,
        declare_namespace_cmd,
        declare_use_namespace_cmd,
        declare_map_yaml_cmd,
        declare_use_sim_time_cmd,
        declare_params_file_cmd,
        declare_autostart_cmd,
        declare_container_name_cmd,
        slam_toolbox_node,
    ]

def generate_launch_description():
    return LaunchDescription([
        OpaqueFunction(function=launch_setup)
    ])
