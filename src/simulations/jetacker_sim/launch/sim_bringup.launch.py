import os
from ament_index_python.packages import get_package_share_directory

from launch_ros.actions import Node
from launch.conditions import IfCondition
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource

# Starts the robot description (TF only, no Gazebo) + the lightweight kinematic/raycast
# sim_node + RViz (using navigation's own rviz config, which has Map/costmap/BT displays
# that jetacker_description's bare view.rviz lacks). No use_sim_time -- sim_node runs on
# plain wall-clock time, there's no stepped simulator/clock to bridge.

def launch_setup(context):
    compiled = os.environ['need_compile']
    if compiled == 'True':
        jetacker_description_path = get_package_share_directory('jetacker_description')
        navigation_package_path = get_package_share_directory('navigation')
    else:
        jetacker_description_path = '/home/ubuntu/ros2_ws/src/simulations/jetacker_description'
        navigation_package_path = '/home/ubuntu/ros2_ws/src/navigation'

    map_yaml = LaunchConfiguration(
        'map', default='/home/ubuntu/ros2_ws/src/slam/maps/map_01.yaml').perform(context)
    init_x = LaunchConfiguration('init_x', default='0.0')
    init_y = LaunchConfiguration('init_y', default='0.0')
    init_yaw = LaunchConfiguration('init_yaw', default='0.0')
    use_rviz = LaunchConfiguration('use_rviz', default='true')

    declare_use_rviz_cmd = DeclareLaunchArgument('use_rviz', default_value=use_rviz)

    # jetacker_description's own launch defaults use_gui/use_rviz/use_sim_time to 'true' --
    # override all three: no joint_state_publisher_gui popup, no its own bare-bones rviz,
    # and no sim-time (we're not bridging a /clock).
    robot_description_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(jetacker_description_path, 'launch', 'robot_description.launch.py')),
        launch_arguments={
            'use_gui': 'false',
            'use_rviz': 'false',
            'use_sim_time': 'false',
        }.items())

    sim_node = Node(
        package='jetacker_sim',
        executable='sim_node',
        name='jetacker_sim_node',
        output='screen',
        parameters=[{
            'map_yaml': map_yaml,
            'init_x': init_x,
            'init_y': init_y,
            'init_yaw': init_yaw,
        }],
        remappings=[('/tf', 'tf'), ('/tf_static', 'tf_static')])

    rviz_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(navigation_package_path, 'launch', 'rviz_navigation.launch.py')),
        condition=IfCondition(use_rviz))

    return [declare_use_rviz_cmd, robot_description_launch, sim_node, rviz_launch]


def generate_launch_description():
    return LaunchDescription([
        OpaqueFunction(function=launch_setup)
    ])
