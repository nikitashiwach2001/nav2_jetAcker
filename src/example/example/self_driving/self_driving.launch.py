import os
from ament_index_python.packages import get_package_share_directory

from launch_ros.actions import Node
from launch import LaunchDescription, LaunchService
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, OpaqueFunction

def launch_setup(context):
    compiled = os.environ['need_compile']
    machine_type = os.environ.get('MACHINE_TYPE')
    start = LaunchConfiguration('start', default='true')
    start_arg = DeclareLaunchArgument('start', default_value=start)
    only_line_follow = LaunchConfiguration('only_line_follow', default='false')
    only_line_follow_arg = DeclareLaunchArgument('only_line_follow', default_value=only_line_follow)

    model_choice = LaunchConfiguration('model', default='yolo26')
    model_arg = DeclareLaunchArgument('model', default_value=model_choice)
    model_choice_str = model_choice.perform(context)

    if compiled == 'True':
        peripherals_package_path = get_package_share_directory('peripherals')
        controller_package_path = get_package_share_directory('controller')
    else:
        peripherals_package_path = '/home/ubuntu/ros2_ws/src/peripherals'
        controller_package_path = '/home/ubuntu/ros2_ws/src/driver/controller'

    depth_camera_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(peripherals_package_path, 'launch/depth_camera.launch.py')),
    )
    controller_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(controller_package_path, 'launch/controller.launch.py')),
    )

    yolov5_node = Node(
        package='example',
        executable='yolov5_node',
        output='screen',
        parameters=[{'classes': ['go', 'right', 'park', 'red', 'green', 'crosswalk']},
                    {'use_depth': True, 'engine': 'traffic_signs_640s_7_0.engine', 'lib': 'libmyplugins.so', 'conf': 0.70}]
    )

    if 'v11' in model_choice_str:
        engine = 'best_traffic_11'
    elif '26' in model_choice_str:
        engine = 'best_traffic_26'
    
    yolo_node = Node(
        package='example',
        executable='yolo_node',
        output='screen',
        parameters=[{'image_topic': '/depth_cam/rgb/image_raw', 'classes': ['go', 'right', 'park', 'red', 'green', 'crosswalk'], 'engine': engine, 'conf': 0.3, 'task': 'detect', 'display': False}]
    )

    self_driving_node = Node(
        package='example',
        executable='self_driving',
        output='screen',
        parameters=[{'start': start, 'only_line_follow': only_line_follow, 'use_depth_cam': True}],
    )

    return [start_arg,
            only_line_follow_arg,
            model_arg,
            depth_camera_launch,
            controller_launch,
            yolo_node,
            self_driving_node,
            ]

def generate_launch_description():
    return LaunchDescription([
        OpaqueFunction(function = launch_setup)
    ])

if __name__ == '__main__':
    # 创建一个LaunchDescription对象(create a LaunchDescription object)
    ld = generate_launch_description()

    ls = LaunchService()
    ls.include_launch_description(ld)
    ls.run()

