import os
from glob import glob
from setuptools import setup

package_name = 'example'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('example', '**/*.launch.py'))),
        (os.path.join('share', package_name, 'config'), glob(os.path.join('config', '*.yaml'))),
        (os.path.join('share', package_name, 'resource'), glob(os.path.join('resource', '*.dae'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ubuntu',
    maintainer_email='1270161395@qq.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'color_detect = example.color_detect.color_detect_node:main',
            'color_track = example.color_track.color_track_node:main',
            'hand_trajectory = example.hand_trajectory.hand_trajectory_node:main',
            'hand_track = example.hand_track.hand_track_node:main',
            'hand_detect = example.hand_track.hand_detect_node:main',
            'body_and_rgb_control = example.body_control.include.body_and_rgb_control:main',
            'body_control = example.body_control.include.body_control:main',
            'body_track = example.body_control.include.body_track:main',
            'fall_down_detect = example.body_control.include.fall_down_detect:main',
            'self_driving = example.self_driving.self_driving:main',
            'yolov5_node = example.yolov5_detect.yolov5_node:main',
            "yolo_detect_demo = example.yolo_detect.yolo_detect_demo:main",
            "yolo_node = example.yolo_detect.yolo_node:main",
        ],
    },
)
