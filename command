#关闭所有ros节点(stop all ROS nodes)
~/.stop_ros.sh

#编译ros2_ws(compile ros2_ws)
cd ~/ros2_ws && ~/.build.sh
#colcon build --event-handlers  console_direct+  --cmake-args  -DCMAKE_BUILD_TYPE=Release --symlink-install
#单独编译某个包(compile a package independently)
colcon build --event-handlers  console_direct+  --cmake-args  -DCMAKE_BUILD_TYPE=Release --symlink-install --packages-select xxx

#######calibration#######
#imu校准(IMU calibration)
ros2 launch ros_robot_controller ros_robot_controller.launch.py
ros2 run imu_calib do_calib --ros-args -r imu:=/ros_robot_controller/imu_raw --param output_file:=/home/ubuntu/ros2_ws/src/calibration/config/imu_calib.yaml

#查看imu校准效果(check IMU calibration effect)
ros2 launch peripherals imu_view.launch.py

#深度摄像头点云可视化(depth camera point-cloud visualization)
#深度摄像头RGB图像可视化(depth camera RGB image visualization)
ros2 launch peripherals depth_camera.launch.py
rviz2

#雷达数据可视化(lidar data visualization)
ros2 launch peripherals lidar_view.launch.py

#######app#######
#雷达功能(lidar function)
ros2 launch app lidar_node.launch.py debug:=true
#雷达避障(lidar obstacle avoidance)
ros2 service call /lidar_app/enter std_srvs/srv/Trigger {}
ros2 service call /lidar_app/set_running interfaces/srv/SetInt64 "{data: 1}"
#关闭
ros2 service call /lidar_app/exit std_srvs/srv/Trigger {}

#雷达跟随(lidar following)
ros2 service call /lidar_app/enter std_srvs/srv/Trigger {}
ros2 service call /lidar_app/set_running interfaces/srv/SetInt64 "{data: 2}"
#关闭(exit)
ros2 service call /lidar_app/exit std_srvs/srv/Trigger {}

#巡线(line following)
ros2 launch app line_following_node.launch.py debug:=true
ros2 service call /line_following/enter std_srvs/srv/Trigger {}
#鼠标左键点击画面取色(left-click image to pick color)
ros2 service call /line_following/set_running std_srvs/srv/SetBool "{data: True}"

#目标跟踪(target tracking)
ros2 launch app object_tracking_node.launch.py debug:=true
ros2 service call /object_tracking/enter std_srvs/srv/Trigger {}
#鼠标左键点击画面取色(left-click image to pick color)
ros2 service call /object_tracking/set_running std_srvs/srv/SetBool "{data: True}"

#ar
ros2 launch app ar_app_node.launch.py debug:=true
ros2 service call /ar_app/enter std_srvs/srv/Trigger {}
ros2 service call /ar_app/set_model interfaces/srv/SetString "{data: \"bicycle\"}"

#######example#######
#二维码生成(QR code creation)
cd ~/ros2_ws/src/example/example/qrcode && python3 qrcode_creater.py
#####################
ros2 launch peripherals depth_camera.launch.py

#二维码检测(QR code detection)
cd ~/ros2_ws/src/example/example/qrcode && python3 qrcode_detecter.py

#人脸检测(face detection)
cd ~/ros2_ws/src/example/example/mediapipe_example && python3 face_detect.py

#人脸网格(face mesh)
cd ~/ros2_ws/src/example/example/mediapipe_example && python3 face_mesh.py

#手关键点检测(hand keypoint detection)
cd ~/ros2_ws/src/example/example/mediapipe_example && python3 hand.py

#肢体关键点检测(body keypoint detection)
cd ~/ros2_ws/src/example/example/mediapipe_example && python3 pose.py

#背景分割(background segmentation)
cd ~/ros2_ws/src/example/example/mediapipe_example && python3 self_segmentation.py

#整体检测(integral detection)
cd ~/ros2_ws/src/example/example/mediapipe_example && python3 holistic.py

#3D物体检测(3D object detection)
cd ~/ros2_ws/src/example/example/mediapipe_example && python3 objectron.py

#指尖轨迹(fingertip trajectory)
cd ~/ros2_ws/src/example/example/mediapipe_example && python3 hand_gesture.py

#颜色识别(color recognition)
cd ~/ros2_ws/src/example/example/color_detect && python3 color_detect_demo.py
#####################
#肢体姿态控制(body posture control)
ros2 launch example body_control.launch.py

#肢体姿态融合RGB控制(body posture and RGB control)
ros2 launch example body_and_rgb_control.launch.py

#人体跟踪(body tracking)
ros2 launch example body_track.launch.py

#跌倒检测(falling down detection)
ros2 launch example fall_down_detect.launch.py

#颜色追踪(color tracking)
ros2 launch example color_track_node.launch.py

#手部跟踪(hand tracking)
ros2 launch example hand_track_node.launch.py

#指尖轨迹(fingertip trajectory)
ros2 launch example hand_trajectory_node.launch.py

#无人驾驶(autonomous driving)
ros2 launch example self_driving.launch.py
#######slam#######
#2D建图(2D mapping)
ros2 launch slam slam.launch.py

#rviz查看建图效果(rviz checking mapping effect)
ros2 launch slam rviz_slam.launch.py

#键盘控制(可选)(keyboard control (optional))
ros2 launch peripherals teleop_key_control.launch.py

#保存地图(save map)
#/home/ubuntu/ros2_ws/src/slam/maps/map_01.yaml
cd ~/ros2_ws/src/slam/maps && ros2 run nav2_map_server map_saver_cli -f "map_01" --ros-args -p map_subscribe_transient_local:=true

#cd ~/ros2_ws/src/slam/maps && ros2 run nav2_map_server map_saver_cli -f "保存名称" --ros-args -p map_subscribe_transient_local:=true -r __ns:=/robot_1

#3D建图(Dabai)(3D mapping (Dabai))
ros2 launch slam rtabmap_slam.launch.py

#rviz查看建图效果(rviz checking mapping effect)
ros2 launch slam rviz_rtabmap.launch.py

#键盘控制(可选)(keyboard control (optional))
ros2 launch peripherals teleop_key_control.launch.py

#######navigation#######
#2D导航(2D navigation)
##rviz发布导航目标(rviz publishing navigation target)
ros2 launch navigation rviz_navigation.launch.py
ros2 launch navigation navigation.launch.py map:=地图名称

#3D导航(3D navigation)
ros2 launch navigation rtabmap_navigation.launch.py

#rviz发布导航目标(rviz publishing navigation target)
ros2 launch navigation rviz_rtabmap_navigation.launch.py

#######simulations#######
#urdf可视化(urdf visualization)
ros2 launch jetacker_description display.launch.py

#######xf_mic_asr_offline#######
#语音控制颜色识别(voice control color recognition)
ros2 launch xf_mic_asr_offline voice_control_color_detect.launch.py

#语音控制颜色跟踪(voice control color tracking)
ros2 launch xf_mic_asr_offline voice_control_color_track.launch.py

#语音控制移动(voice control movement)
ros2 launch xf_mic_asr_offline voice_control_move.launch.py

#语音控制导航(voice control navigation)
ros2 launch xf_mic_asr_offline voice_control_navigation.launch.py map:='地图名称'

########software
ros2 launch peripherals depth_camera.launch.py

#lab_tool
python3 ~/software/lab_tool/main.py

#collect_picture
python3 ~/software/collect_picture/main.py

#servo_tool
python3 ~/software/servo_tool/main.py

#######large_models_examples#######
#前进2s, 右转2s, 然后后退
#手动触发
#ros2 topic pub --once /vocal_detect/asr_result std_msgs/msg/String '{data: "前进2s, 右转2s, 然后后退"}'
ros2 launch large_models_examples llm_control_move.launch.py

#追踪红色物体
ros2 launch large_models_examples llm_color_track.launch.py

#沿着黑线走，遇到障碍就停下
ros2 launch large_models_examples llm_visual_patrol.launch.py

#描述下你看到了什么
ros2 launch large_models_examples vllm_with_camera.launch.py

#跟着前面穿白色衣服的人
ros2 launch large_models_examples vllm_track.launch.py

#去前台看看大门有没有关，然后回来告诉我
ros2 launch large_models_examples vllm_navigation.launch.py map:=map_01

# 路网规划
## 检查和标记路网
cd /home/ubuntu/ros2_ws/src/large_models_examples/large_models_examples/road_network
python3 road_network_builder.py
## 启动路网规划
ros2 launch large_models_examples road_network.launch.py
## 发送路网标记
ros2 topic pub /request_waypoint std_msgs/msg/Int32 "data: 1" --once

## llm 路网规划
# Pi5 和Jetson 用 Vmware 启动RVIZ 
# ros2 run rviz2 rviz2 -d ~/ros2/src/navigation_road_network.rviz
# ros2 run rviz2 rviz2 -d ~/ros2/src/navigation_road_network_pro.rviz
ros2 launch large_models_examples road_network_tool.launch.py
ros2 topic pub --once /vocal_detect/asr_result std_msgs/msg/String '{data: "去公园看看有什么"}'


#### function_calling ###
ros2 launch large_models_examples llm_control_progress.launch.py
ros2 topic pub --once /vocal_detect/asr_result std_msgs/msg/String '{data: "沿着黑线走，遇到障碍物就停下，接着向左转，最后给我描述一下你看到了什么"}'
#################



