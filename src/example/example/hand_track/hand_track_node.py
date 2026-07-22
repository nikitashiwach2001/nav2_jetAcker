#!/usr/bin/env python3
# encoding: utf-8
# @data:2022/11/07
# @author:aiden
# 手跟随(hand following)
import time
import rclpy
import signal
import threading
import sdk.pid as pid
from rclpy.node import Node
from std_srvs.srv import Trigger
from interfaces.msg import Point2D
from geometry_msgs.msg import Twist
from rclpy.executors import MultiThreadedExecutor
from servo_controller_msgs.msg import ServosPosition
from rclpy.callback_groups import ReentrantCallbackGroup
from servo_controller.bus_servo_control import set_servo_position

class HandTrackNode(Node):
    def __init__(self, name):
        rclpy.init()
        super().__init__(name)
        self.name = name
        self.image = None
        self.center = None
        self.running = True
        self.y_dis = 500

        self.pid_y = pid.PID(0.03, 0.0, 0.0)
        signal.signal(signal.SIGINT, self.shutdown)

        self.mecanum_pub = self.create_publisher(Twist, '/controller/cmd_vel', 1)
        self.joints_pub = self.create_publisher(ServosPosition, '/servo_controller', 1) # 舵机控制(servo control)

        self.image_sub = self.create_subscription(Point2D, '/hand_detect/center', self.get_hand_callback, 1)
        timer_cb_group = ReentrantCallbackGroup()

        self.timer = self.create_timer(0.0, self.init_process, callback_group=timer_cb_group)

    def init_process(self):
        self.timer.cancel()

        self.init_action()

        threading.Thread(target=self.main, daemon=True).start()
        self.create_service(Trigger, '~/init_finish', self.get_node_state)
        self.get_logger().info('\033[1;32m%s\033[0m' % 'start')

    def get_node_state(self, request, response):
        response.success = True
        return response

    def shutdown(self, signum, frame):
        self.running = False

    def init_action(self):
        set_servo_position(self.joints_pub, 0.5, ((1, 500), ))
        time.sleep(0.5)
        self.mecanum_pub.publish(Twist())

    def get_hand_callback(self, msg):
        if msg.width != 0:
            self.center = msg
        else:
            self.center = None

    def main(self):
        while self.running:
            if self.center is not None:
                t1 = time.time()
                self.pid_y.SetPoint = self.center.width / 2
                self.pid_y.update(self.center.width - self.center.x)
                self.y_dis += self.pid_y.output
                if self.y_dis < 200:
                    self.y_dis = 200
                if self.y_dis > 800:
                    self.y_dis = 800

                t2 = time.time()
                t = t2 - t1
                if t < 0.02:
                    time.sleep(0.02 - t)
                set_servo_position(self.joints_pub, 0.02, ((1, int(self.y_dis)), ))

            else:
                time.sleep(0.01)

        self.init_action()
        rclpy.shutdown()

def main():
    node = HandTrackNode('hand_track')
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    node.destroy_node()

if __name__ == "__main__":
    main()

