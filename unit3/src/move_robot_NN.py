#! /usr/bin/env python

import rospy
from geometry_msgs.msg import Twist



class MoveRobot():
    def __init__(self):
        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.move_msg = Twist()

    def send_cmd(self, linear=0, angular=0):
        self.move_msg.linear.x = linear
        self.move_msg.angular.z = angular
        self.publish_once_in_topic()

    def publish_once_in_topic(self):
        while not rospy.is_shutdown():
            connections = self.pub.get_num_connections()
            if connections > 0:
                self.pub.publish(self.move_msg)
            
                break
            else:
                continue

if __name__ == '__main__':
    rospy.init_node('node')
    moverobot_object = MoveRobot()
    moverobot_object.send_cmd(0,0)
        
