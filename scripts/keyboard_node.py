#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Twist
import sys, select, termios, tty

def keyboard_node():

    rospy.init_node('keyboard_node', anonymous=True)

    key_pub = rospy.Publisher('/keyboard_cmd_vel', Twist, queue_size = 1)

    moveBindings1 = {'t':(0,0,0,1), 'b':(0,0,0,-1),}
    moveBindings2 = {'y':(0,0,0,1), 'n':(0,0,0,-1),}
    x = 0
    y = 0
    z = 0
    th = 0
    speed = 1.0
    turn = 1.0
    while(1):
        key = getKey()
        if key in moveBindings1.keys():
            x = moveBindings1[key][0]
            y = moveBindings1[key][1]
            z = moveBindings1[key][2]
            th = moveBindings1[key][3]
            twist = Twist()
            twist.linear.x = x*speed
            twist.linear.y = y*speed
            twist.linear.z = z*speed
            twist.angular.x = 0
            twist.angular.y = th*turn
            twist.angular.z = 0
            key_pub.publish(twist)
        elif key in moveBindings2.keys():
            x = moveBindings2[key][0]
            y = moveBindings2[key][1]
            z = moveBindings2[key][2]
            th = moveBindings2[key][3]
            twist = Twist()
            twist.linear.x = x*speed
            twist.linear.y = y*speed
            twist.linear.z = z*speed
            twist.angular.x = 0
            twist.angular.y = th*turn
            twist.angular.z = 0
            key_pub.publish(twist)

    rospy.spin()


def getKey():
    settings = termios.tcgetattr(sys.stdin)
    tty.setraw(sys.stdin.fileno())
    select.select([sys.stdin], [], [], 0)
    key = sys.stdin.read(1)
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


if __name__ == '__main__':
    try:
        keyboard_node()
    except rospy.ROSInterruptException:
        pass