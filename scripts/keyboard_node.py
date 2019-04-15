#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
import sys, select, termios, tty

msg = """
Reading from the keyboard  and Publishing to Twist!
---------------------------
Moving vehicle joint:
    t: up   (+z)
    g: stop (0)
    b: down (-z)

CTRL-C to quit
"""

moveBindings1 = {'t':(0,0,1,1),'g':(0,0,0,0), 'b':(0,0,-1,-1),}
moveBindings2 = {'y':(0,0,0,1),'h':(0,0,0,0), 'n':(0,0,0,-1),}



def getKey():
    tty.setraw(sys.stdin.fileno())
    select.select([sys.stdin], [], [], 0)
    key = sys.stdin.read(1)
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


if __name__ == '__main__':
    settings = termios.tcgetattr(sys.stdin)

    key_pub = rospy.Publisher('/keyboard_cmd_vel', Twist, queue_size = 1)
    rospy.init_node('keyboard_node', anonymous=True)

    x = 0
    y = 0
    z = 0
    th = 0
    speed = 0.3
    turn = 1.0
    try:
        print(msg)
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            key = getKey()
            if key in moveBindings1.keys():
                x = moveBindings1[key][0]
                y = moveBindings1[key][1]
                z = moveBindings1[key][2]
                th = moveBindings1[key][3]
            else:
                x = 0
                y = 0
                z = 0
                th = 0
                if (key == '\x03'):
                    break
            
            twist = Twist()
            twist.linear.x = x*speed
            twist.linear.y = y*speed
            twist.linear.z = z*speed
            twist.angular.x = 0
            twist.angular.y = th*turn
            twist.angular.z = 0
            key_pub.publish(twist)
    
    except Exception as e:
        print(e)
        
    finally:
        twist = Twist()
        twist.linear.x = 0; twist.linear.y = 0; twist.linear.z = 0
        twist.angular.x = 0; twist.angular.y = 0; twist.angular.z = 0
        key_pub.publish(twist)

        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)



