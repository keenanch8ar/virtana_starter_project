#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
import sys, select, termios, tty

msg = """
Reading from the keyboard  and Publishing to JointState!
---------------------------
Moving vehicle joint 1 & 2:
Joint 1:                Joint 2:
    t: up   (+z)            y: rotate up (+z)
    g: stop (0)             h: stop (0)
    b: down (-z)            n: rotate down (-z)

CTRL-C to quit
"""

moveBindings1 = {'t':-1.0,'g':0.0, 'b':1.0}
moveBindings2 = {'y':-1.0,'h':0.0, 'n':1.0}



def getKey():
    tty.setraw(sys.stdin.fileno())
    select.select([sys.stdin], [], [], 0)
    key = sys.stdin.read(1)
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


if __name__ == '__main__':
    settings = termios.tcgetattr(sys.stdin)

    key_pub = rospy.Publisher('/keyboard/joints', JointState, queue_size = 1)
    rospy.init_node('keyboard_node', anonymous=True)

    vel1 = 0.0
    vel2 = 0.0
    speed_scaling = 1.0

    try:
        print(msg)
        joint = JointState()
        joint.header = Header()
        joint.name = ['joint1','joint2']
        joint.position = []
        joint.effort = []
        while not rospy.is_shutdown():

            key = getKey()
            if key in moveBindings1.keys():
                vel1 = moveBindings1[key]
            elif key in moveBindings2.keys():
                vel2 = moveBindings2[key]
            else:
                vel1 = 0.0
                vel2 = 0.0
                if (key == '\x03'):
                    break

            joint.header.stamp = rospy.Time.now()
            joint.velocity = [vel1*speed_scaling,vel2*speed_scaling]
            key_pub.publish(joint)

    except Exception as e:
        print(e)
        
    finally:

        joint = JointState()
        joint.name = ['joint1', 'joint2']
        joint.position = []
        joint.effort = []
        joint.header.stamp = rospy.Time.now()
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)



