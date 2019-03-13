#!/usr/bin/env python 
import rospy
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker


def vizualize_vehicle():

    rospy.init_node('vehicle_rviz', anonymous=True)

    rospy.Subscriber("move_vehicle/cmd", PoseStamped, callback)

    rospy.spin()

def callback(data):

    viz_points = Marker()
    viz_points.header.frame_id = "/my_frame"
    viz_points.header.stamp = rospy.Time.now()
    viz_points.ns = "points_and_lines"
    viz_points.id = 0
    viz_points.action = Marker.ADD
    viz_points.type = 1

    viz_points.pose = data

if __name__ == '__main__':
    try:
        vizualize_vehicle()
    except rospy.ROSInterruptException:
        pass