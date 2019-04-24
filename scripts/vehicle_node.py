#!/usr/bin/env python

import rospy
import tf
from math import ceil
import numpy as np
import copy
import threading
import sensor_msgs.point_cloud2 as pcl2
from scipy.ndimage import gaussian_filter
from numpy import sin, cos
from geometry_msgs.msg import Twist, PoseStamped, Pose, Point
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, Header
from sensor_msgs.msg import PointCloud2, JointState
from visualization_msgs.msg import Marker
from scipy.interpolate import RectBivariateSpline

class VehicleBot(object):

    """
    Creates a simulated bulldozer vehicle that is capable of traversing over rugged terrain. 
    The terrain parameters can be adjusted in the parameters.yaml file.
    """

    def __init__(self):

        rospy.init_node('vehicle_node', anonymous=True)
        
        #Load parameters to be used in node. Description of each in parameters yaml file.
        self.__dict__.update(rospy.get_param('/map_params'))

        #Create Gaussian Array of normally distributed noise
        self.gaussian_array = self.create_gaussian_array()
        
        #Create Twist variables to store cmd_vel
        self.twist = Twist()
        self.joint = JointState()

        #Create variables for calculations
        self.vehicle_yaw = 0.0
        self.joint1_pitch = 0.0
        self.joint2_pitch = 0.0
        self.pose = Pose()
        self.joint1_pose = Pose()
        self.joint2_pose = Pose()

        #Create a lock to prevent race conditions when calculating position
        self.lock = threading.Lock()

        #Create a lock to prevent race conditions when calculating position
        self.joint_lock = threading.Lock()

        #Timer to update pose every 50ms
        rospy.Timer(rospy.Duration(0.05), self.update_position)

        #Subscriber for teleop key
        rospy.Subscriber("/turtlebot_teleop/cmd_vel", Twist, self.velocity_cmd_callback)

        #Subscriber for joint teleop key
        rospy.Subscriber("/keyboard/joints", JointState, self.joint_cmd_callback)

        #Publisher for PoseStamped() Message
        self.pose_publisher = rospy.Publisher("/move_vehicle/cmd", PoseStamped, queue_size=1)

        #Publisher for Joint1 PoseStamped() Message
        self.joint1_pose_publisher = rospy.Publisher("/move_joint1/cmd", PoseStamped, queue_size=1)

        #Publisher for Joint2 PoseStamped() Message
        self.joint2_pose_publisher = rospy.Publisher("/move_joint2/cmd", PoseStamped, queue_size=1)
        
        #Publisher for a Float32MultiArray() Message
        self.pose_array_publisher = rospy.Publisher("/move_vehicle/cmd_array", Float32MultiArray, queue_size=1)

        #Latched Publisher for PointCloud2 terrain map
        self.point_cloud_publisher = rospy.Publisher("/point_cloud", PointCloud2, latch = True, queue_size=1)

        #Publisher for Vehicle Marker
        self.grid_publisher = rospy.Publisher('/grid_marker', Marker, queue_size=1)

        #Keeps program running until interrupted
        rospy.spin()

    def create_gaussian_array(self):

        """
        Creates an array filled with gaussian noise using the parameters loaded
        from the parameter server.
        """

        #Fill array of size l x w with Gaussian Noise.
        l = int(ceil(self.length/self.resolution))
        w = int(ceil(self.width/self.resolution))
        grid_map = (l,w)
        gaussian_array = np.random.normal(self.mu, self.sigma, grid_map)

        #Filter the array to smoothen the variation of the noise
        gaussian_array = gaussian_filter(gaussian_array, self.sigma_filter)

        return gaussian_array

    def velocity_cmd_callback(self, data):
        
        """
        Updates the most recent command velocity to a twist variable
        """
        with self.lock:
            self.twist = data

    def joint_cmd_callback(self, data):

        with self.joint_lock:
            self.joint = data

    def update_position(self, event):

        """
        Computes the pose of the vehicle in the terrian map. 
        """

        #Create a copy of the most recent stored twist data to perform calculations
        with self.lock:
            velocity_data = copy.deepcopy(self.twist)


        #time elapsed since last update position call
        if hasattr(event, 'last_real'):
            if event.last_real is None:
                time = rospy.Duration(0.05)
            else:
                time = event.current_real - event.last_real
        
        time = time.to_sec()

        #Calculate angle turned in the given time using omega = theta/time
        angle = velocity_data.angular.z*time

        #Calculate distance travelled in the given time using linear velocity = arc distance/time
        distance = velocity_data.linear.x*time

        #Calculate yaw of the robot
        self.vehicle_yaw += angle

        #Calculate vehicle x, y, z position coordinates
        # TODO recalculate the position based on traveling in a circular arc.
        self.pose.position.x += (distance)*cos(self.vehicle_yaw)
        self.pose.position.y += (distance)*sin(self.vehicle_yaw)

        #Calculate z position using linear interpolation and create cloud array
        
        #1. Create range to be used in interpolation function
        x = np.arange(0, self.gaussian_array.shape[0]*self.resolution, self.resolution)
        y = np.arange(0, self.gaussian_array.shape[1]*self.resolution, self.resolution)

        #2. Create cloud array to be converted to point cloud for vizualization
        #TODO do this without for loops
        cloud = []
        for i in range(self.gaussian_array.shape[0]):
            for j in range(self.gaussian_array.shape[1]):
                innerlist = []
                innerlist.append(x[i])
                innerlist.append(y[j])
                innerlist.append(self.gaussian_array[i][j])
                cloud.append(innerlist)

        #3. Create interpolation function based on the ranges and gaussian data
        f = RectBivariateSpline(x,y, self.gaussian_array)

        #4. Find z value for x and y coordinate of vehicle using interpolation function
        # TODO compute z height based on footprint
        self.pose.position.z = f(self.pose.position.x, self.pose.position.y)

        #Convert Euler Angles to Quarternion
        q = tf.transformations.quaternion_from_euler(0.0, 0.0, self.vehicle_yaw)

        #Broadcast vehicle frame which is a child of the world frame
        br = tf.TransformBroadcaster()
        br.sendTransform((self.pose.position.x, self.pose.position.y, self.pose.position.z), 
                        q, rospy.Time.now(),"vehicle_frame", "map")

        #Construct the homogenous transformation matrix for map to vehicle frame
        translation = [self.pose.position.x, self.pose.position.y, self.pose.position.z]
        map_T_vehicle = tf.transformations.quaternion_matrix(q) 
        map_T_vehicle[:3,3] = np.array(translation)

        #Create footprint of vehicle
        x1 = np.linspace((-self.vehicle_length/2), (self.vehicle_length/2),30)
        y1 = np.linspace((-self.vehicle_width/2), (self.vehicle_width/2),15)
        x1, y1 = np.meshgrid(x1, y1)
        x1 = x1.ravel()
        y1 = y1.ravel()

        #For every point in the vehicle footprint, calculate the position wrt to the vehicle's frame
        # and its interpolated z value. Add this point to a list of points for visualization.
        # TODO Flatten into a single matrix multiply to remove for loop
        points = []
        for i in range(x1.shape[0]):
            p = Point()
            footprint = np.array([[x1[i]],[y1[i]],[0.0],[1.0]])
            footprint = np.matmul(map_T_vehicle, footprint)
            footprint[2,0] =  f(footprint[0,0], footprint[1,0])
            p.x = footprint[0,0]
            p.y = footprint[1,0]
            p.z = footprint[2,0]
            points.append(p)

#################################################################################################
        #Create a copy of the most recent stored twist data to perform calculations
        with self.joint_lock:
            joint_data = copy.deepcopy(self.joint)

        if not joint_data.velocity:
            joint_data.velocity = [0.0,0.0]
        
        angle = joint_data.velocity[0]*time
        angle2 = joint_data.velocity[1]*time

        self.joint1_pitch += angle
        self.joint2_pitch += angle2

        static_rot = tf.transformations.quaternion_from_euler(0.0, 0.0, 3.14159)
        translation =  [0.0, 0.0, 0.0]
        V_T_SRz = tf.transformations.quaternion_matrix(static_rot)
        V_T_SRz[:3,3] = np.array(translation)

        dynamic_rot = tf.transformations.quaternion_from_euler(0.0, self.joint1_pitch, 0.0)
        translation =  [0.0, 0.0, 0.0]
        SRz_T_J1 = tf.transformations.quaternion_matrix(dynamic_rot)
        SRz_T_J1[:3,3] = np.array(translation)

        no_rot = tf.transformations.quaternion_from_euler(0.0, 0.0, 0.0)
        translation =  [0.1, 0.0, 0.0]
        J1_T_STx = tf.transformations.quaternion_matrix(no_rot)
        J1_T_STx[:3,3] = np.array(translation)

        dynamic_rot2 = tf.transformations.quaternion_from_euler(0.0, self.joint2_pitch, 0.0)
        translation =  [0.0, 0.0, 0.0]
        STx_T_J2 = tf.transformations.quaternion_matrix(dynamic_rot2)
        STx_T_J2[:3,3] = np.array(translation)

        V_T_J1 = np.matmul(V_T_SRz, SRz_T_J1)
        V_T_STx = np.matmul(V_T_J1, J1_T_STx)
        V_T_J2 = np.matmul(V_T_STx, STx_T_J2)
        rot_J1 = tf.transformations.quaternion_from_matrix(V_T_J1)
        rot_J2 = tf.transformations.quaternion_from_matrix(V_T_J2)

        self.joint1_pose.position.x = V_T_J1[0][3]
        self.joint1_pose.position.y = V_T_J1[1][3]
        self.joint1_pose.position.z = V_T_J1[2][3]
        self.joint1_pose.orientation.x = rot_J1[0]
        self.joint1_pose.orientation.y = rot_J1[1]
        self.joint1_pose.orientation.z = rot_J1[2]
        self.joint1_pose.orientation.w = rot_J1[3]

        self.joint2_pose.position.x = V_T_J2[0][3]
        self.joint2_pose.position.y = V_T_J2[1][3]
        self.joint2_pose.position.z = V_T_J2[2][3]
        self.joint2_pose.orientation.x = rot_J2[0]
        self.joint2_pose.orientation.y = rot_J2[1]
        self.joint2_pose.orientation.z = rot_J2[2]
        self.joint2_pose.orientation.w = rot_J2[3]

        ripper_point =  [0.05, 0.0, 0.0, 1.0]
        map_T_J2 =  np.matmul(map_T_vehicle, V_T_J2)
        tip =  np.matmul(map_T_J2, ripper_point)
        ripper_tip = Point()
        ripper_tip.x = tip[0]
        ripper_tip.y = tip[1]
        ripper_tip.z = tip[2]
        points.append(ripper_tip)

        index_x = int(tip[0]/self.resolution)
        index_y = int(tip[1]/self.resolution)

        print_x = np.arange((index_x-1),(index_x+2), 1)
        print_y = np.arange((index_y-1),(index_y+2), 1)
        print_x1, print_y1 = np.meshgrid(print_x, print_y)
        print_x2 = print_x1.ravel()
        print_y2 = print_y1.ravel()

        if (0 <= index_x <= self.gaussian_array.shape[0]) and (0 <= index_y <= self.gaussian_array.shape[1]):
            if (self.gaussian_array[index_x][index_y] >= tip[2]):
                diff = self.gaussian_array[index_x][index_y] - tip[2]
                for i in range(print_x2.shape[0]):
                    if (0 <= print_x2[i] <= self.gaussian_array.shape[0]) and (0 <= print_y2[i] <= self.gaussian_array.shape[1]):
                            self.gaussian_array[print_x2[i]][print_y2[i]] += diff/9
                self.gaussian_array[index_x][index_y] = tip[2]


        #Publish all messages
        self.publish_messages(self.pose.position.x, self.pose.position.y, 
                             self.pose.position.z, q, cloud, points, self.joint1_pose, 
                             self.joint2_pose)


    def publish_messages(self, x, y, z, q, cloud, points, joint1, joint2):

        """
        Publishes the pose stamped, multi-array, point-cloud and vehicle footprint vizualization
        marker. 
        """

        ##################################################################################

        #Create a posestamped message containing position information

        #Create pose message
        msg = PoseStamped()

        #Header details for pose message
        msg.header.frame_id = "map"
        msg.header.stamp = rospy.Time.now()

        #Pose information
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = z
        msg.pose.orientation.x = q[0]
        msg.pose.orientation.y = q[1]
        msg.pose.orientation.z = q[2]
        msg.pose.orientation.w = q[3]


        ##################################################################################

        #Create an multi array message containing pose information

        #Create array message
        array_msg = Float32MultiArray()
        array_msg.layout.dim.append(MultiArrayDimension())
        array_msg.layout.dim[0].label = "vehicle_position"
        array_msg.layout.dim[0].size = 3
        array_msg.layout.dim[0].stride = 3

        #Append data
        array_msg.data.append(x)
        array_msg.data.append(y)
        array_msg.data.append(z)

        ##################################################################################

        #Create point cloud and publish to rviz

        #Create a point cloud from the xyz values in the array list
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'map'
        point_cloud = pcl2.create_cloud_xyz32(header, cloud)

        ##################################################################################

        #Create a marker to vizualize the footprint of the vehicle
        viz_points = Marker()
        viz_points.header.frame_id = "map"
        viz_points.header.stamp = rospy.Time.now()
        viz_points.ns = "grid_marker"
        viz_points.id = 1
        viz_points.action = viz_points.ADD
        viz_points.type = viz_points.CUBE_LIST

        viz_points.scale.x = 0.01
        viz_points.scale.y = 0.01
        viz_points.scale.z = 0.01

        viz_points.color.a = 1.0
        viz_points.color.r = 1.0
        viz_points.color.g = 0.0
        viz_points.color.b = 0.0
        viz_points.points = points


        ################################################################

        #Create pose message
        msg1 = PoseStamped()
        msg2 = PoseStamped()
        #Header details for pose message
        msg1.header.frame_id = "vehicle_frame"
        msg1.header.stamp = rospy.Time.now()

        msg2.header.frame_id = "vehicle_frame"
        msg2.header.stamp = rospy.Time.now()

        #Pose information
        msg1.pose = joint1
        msg2.pose = joint2

        #Publish pose, vizualization, array information and point cloud
        self.pose_publisher.publish(msg)
        self.joint1_pose_publisher.publish(msg1)
        self.joint2_pose_publisher.publish(msg2)
        self.pose_array_publisher.publish(array_msg)
        self.point_cloud_publisher.publish(point_cloud)
        self.grid_publisher.publish(viz_points)


if __name__ == '__main__':
    try:
        x = VehicleBot()
    except rospy.ROSInterruptException:
        pass







