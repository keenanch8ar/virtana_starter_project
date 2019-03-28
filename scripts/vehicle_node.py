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
from scipy.interpolate import griddata
from visualization_msgs.msg import Marker
from scipy import interpolate
from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import RegularGridInterpolator



class VehicleBot:

    def __init__(self):

        rospy.init_node('vehicle_node', anonymous=True)
        
        #Load parameters to be used in node
        self.mu = rospy.get_param('/mu')
        self.sigma = rospy.get_param('/sigma')
        self.sigma_filt = rospy.get_param('/sigma_filt')
        self.length = rospy.get_param('/length')
        self.width = rospy.get_param('/width')
        self.resolution =  rospy.get_param('/resolution')
        self.vehicle_length =  rospy.get_param('/v_length')
        self.vehicle_width =  rospy.get_param('/v_width')
        self.x_scale = rospy.get_param('/x_scale')
        self.y_scale = rospy.get_param('/y_scale')

        #Fill array of size h x w with Gaussian Noise given the height and width of the point cloud
        h = self.resolution
        w = self.resolution
        grid_map = (h,w)
        self.gaussian_array = np.random.normal(self.mu, self.sigma, grid_map)

        #Filter the array to smoothen the variation of the noise
        self.gaussian_array = gaussian_filter(self.gaussian_array, self.sigma_filt)
        
        #Create Point Cloud
        self.cloud = self.create_terrain_map()

        #Create Twist variables to store cmd_vel
        self.twist = Twist()

        #Create variables for calculations
        self.vehicle_yaw = 0.0
        self.pose = Pose()

        #Create a lock to prevent race conditions when calculating position
        self.lock = threading.Lock()

        #Timer to update pose every 50ms
        rospy.Timer(rospy.Duration(0.05), self.update_position)

        #Subscriber for to teleop key
        rospy.Subscriber("/turtlebot_teleop/cmd_vel", Twist, self.velocity_cmd_callback)

        #Publisher for PoseStamped() Message
        self.pose_publisher = rospy.Publisher("/move_vehicle/cmd", PoseStamped, queue_size=1)

        #Publisher for a Float32MultiArray() Message
        self.pose_array_publisher = rospy.Publisher("/move_vehicle/cmd_array", Float32MultiArray, queue_size=1)

        #Publisher for PointCloud2 terrain map
        self.point_cloud_publisher = rospy.Publisher("/point_cloud", PointCloud2, latch = True, queue_size=1)

        #Publisher for Vehicle Marker
        self.grid_publisher = rospy.Publisher('/grid_marker', Marker, queue_size=1)

        #Keeps program running until interrupted
        rospy.spin()


    def create_terrain_map(self):

        """
        Creates a list of points from an array of gaussian noise given a specified height and width 
        """

        #Create an list and fill the list with x, y values and z the values will be the filtered gaussian noise
        # TODO perform these calculations without for loops
        cloud = []
        x = np.linspace(0, self.length, self.resolution)
        y = np.linspace(0, self.width, self.resolution)
        for i in range(len(self.gaussian_array)):
            for j in range(len(self.gaussian_array[i])):
                innerlist = []
                innerlist.append(x[i])
                innerlist.append(y[j])
                innerlist.append(self.gaussian_array[i][j])
                cloud.append(innerlist)
        return cloud

    def velocity_cmd_callback(self, data):
        
        """
        Updates the most recent command velocity to a twist variable
        """
        self.lock.acquire()
        try:
            self.twist = data
        finally:
            self.lock.release()


    def update_position(self, event):

        """
        Computes the pose of the vehicle within a given footprint height and width 
        """

        #Create a copy of the most recent stored twist data to perform calculations with
        self.lock.acquire()
        try:
            velocity_data = copy.deepcopy(self.twist)
        finally:
            self.lock.release()

        #time elapsed since last update position call
        if hasattr(event, 'last_real'):
            if event.last_real is None:
                time = rospy.Duration(0.05)
            else:
                time = event.current_real - event.last_real
        
        time = time.to_sec()

        #Calculate angle turned in the given time using omega = theta/time
        #TODO Currently it yaws, then moves forward. Revist calculation to do both simultaneously
        angle = velocity_data.angular.z*time

        #Calculate distance travelled in the given time using linear velocity = arc distance/time
        distance = velocity_data.linear.x*time

        #Calculate yaw of the robot
        self.vehicle_yaw += angle

        #Calculate vehicle x, y, z position coordinates
        # TODO recalculate the position based on traveling in a circular arc.
        self.pose.position.x += (distance)*cos(self.vehicle_yaw)
        self.pose.position.y += (distance)*sin(self.vehicle_yaw)

        #Calculate z position using linear interpolation
        x = np.linspace(0, self.length, self.resolution)
        y = np.linspace(0, self.width, self.resolution)

        f = interpolate.interp2d(x,y,self.gaussian_array,kind='cubic') #x represents column coordinates, y represents row coordinates

        x1 = np.linspace((self.pose.position.x-self.vehicle_length), (self.pose.position.x+self.vehicle_length),5)
        y1 = np.linspace((self.pose.position.y-self.vehicle_width), (self.pose.position.y+self.vehicle_width),5)

        z = f(y1, x1)

        flat_list = []

        for sublist in z:
            for item in sublist:
                flat_list.append(item)
        
        avg = sum(flat_list)/len(flat_list)

        self.pose.position.z = avg

        print(self.pose.position.x, self.pose.position.y, self.pose.position.z)

        #Convert Euler Angles to Quarternion
        q = tf.transformations.quaternion_from_euler(0.0, 0.0, self.vehicle_yaw)

        #Broadcast vehicle frame which is a child of the world frame
        br = tf.TransformBroadcaster()
        br.sendTransform((self.pose.position.x, self.pose.position.y, self.pose.position.z), 
                        q, rospy.Time.now(),"vehicle_frame", "map")

        #Publish all messages
        self.publish_messages(self.pose.position.x, self.pose.position.y, 
                            self.pose.position.z, q)


    def publish_messages(self, x, y, z, q):

        """
        Publishes the pose stamped, multi-array, point-cloud and vehicle footprint vizualization marker. 
        """

        #####################################################################################

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
        point_cloud = pcl2.create_cloud_xyz32(header, self.cloud)

        ##################################################################################

        #Create a marker to vizualize the footprint of the vehicle
        viz_points = Marker()
        viz_points.header.frame_id = "map"
        viz_points.header.stamp = rospy.Time.now()
        viz_points.ns = "marker"
        viz_points.id = 1
        viz_points.action = viz_points.ADD
        viz_points.type = viz_points.CUBE

        viz_points.scale.x = 0.1
        viz_points.scale.y = 0.1
        viz_points.scale.z = 0.001

        viz_points.pose.position.x = x
        viz_points.pose.position.y = y
        viz_points.pose.position.z = z
        viz_points.pose.orientation.x = q[0]
        viz_points.pose.orientation.y = q[1]
        viz_points.pose.orientation.z = q[2]
        viz_points.pose.orientation.w = q[3]

        viz_points.color.a = 1.0
        viz_points.color.r = 1.0
        viz_points.color.g = 0.0
        viz_points.color.b = 0.0

        ##################################################################################

        #Publish pose, vizualization, array information and point cloud
        self.pose_publisher.publish(msg)
        self.pose_array_publisher.publish(array_msg)
        self.point_cloud_publisher.publish(point_cloud)
        self.grid_publisher.publish(viz_points)


if __name__ == '__main__':
    try:
        x = VehicleBot()
    except rospy.ROSInterruptException:
        pass


        