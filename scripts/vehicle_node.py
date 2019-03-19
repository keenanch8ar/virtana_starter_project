#!/usr/bin/env python
import rospy
import tf
import numpy as np
import ros_numpy
from scipy.ndimage import gaussian_filter
from numpy import sin, cos
from geometry_msgs.msg import Twist, PoseStamped, Pose, Point
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, Header
from sensor_msgs.msg import PointCloud2, JointState
from scipy.interpolate import griddata
import sensor_msgs.point_cloud2 as pcl2
from visualization_msgs.msg import Marker
from math import pi
import Queue


class VehicleBot:

    def __init__(self):

        rospy.init_node('vehicle_node', anonymous=True)

        #Subscriber for to teleop key
        rospy.Subscriber("/turtlebot_teleop/cmd_vel", Twist, self.velocity_cmd_callback)

        #Subscriber for to keyboard commands for joints
        rospy.Subscriber("/keyboard_cmd_vel", Twist, self.joints_listener)

        #Publisher for PoseStamped() Message
        self.pose_publisher = rospy.Publisher("/move_vehicle/cmd", PoseStamped, queue_size=10)

        #Publisher for Joint PoseStamped() Message
        self.joint_pose_publisher = rospy.Publisher("/move_joint/cmd", PoseStamped, queue_size=10)

        #Publisher for a Float32MultiArray() Message
        self.pose_array_publisher = rospy.Publisher("/move_vehicle/cmd_array", Float32MultiArray, queue_size=10)

        #Publisher for PointCloud2 terrain map
        self.point_cloud_publisher = rospy.Publisher("/point_cloud", PointCloud2, latch = True, queue_size=10)

        #Publisher for Vehicle Marker
        self.grid_publisher = rospy.Publisher('/grid_marker', Marker, queue_size=10 )

        #Publisher for Joints State
        self.joint_publisher = rospy.Publisher('/my_joint_states', JointState, queue_size=1) 

        #Create Point Cloud
        self.cloud = self.create_terrain_map()

        #Create Joint State Message
        self.js = JointState()

        #Create Twist variables to store cmd_vel
        self.twist = Twist()
        self.joint_twist = Twist()

        #Create variables for calculations
        self.vehicle_yaw = 0.0
        self.vehicle_pitch = 0.0
        self.vehicle_roll = 0.0
        self.vehicle_x = 0.0
        self.vehicle_y = 0.0
        self.vehicle_z = 0.0

        self.joint_yaw = -1.0
        self.joint_pitch = 0.0
        self.joint_roll = 0.0
        self.joint_x = 0.0
        self.joint_y = 0.0
        self.joint_z = 0.0

        self.cloud_msg = PointCloud2()

        #Timer to update pose every 50ms
        rospy.Timer(rospy.Duration(0.05), self.update_position)

        #Timer to publish Joint msg and pose for joint every 10ms (doesn't work yet)
        #rospy.Timer(rospy.Duration(0.05), self.update_joint_positions)

        #Keeps program running until interrupted
        rospy.spin()

    def create_terrain_map(self):

        #Get parameters
        mu = rospy.get_param('/mu')
        sigma = rospy.get_param('/sigma')
        sigma_filt = rospy.get_param('/sigma_filt')
        height = rospy.get_param('/height')
        width = rospy.get_param('/width')
        resolution =  rospy.get_param('/resolution')

        #Fill array of size h x w with Gaussian Noise given the height and width of the point cloud
        h = int(((height*2) + 1)*(resolution*1000))
        w = int(((width*2) + 1)*(resolution*1000))
        grid_map = (h,w)
        gaussian_array = np.random.normal(mu, sigma, grid_map)

        #Filter the array to smoothen the variation of the noise
        gaussian_array = gaussian_filter(gaussian_array, sigma_filt)

        #Create an list and fill the list with x, y values and z the values will be the filtered gaussian noise
        # TODO perform these calculations without for loops
        incr_x = 0
        incr_y = 0
        cloud = []
        for x in np.arange(-height, (height+1), resolution):
            for y in np.arange(-width, (width+1), resolution):
                innerlist = []
                innerlist.append(x)
                innerlist.append(y)
                innerlist.append(gaussian_array[incr_x][incr_y])
                cloud.append(innerlist)
                incr_y+=1
            incr_x+= 1
            incr_y = 0

        return cloud


    def velocity_cmd_callback(self, data):

        #assign updated command velocities to twist variable

        self.twist = data


    def update_position(self, event):

        #Load parameters
        resolution =  rospy.get_param('/resolution')
        v_height =  rospy.get_param('/v_height')
        v_width =  rospy.get_param('/v_width')

        #Local variable to calculate position based on most recent command velocity data
        position_data = self.twist

        #time elapsed since last update position call
        if hasattr(event, 'last_real'):
            if event.last_real is None:
                event.last_real = rospy.Time.now()
                time = event.current_real - event.last_real
            else:
                time = event.current_real - event.last_real
        
        time = time.to_sec()

        #Calculate angle turned in the given time using omega = theta/time
        angle = position_data.angular.z*time

        #Calculate distance travelled in the given time using linear velocity = arc distance/time
        distance = position_data.linear.x*time

        #Calculate yaw, pitch and roll of the robot (pitch and roll currently not calculated)
        self.vehicle_roll = 0.0
        self.vehicle_pitch = 0.0
        self.vehicle_yaw += angle

        #Calculate vehicle x, y, z position coordinates 
        self.vehicle_x += (distance)*cos(self.vehicle_yaw)
        self.vehicle_y += (distance)*sin(self.vehicle_yaw)

        #Convert the point cloud to an array for faster access
        myarray = np.asarray(self.cloud)

        points = []
        values = []

        #Find all the points within the point cloud array that lie within the footprint of the vehicle
        for x in range(len(myarray)):
            if (self.vehicle_x - v_height - resolution) <= myarray[x,0] <=  (self.vehicle_x + v_height + resolution):
                if (self.vehicle_y - v_width - resolution) <= myarray[x,1] <= (self.vehicle_y + v_width + resolution):
                    innerlist = []
                    innerlist.append(myarray[x,0])
                    innerlist.append(myarray[x,1])
                    values.append(myarray[x,2])
                    points.append(innerlist)

        #Create a grid mesh of the size given by the footprint of the vehicle
        x1,y1 = np.meshgrid(np.linspace((self.vehicle_x-v_height), (self.vehicle_x+v_height), 50), np.linspace((self.vehicle_y-v_width), (self.vehicle_y+v_width), 50))
                
        #Linear Interpolation of the grid mesh onto the points in the point cloud that lie within the grid mesh
        grid_z2 = griddata(points, values, (x1, y1), method='cubic')

        flat_list = []

        #Flatten the list of interpolated points
        for sublist in grid_z2:
            for item in sublist:
                flat_list.append(item)
        
        #Find the average and assign the calculated z-height to the vehicle
        avg = sum(flat_list)/len(flat_list)
        self.vehicle_z = avg

        #Convert Euler Angles to Quarternion
        q = tf.transformations.quaternion_from_euler(self.vehicle_roll, self.vehicle_pitch, self.vehicle_yaw)
        
        #Publish all messages
        self.publish_messages(self.vehicle_x, self.vehicle_y, self.vehicle_z, x1, y1, grid_z2, q)


    def publish_messages(self, x, y, z, x1, y1, grid_z2, q):

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

        #Broadcast vehicle frame which is a child of the world frame
        br = tf.TransformBroadcaster()
        br.sendTransform((x, y, z), q, rospy.Time.now(),"vehicle_frame", "map")

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
        viz_points.ns = "grid_marker"
        viz_points.id = 1
        viz_points.action = viz_points.ADD
        viz_points.type = viz_points.CUBE_LIST

        viz_points.scale.x = 0.02
        viz_points.scale.y = 0.02
        viz_points.scale.z = 0.02

        viz_points.color.a = 1.0
        viz_points.color.r = 1.0
        viz_points.color.g = 0.0
        viz_points.color.b = 0.0
        
        for i in range(len(x1)):
            for j in range(len(x1[i])):
                p = Point()
                p.x = x1[i][j]
                p.y = y1[i][j]
                p.z = grid_z2[i][j]
                viz_points.points.append(p)


        ##################################################################################

        #Publish pose, vizualization, array information and point cloud
        self.pose_publisher.publish(msg)
        self.pose_array_publisher.publish(array_msg)
        self.point_cloud_publisher.publish(point_cloud)
        self.grid_publisher.publish(viz_points)

        
    #Doesn't work yet, ignore
    def joints_listener(self, data):

        self.joint_twist = data

    #Doesn't work yet, ignore
    def update_joint_positions(self, event):
        
        #Local variable to calculate position based on most recent command velocity data
        position_data = self.twist

        #time elapsed since last update position call
        if hasattr(event, 'last_real'):
            if event.last_real is None:
                event.last_real = rospy.Time.now()
                time = event.current_real - event.last_real
            else:
                time = event.current_real - event.last_real
        
        time = time.to_sec()

        #Calculate angle turned in the given time using omega = theta/time
        angle = position_data.angular.z*time

        #Calculate distance travelled in the given time using linear velocity = arc distance/time
        distance = position_data.linear.x*time

        #Calculate yaw, pitch and roll of the robot (pitch and roll currently not calculated)
        self.joint_roll = 0.0
        self.joint_pitch = 0.0
        self.joint_yaw += angle

        #Calculate vehicle x, y, z position coordinates 
        self.joint_x += (distance)*cos(self.joint_yaw)
        self.joint_y += (distance)*sin(self.joint_yaw)
        self.joint_z = 0.0

        self.js.header = Header()
        self.js.header.stamp = rospy.Time.now()
        self.js.position = [self.joint_x, self.joint_y, self.joint_z]
        self.js.velocity = [position_data.angular.y]

        self.joint_publisher.publish(self.js)

        ###############################################################

        #Header
        msg = PoseStamped()

        #Header
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "map"

        msg.pose.position.x =  self.joint_x
        msg.pose.position.y =  self.joint_y
        msg.pose.position.z =  self.joint_z

         #Convert Euler to Quarternion
        q1 = tf.transformations.quaternion_from_euler(self.joint_roll, self.joint_pitch, self.joint_yaw, 'sxyz')

        msg.pose.orientation.x = q1[0]
        msg.pose.orientation.y = q1[1]
        msg.pose.orientation.z = q1[2]
        msg.pose.orientation.w = q1[3]

        #Broadcast vehicle frame which is a child of the world frame
        br2 = tf.TransformBroadcaster()
        br2.sendTransform((-1.0, 0.0, 0.0), (0.0,0.0,0.0,1), rospy.Time.now(),"joint_frame", "vehicle_frame")

        self.joint_pose_publisher.publish(msg)
        

if __name__ == '__main__':
    try:
        x = VehicleBot()
    except rospy.ROSInterruptException:
        pass