#!/usr/bin/env python
import rospy
import tf
import numpy as np
import ros_numpy
from scipy.ndimage import gaussian_filter
from numpy import sin, cos
from geometry_msgs.msg import Twist, PoseStamped, Pose
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, Header
from sensor_msgs.msg import PointCloud2, JointState
from scipy.interpolate import griddata
import sensor_msgs.point_cloud2 as pcl2
from visualization_msgs.msg import Marker


class VehicleBot:

    def __init__(self):

        rospy.init_node('vehicle_node', anonymous=True)

        #Subscriber for to teleop key
        rospy.Subscriber("/turtle1/cmd_vel", Twist, self.velocity_cmd_callback)

        #Publisher for PoseStamped() Message
        self.pose_publisher = rospy.Publisher("/move_vehicle/cmd", PoseStamped, queue_size=10)

        #Publisher for a Float32MultiArray() Message
        self.pose_array_publisher = rospy.Publisher("/move_vehicle/cmd_array", Float32MultiArray, queue_size=10)

        #Publisher for PointCloud2 terrain map
        self.point_cloud_publisher = rospy.Publisher("/point_cloud", PointCloud2, latch = True, queue_size=10)

        #Publisher for Vehicle Marker
        self.marker_publisher = rospy.Publisher('/vehicle_marker', Marker, queue_size=10 )

        #Publisher for Joints State
        self.joint_publisher = rospy.Publisher('/my_joint_states', JointState, queue_size=1) 

        #Create Vehicle Marker
        self.marker = self.create_vehicle_marker()

        #Create Pose Stamped Message 
        self.pose_stamped = self.create_pose_stamped()

        #Create Float32MultiArray() Message
        self.float_array = self.create_float_array()

        #Create Point Cloud
        self.cloud = self.create_terrain_map()

        #Create Joint State Message
        self.js = JointState()

        #Create Twist and Pose variables to use for calculations
        self.twist = Twist()

        self.pose = Pose()

        self.cloud_msg = PointCloud2()

        #Timer to update pose every 10ms
        rospy.Timer(rospy.Duration(0.01), self.update_position)

        #Timer to publish PoseStamped msg every 10ms
        rospy.Timer(rospy.Duration(0.01), self.publish_pose_msg)

        #Timer to publish Float32MultiArray msg every 10ms
        rospy.Timer(rospy.Duration(0.01), self.publish_array_msg)

        #Timer to publish Terrain Map
        self.point_cloud_timer = rospy.Timer(rospy.Duration(0.5), self.publish_terrain_map)

        #Keeps program runninng until interrupted
        rospy.spin()


    def create_vehicle_marker(self):

        v_height = rospy.get_param('/v_height')
        v_width = rospy.get_param('/v_width')

        viz_points = Marker()
        viz_points.header.frame_id = "/map"
        viz_points.header.stamp = rospy.Time.now()
        viz_points.ns = "vehicle_marker"
        viz_points.id = 0
        viz_points.action = viz_points.ADD
        viz_points.type = viz_points.CUBE

        viz_points.scale.x = v_height
        viz_points.scale.y = v_width
        viz_points.scale.z = 0.0

        viz_points.color.a = 1.0
        viz_points.color.r = 1.0
        viz_points.color.g = 0.0
        viz_points.color.b = 0.0

        viz_points.pose.position.x = 0.0
        viz_points.pose.position.y = 0.0
        viz_points.pose.position.z = 0.0
        viz_points.pose.orientation.x = 0.0
        viz_points.pose.orientation.y = 0.0
        viz_points.pose.orientation.z = 0.0
        viz_points.pose.orientation.w = 1.0

        return viz_points


    def create_pose_stamped(self):

        #Create posestamped message
        msg = PoseStamped()

        #Header details (if im sending a posestmaped message to my command my vehicle, does the header have to be the world frame of reference or to my robot's frame of reference?)
        msg.header.frame_id = "map"
        msg.header.stamp = rospy.Time.now()

        #Pose information
        msg.pose.position.x = 0.0
        msg.pose.position.y = 0.0
        msg.pose.position.z = 0.0

        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = 0.0
        msg.pose.orientation.w = 1.0

        return msg


    def create_float_array(self):
        #Create array message
        array_msg = Float32MultiArray()

        #Array details
        array_msg.layout.dim.append(MultiArrayDimension())
        array_msg.layout.dim[0].label = "vehicle_position"
        array_msg.layout.dim[0].size = 3
        array_msg.layout.dim[0].stride = 3

        return array_msg


    def create_terrain_map(self):

        #Get parameters
        h = rospy.get_param('/gauss_h')
        w = rospy.get_param('/gauss_w')
        mu = rospy.get_param('/mu')
        sigma = rospy.get_param('/sigma')
        sigma_filt = rospy.get_param('/sigma_filt')
        height = rospy.get_param('/height')
        width = rospy.get_param('/width')
        resolution =  rospy.get_param('/resolution')

        #Fill array of size h x w with Gaussian Noise
        grid_map = (h,w)
        gaussian_array = np.random.normal(mu, sigma, grid_map)

        #Filter the array to smoothen the variation of the noise
        gaussian_array = gaussian_filter(gaussian_array, sigma_filt)

        #Create an list and fill the list with x, y values and z the values will be the filtered gaussian noise
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

        resolution =  rospy.get_param('/resolution')
        #time elapsed since last update position call (using event.current_real - event.last_real throws an error on first run as event.last_real does not exist in the first run)
        time = 0.01

        #Calculate angle turned in the given time using omega = theta/time
        angle = self.twist.angular.z*time

        #Calculate distance travelled in the given time using linear velocity = arc distance/time
        distance = self.twist.linear.x*time

        #Calculate yaw, pitch and roll of the robot (pitch and roll currently not calculated)
        self.pose.orientation.x = 0.0
        self.pose.orientation.y = 0.0
        self.pose.orientation.z += angle

        #Calculate vehicle x, y, z position coordinates 
        self.pose.position.x += (distance)*cos(self.pose.orientation.z)
        self.pose.position.y += (distance)*sin(self.pose.orientation.z)

        myarray = np.asarray(self.cloud)

        points = []
        values = []

        for x in range(len(myarray)):
            if (self.pose.position.x - 0.05 - resolution) <= myarray[x,0] <=  (self.pose.position.x + 0.05 + resolution):
                if (self.pose.position.y - 0.05 - resolution) <= myarray[x,1] <= (self.pose.position.y + 0.05 + resolution):
                    innerlist = []
                    innerlist.append(myarray[x,0])
                    innerlist.append(myarray[x,1])
                    values.append(myarray[x,2])
                    points.append(innerlist)

        x1, y1 = np.meshgrid(np.linspace((self.pose.position.x-0.05), (self.pose.position.x+0.05), 50), np.linspace((self.pose.position.y-0.05), (self.pose.position.y+0.05), 50))
                
        grid_z2 = griddata(points, values, (x1, y1), method='linear')

        flat_list = []

        for sublist in grid_z2:
            for item in sublist:
                flat_list.append(item)
                
        avg = sum(flat_list)/len(flat_list)

        self.pose.position.z = avg


    def publish_pose_msg(self, event):
        
        #Header
        self.pose_stamped.header.stamp = rospy.Time.now()

        #Pose information
        self.pose_stamped.pose.position.x = self.pose.position.x
        self.pose_stamped.pose.position.y = self.pose.position.y
        self.pose_stamped.pose.position.z = self.pose.position.z

        #Convert Euler to Quarternion
        q = tf.transformations.quaternion_from_euler(self.pose.orientation.x, self.pose.orientation.y, self.pose.orientation.z, 'sxyz')
        
        self.pose_stamped.pose.orientation.x = q[0]
        self.pose_stamped.pose.orientation.y = q[1]
        self.pose_stamped.pose.orientation.z = q[2]
        self.pose_stamped.pose.orientation.w = q[3]

        #Broadcast vehicle frame which is a child of the world frame
        br = tf.TransformBroadcaster()
        br.sendTransform((self.pose.position.x, self.pose.position.y, self.pose.position.z ), q, rospy.Time.now(),"vehicle_frame", "map")

        self.marker.pose.position.x = self.pose.position.x 
        self.marker.pose.position.y = self.pose.position.y 
        self.marker.pose.position.z = self.pose.position.z 
        self.marker.pose.orientation.x = q[0]
        self.marker.pose.orientation.y = q[1]
        self.marker.pose.orientation.z = q[2]
        self.marker.pose.orientation.w = q[3]

        #Publish marker and pose information
        self.marker_publisher.publish(self.marker)
        self.pose_publisher.publish(self.pose_stamped)


    def publish_array_msg(self, event):

        #Append data
        self.float_array.data.append(self.pose.position.x)
        self.float_array.data.append(self.pose.position.y)
        self.float_array.data.append(self.pose.orientation.z)

        #Publish
        self.pose_array_publisher.publish(self.float_array)


    def publish_terrain_map(self, event):

        #Create a point cloud from the xyz values in the array list
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'map'
        point_cloud = pcl2.create_cloud_xyz32(header, self.cloud)

        #Publish point cloud
        self.point_cloud_publisher.publish(point_cloud)
    

#Needs to subscribe to a node that will provide keyboard input for movement of the node. then will publish the joint states


if __name__ == '__main__':
    try:
        x = VehicleBot()
    except rospy.ROSInterruptException:
        pass