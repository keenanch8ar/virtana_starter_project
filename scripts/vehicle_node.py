#!/usr/bin/env python

import rospy
import tf
from math import ceil
import numpy as np
import copy
import threading
import sensor_msgs.point_cloud2 as pcl2
import tf_conversions
from scipy.ndimage import gaussian_filter
from numpy import sin, cos
from geometry_msgs.msg import Twist, PoseStamped, Pose, Point
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, Header
from sensor_msgs.msg import PointCloud2, JointState
from visualization_msgs.msg import Marker
from scipy.interpolate import RectBivariateSpline

class VehicleBot(object):

    """
    Creates a simulated bulldozer vehicle that is capable of traversing 
    over rugged terrain. The terrain parameters can be adjusted in the 
    parameters.yaml file.
    """

    def __init__(self):

        rospy.init_node('vehicle_node', anonymous=True)
        
        # Load parameters to be used in node. Description of each in parameters yaml file.
        self.__dict__.update(rospy.get_param('/map_params'))
        self.__dict__.update(rospy.get_param('/vehicle_params'))

        # Create Gaussian Array of normally distributed noise
        self.gaussian_array = self.create_gaussian_array()
        
        # Create Twist and Joint State variables to store cmd_vel
        self.twist = Twist()
        self.joint = JointState()

        # Create variables for calculations
        self.vehicle_yaw = 0.0
        self.joint1_pitch = 0.0
        self.joint2_pitch = 0.0
        self.pose = Pose()

        # Create locks to prevent race conditions when calculating position
        self.lock = threading.Lock()
        self.joint_lock = threading.Lock()

        # Subscriber for teleop key
        rospy.Subscriber("/turtlebot_teleop/cmd_vel", Twist, self.velocity_cmd_callback)

        # Subscriber for joint teleop key
        rospy.Subscriber("/keyboard/joints", JointState, self.joint_cmd_callback)

        # Publisher for PoseStamped() Message
        self.pose_publisher = rospy.Publisher("/move_vehicle/cmd", PoseStamped, queue_size=1)

        # Publisher for Joint1 PoseStamped() Message
        self.joint1_pose_publisher = rospy.Publisher("/move_joint1/cmd", PoseStamped, queue_size=1)

        # Publisher for Joint2 PoseStamped() Message
        self.joint2_pose_publisher = rospy.Publisher("/move_joint2/cmd", PoseStamped, queue_size=1)
        
        # Publisher for a Float32MultiArray() Message
        self.pose_array_publisher = rospy.Publisher("/move_vehicle/cmd_array", Float32MultiArray, queue_size=1)

        # Latched Publisher for PointCloud2 terrain map
        self.point_cloud_publisher = rospy.Publisher("/point_cloud", PointCloud2, latch = True, queue_size=1)

        # Publisher for Vehicle Marker
        self.grid_publisher = rospy.Publisher('/grid_marker', Marker, queue_size=1)

        # Timer to update pose every 50ms
        rospy.Timer(rospy.Duration(0.05), self.update_position)

        # Keeps program running until interrupted
        rospy.spin()

    def create_gaussian_array(self):

        """
        Creates an array filled with gaussian noise using the parameters loaded
        from the parameter server.
        """

        # Fill array of size l x w with Gaussian Noise.
        terrain_length = int(ceil(self.length/self.resolution))
        terrain_width = int(ceil(self.width/self.resolution))
        gaussian_array = np.random.normal(self.mu, self.sigma, (terrain_length,terrain_width))

        # Filter the array to smoothen the variation of the noise
        gaussian_array = gaussian_filter(gaussian_array, self.sigma_filter)

        return gaussian_array

    def velocity_cmd_callback(self, data):
        
        """
        Updates the most recent command velocity for the vehicle to a Twist variable
        """
        with self.lock:
            self.twist = data

    def joint_cmd_callback(self, data):

        """
        Updates the most recent command velocity for the joints to a Joint State variable
        """
        with self.joint_lock:
            self.joint = data

    def update_position(self, event):

        """
        Computes the pose of the vehicle and the ripper in the terrian map.
        """

        # Create a copy of the most recent stored twist data to perform calculations
        with self.lock:
            velocity_data = copy.deepcopy(self.twist)

        # Time elapsed since last update position call
        if hasattr(event, 'last_real'):
            if event.last_real is None:
                time = rospy.Duration(0.05)
            else:
                time = event.current_real - event.last_real
        
        time = time.to_sec()

        # Calculate angle turned in the given time using omega = theta/time
        angle = velocity_data.angular.z*time

        # Calculate distance travelled in the given time using linear velocity = arc distance/time
        distance = velocity_data.linear.x*time

        # Calculate yaw of the robot
        self.vehicle_yaw += angle

        # Calculate vehicle x, y, z position coordinates
        # TODO recalculate the position based on traveling in a circular arc.
        self.pose.position.x += (distance)*cos(self.vehicle_yaw)
        self.pose.position.y += (distance)*sin(self.vehicle_yaw)

        # Calculate z position using linear interpolation and create cloud array
        
        # 1. Create range to be used in interpolation function
        terrain_points_x = np.arange(0, self.gaussian_array.shape[0]*self.resolution, self.resolution)
        terrain_points_y = np.arange(0, self.gaussian_array.shape[1]*self.resolution, self.resolution)

        # 2. Create cloud array to be converted to point cloud for vizualization
        #TODO do this without for loops
        terrain_grid_points = []
        for i in range(self.gaussian_array.shape[0]):
            for j in range(self.gaussian_array.shape[1]):
                innerlist = []
                innerlist.append(terrain_points_x[i])
                innerlist.append(terrain_points_y[j])
                innerlist.append(self.gaussian_array[i][j])
                terrain_grid_points.append(innerlist)
        
        # 3. Create interpolation function based on the ranges and gaussian data
        interp_func = RectBivariateSpline(terrain_points_x, terrain_points_y, self.gaussian_array)

        # 4. Find z value for x and y coordinate of vehicle using interpolation function
        # TODO compute z height based on footprint
        self.pose.position.z = interp_func(self.pose.position.x, self.pose.position.y)

        # Convert Euler Angles to Quarternion
        V_rotation = tf.transformations.quaternion_from_euler(0.0, 0.0, self.vehicle_yaw)

        # Broadcast vehicle frame which is a child of the world frame
        br = tf.TransformBroadcaster()
        br.sendTransform((self.pose.position.x, self.pose.position.y, self.pose.position.z), 
                        V_rotation, rospy.Time.now(),"vehicle_frame", "map")

        # Construct the homogenous transformation matrix for map to vehicle frame
        V_translation = [self.pose.position.x, self.pose.position.y, self.pose.position.z]
        map_T_V = tf.transformations.quaternion_matrix(V_rotation) 
        map_T_V[:3,3] = np.array(V_translation)

        # Create footprint of vehicle
        V_footprint_range_x = np.linspace((-self.vehicle_length/2), (self.vehicle_length/2), 30)
        V_footprint_range_y = np.linspace((-self.vehicle_width/2), (self.vehicle_width/2), 15)
        V_footprint_mesh_x, V_footprint_mesh_y = np.meshgrid(V_footprint_range_x, V_footprint_range_y)
        V_footprint_x = V_footprint_mesh_x.ravel()
        V_footprint_y = V_footprint_mesh_y.ravel()

        # For every point in the vehicle footprint, calculate the position wrt to the vehicle's frame
        # and its interpolated z value. Add this point to a list of points for visualization.
        # TODO Flatten into a single matrix multiply to remove for loop
        V_viz_points = []
        for i in range(V_footprint_x.shape[0]):
            p = Point()
            V_footprint_point = np.array([[V_footprint_x[i]],[V_footprint_y[i]], [0.0], [1.0]])
            V_footprint_point = np.matmul(map_T_V, V_footprint_point)
            V_footprint_point[2, 0] =  interp_func(V_footprint_point[0, 0], V_footprint_point[1, 0])
            p.x = V_footprint_point[0, 0]
            p.y = V_footprint_point[1, 0]
            p.z = V_footprint_point[2, 0]
            V_viz_points.append(p)

        #####################################################################################
        # Create a copy of the most recent stored JointState data to perform calculations
        with self.joint_lock:
            joint_data = copy.deepcopy(self.joint)

        # If the data is empty on first run, fill with 0.0
        if not joint_data.velocity:
            joint_data.velocity = [0.0,0.0]
        
        # Calculate angle based on velocity data and time
        angle = joint_data.velocity[0]*time
        angle2 = joint_data.velocity[1]*time

        self.joint1_pitch += angle
        self.joint2_pitch += angle2

        # Transformations from vehicle frame to Joint1 and Joint2
    
        # Static rotation about z-axis 
        static_rot = tf.transformations.quaternion_from_euler(0.0, 0.0, 3.14159)
        translation =  [0.0, 0.0, 0.0]
        V_T_SRz = tf.transformations.quaternion_matrix(static_rot)
        V_T_SRz[:3,3] = np.array(translation)

        # Dynamic rotation about the y-axis of Joint 1
        rot_SRz_T_J1 = [[cos(self.joint1_pitch), 0.0, sin(self.joint1_pitch)],
                         [0.0, 1.0, 0.0],
                         [-sin(self.joint1_pitch), 0.0, cos(self.joint1_pitch)]]

        trans_SRz_T_J1 = [0.0, 0.0, 0.0, 1.0]

        SRz_T_J1 = np.zeros((4,4))
        SRz_T_J1[:3,:3] = rot_SRz_T_J1
        SRz_T_J1[:4,3] = trans_SRz_T_J1

        # Translation based on length of Joint 1 arm 
        no_rot = tf.transformations.quaternion_from_euler(0.0, 0.0, 0.0)
        translation =  [self.joint1_length, 0.0, 0.0]
        J1_T_STx = tf.transformations.quaternion_matrix(no_rot)
        J1_T_STx[:3,3] = np.array(translation)

        # Dynamic rotation about y-axis of Joint 2
        dynamic_rot2 = tf.transformations.quaternion_from_euler(0.0, self.joint2_pitch, 0.0)
        translation =  [0.0, 0.0, 0.0]
        STx_T_J2 = tf.transformations.quaternion_matrix(dynamic_rot2)
        STx_T_J2[:3,3] = np.array(translation)

        # matrix multiplication to form the homogenous matrices
        V_T_J1 = np.matmul(V_T_SRz, SRz_T_J1)
        V_T_STx = np.matmul(V_T_J1, J1_T_STx)
        V_T_J2 = np.matmul(V_T_STx, STx_T_J2)

        frame_J1 = tf_conversions.fromMatrix(V_T_J1)
        frame_J2 = tf_conversions.fromMatrix(V_T_J2)

        # The ripper tip is a point in the J2's frame, this is based on the length of the ripper
        ripper_tip_point_J2 =  [self.ripper_length, 0.0, 0.0, 1.0]
        map_T_J2 =  np.matmul(map_T_V, V_T_J2)
        ripper_tip_pt_map =  np.matmul(map_T_J2, ripper_tip_point_J2)
        ripper_tip_point_viz = Point()
        ripper_tip_point_viz.x = ripper_tip_pt_map[0]
        ripper_tip_point_viz.y = ripper_tip_pt_map[1]
        ripper_tip_point_viz.z = ripper_tip_pt_map[2]
        V_viz_points.append(ripper_tip_point_viz)

        # use the ripper's position as an index value to access the gaussian array
        ripper_tip_cell_index_x = int(ripper_tip_pt_map[0]/self.resolution)
        ripper_tip_cell_index_y = int(ripper_tip_pt_map[1]/self.resolution)

        # Create a range of index values surrounding index_x and y
        nearby_index_cells_range_x = np.arange((ripper_tip_cell_index_x-1),(ripper_tip_cell_index_x+2), 1)
        nearby_index_cells_range_y = np.arange((ripper_tip_cell_index_y-1),(ripper_tip_cell_index_y+2), 1)
        nearby_index_cells_mesh_x, nearby_index_cells_mesh_y = np.meshgrid(nearby_index_cells_range_x,nearby_index_cells_range_y)
        nearby_index_cells_x = nearby_index_cells_mesh_x.ravel()
        nearby_index_cells_y = nearby_index_cells_mesh_y.ravel()

        # First check if the index is within the gaussian array, if it is, then check if the tip of
        # the ripper is beneath the soil, if it is, then remove the soil above the tip and disperse
        # it to the surrounding cells, provided those cells are also within the gaussian array
        # TODO Remove use of for loops and excess if statements

        if (0 <= ripper_tip_cell_index_x <= (self.gaussian_array.shape[0]-1)) and (0 <= ripper_tip_cell_index_y <= (self.gaussian_array.shape[1]-1)):
            if (self.gaussian_array[ripper_tip_cell_index_x][ripper_tip_cell_index_y] > ripper_tip_pt_map[2]):
                diff = self.gaussian_array[ripper_tip_cell_index_x][ripper_tip_cell_index_y] - ripper_tip_pt_map[2]
                for i in range(nearby_index_cells_x.shape[0]):
                    if (0 <= nearby_index_cells_x[i] <= (self.gaussian_array.shape[0]-1)) and (0 <= nearby_index_cells_y[i] <= (self.gaussian_array.shape[1]-1)):
                            self.gaussian_array[nearby_index_cells_x[i]][nearby_index_cells_y[i]] += diff/8
                self.gaussian_array[ripper_tip_cell_index_x][ripper_tip_cell_index_y] = ripper_tip_pt_map[2]

        # Publish all messages
        self.publish_messages(V_translation, V_rotation, terrain_grid_points, V_viz_points, frame_J1, frame_J2)

    def publish_messages(self, V_translation, V_rotation, terrain_grid_points, V_viz_points, frame_J1, frame_J2):

        """
        Publishes the pose stamped, multi-array, point-cloud and vehicle footprint vizualization
        marker. 
        """

        ##################################################################################

        # Create a posestamped message containing position information

        # Create pose message
        msg = PoseStamped()

        # Header details for pose message
        msg.header.frame_id = "map"
        msg.header.stamp = rospy.Time.now()

        # Pose information
        msg.pose.position.x = V_translation[0]
        msg.pose.position.y = V_translation[1]
        msg.pose.position.z = V_translation[2]
        msg.pose.orientation.x = V_rotation[0]
        msg.pose.orientation.y = V_rotation[1]
        msg.pose.orientation.z = V_rotation[2]
        msg.pose.orientation.w = V_rotation[3]


        ##################################################################################

        # Create an multi array message containing pose information

        # Create array message
        array_msg = Float32MultiArray()
        array_msg.layout.dim.append(MultiArrayDimension())
        array_msg.layout.dim[0].label = "vehicle_position"
        array_msg.layout.dim[0].size = 3
        array_msg.layout.dim[0].stride = 3

        # Append data
        array_msg.data.append(V_translation[0])
        array_msg.data.append(V_translation[1])
        array_msg.data.append(V_translation[2])

        ##################################################################################

        # Create point cloud and publish to rviz

        # Create a point cloud from the xyz values in the array list
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'map'
        point_cloud = pcl2.create_cloud_xyz32(header, terrain_grid_points)

        ##################################################################################

        # Create a marker to vizualize the footprint of the vehicle
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
        viz_points.points = V_viz_points


        ################################################################

        # Create pose message for joints 1 & 2
        msg1 = PoseStamped()
        msg2 = PoseStamped()

        # Header details for pose message
        msg1.header.frame_id = "vehicle_frame"
        msg1.header.stamp = rospy.Time.now()

        msg2.header.frame_id = "vehicle_frame"
        msg2.header.stamp = rospy.Time.now()

        # Pose information
        joint_1 = tf_conversions.toMsg(frame_J1)
        joint_2 = tf_conversions.toMsg(frame_J2)
        
        msg1.pose = joint_1
        msg2.pose = joint_2

        # Publish pose, vizualization, array information and point cloud
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







