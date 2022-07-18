#!/usr/bin/env python

import rospy
import tf
import tf_conversions
import tf2_ros
import geometry_msgs
import numpy as np
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Quaternion




class IMU_ENU_repub(object):

    def __init__(self):

        # Get ROS general parameters
        # Define topics for raw imu data
        self.imuraw_topic_input = '/imu/data'
        self.imuraw_topic_output = '/imu/data_ENU'

        # self.imutopic_input = 'husky_b1/gx5/imu/data'
        # self.imutopic_output = 'husky_b1/gx5/imu/data_ENU'

        # Set up ROS publisher/subscriber
        self.sub = rospy.Subscriber(self.imuraw_topic_input, Imu, self._state_callback_raw, queue_size=1)      # To get estimate pose and velocity of vehicle

        self.pub_raw = rospy.Publisher(self.imuraw_topic_output,Imu, queue_size=1)

    def _state_callback_raw(self, data):

        # Function to calculate Euler angle from quaternion in IMU data message
        # Build quaternion from orientation
        quaternion_NED = [
        data.orientation.x,
        data.orientation.y,
        data.orientation.z,
        data.orientation.w]
        # Find all Euler angles from quaternion transformation
        euler_ENU = tf.transformations.euler_from_quaternion(quaternion_NED)

        # Rotate NED Euler angles to ENU Euler angles
        # roll_ENU = euler_ENU[1]
        # pitch_ENU = euler_ENU[0]

        roll_ENU = 0.0
        pitch_ENU = 0.0
        yaw_ENU = -euler_ENU[2] + np.pi/2
        #misan: angles should be btw -180<yaw<+180, but this equationcan fall out of this range
                # e.g. when ENU[2] is -150 yaw_ENU works out to +240
                # BUT tf will handle angles >+180 apprpriately(?) so need for if loops to handle these(?)

        # Convert ENU Euler angle to quaternion for publishing
        euler_ENU = [roll_ENU, pitch_ENU, yaw_ENU]
        quaternion_ENU = tf.transformations.quaternion_from_euler(roll_ENU, pitch_ENU, yaw_ENU,axes='sxyz')


        # Build Message to transform
        imudataENU = Imu()
        imudataENU.header.seq = data.header.seq
        imudataENU.header.stamp = data.header.stamp
        imudataENU.header.frame_id = 'base_link'

        # Republish quaternion after rotation to ENU
        imudataENU.orientation.x = quaternion_ENU[0]
        imudataENU.orientation.y = quaternion_ENU[1]
        imudataENU.orientation.z = quaternion_ENU[2]
        imudataENU.orientation.w = quaternion_ENU[3]
        imudataENU.orientation_covariance = data.orientation_covariance
        # Republish angular velocities from NED to ENU
        imudataENU.angular_velocity.x = data.angular_velocity.y
        imudataENU.angular_velocity.y = data.angular_velocity.x
        imudataENU.angular_velocity.z = -data.angular_velocity.z
        imudataENU.angular_velocity_covariance = data.angular_velocity_covariance

        # Republish accelerations from NED to ENU
        imudataENU.linear_acceleration.x = data.linear_acceleration.y
        imudataENU.linear_acceleration.y = data.linear_acceleration.x
        imudataENU.linear_acceleration.z = -data.linear_acceleration.z
        imudataENU.linear_acceleration_covariance = data.linear_acceleration_covariance
        
        # Publish IMU data that is now in ENU
        self.pub_raw.publish(imudataENU)


if __name__ == '__main__':
    # Initialize ROS
    rospy.init_node('IMU_ENU_repub')

    # Create the vehicle odometry object
    IMU_ENU_repub()
    
    # Wait for messages on topics, go to callback function when new messages arrive.
    rospy.spin()
