#!/usr/bin/env python

import rospy
import copy
import numpy as np

# Importing ROS messages from standard ROS messages
from std_msgs.msg import Empty, Bool, Int8, Float32MultiArray
from near_to_far_sim.msg import IMU_trigger
from sensor_msgs.msg import Imu
from near_to_far_sim.msg import Num


class Record_IMU_Data(object):

    def __init__(self):

        # **** Initialize variables from handlers ****
        self.number_of_clusters = None
        self.cluster_IMU_readings = None
        self.cluster_numbers = 0
        self.start = False
        self.count = 0

        self.sub = rospy.Subscriber('driveable', Int8, self.set_no_of_clusters, queue_size=1)       # topic to know number of driveable clusters
        rospy.Subscriber('get_IMU_data', IMU_trigger, self.start_or_stop, queue_size=1)
        rospy.Subscriber('imu/data', Imu, self.imu_readings, queue_size=1)

        self.pub_imu = rospy.Publisher('cluster_acclns', Num , queue_size=1, latch=True)

        # Define timer for callback function
        self.T = rospy.get_param('~rate', 0.1)
        self.timer = rospy.Timer(rospy.Duration(self.T), self.timer_callback)

    # -------------------------- Sensor Data Handlers --------------------------------
    # Callbacks
    def set_no_of_clusters(self, data):
        self.number_of_clusters = copy.deepcopy(data.data)
        print(self.number_of_clusters)
        self.cluster_IMU_readings = np.zeros(self.number_of_clusters)
        print(self.cluster_IMU_readings)
        # self.sub.unregister()
        # print('unregistered')

    def start_or_stop(self, data):
        position = copy.deepcopy(data.start_stop)
        if position=='start':
            self.start = True
            self.cluster_numbers = copy.deepcopy(data.cluster_number)
            self.n = (self.cluster_numbers / 2) - 1
            # the clusters to be driven over for IMU data are 2,4,6...(cos path 0 is pth thru blind spot and path 1 is path to first cluster), so use this as index in the cluster_IMU_readings array

        elif position=='stop':
            self.start = False

    def imu_readings(self, data):
        self.imu_reading = copy.deepcopy(data)


    def timer_callback(self, event):            # get IMU data

        # Conditional statement to wait until trigger has been recieved - Only relevant when wanting to start controller on remote trigger
        if self.number_of_clusters is None:
            rospy.logwarn_throttle(2.5, "Specify number of driveable cluster...")
            return

        # Conditional statement to wait until path has been received
        if self.start is False:
            rospy.logwarn_throttle(2.5, "Not IMU path...")
            self.count = 0
            return

        if self.start is True:
            rospy.logwarn_throttle(2.5, "IMU path...")
            self.count = 0

        while self.start == True:

            self.count = self.count+1

            self.cluster_IMU_readings[self.n] = self.cluster_IMU_readings[self.n] + np.sqrt((self.imu_reading.linear_acceleration.x)**2 + (self.imu_reading.linear_acceleration.y)**2)

        # Publish data continuously...but, in each cycle, only the last published message is relevant

        print(self.cluster_IMU_readings)

        self.cluster_IMU_readings[self.n] = np.exp(8*(self.cluster_IMU_readings[self.n]/self.count))    #get the mean
        imu = Num()
        imu.cluster_attributes = copy.deepcopy(self.cluster_IMU_readings)           #or just use an array message from std_msgs

        print(self.cluster_IMU_readings)
        self.pub_imu.publish(imu)


if __name__ == '__main__':
    # init()
    rospy.init_node('record_IMU')

    # Create the vehicle path following object
    Record_IMU_Data()

    # Wait for messages on topics, go to callback function when new messages arrive.
    rospy.spin()