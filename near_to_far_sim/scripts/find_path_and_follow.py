#!/usr/bin/env python
from __future__ import print_function
import cv2
import numpy as np
import math as mt
import tf
import copy
import rospy
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import Pose, PointStamped, PoseStamped, Twist
from std_msgs.msg import Empty, Bool, Int8, Int16, Float32MultiArray
from near_to_far_sim.msg import IMU_trigger, Num
from sensor_msgs.msg import Joy, Image
from cv_bridge import CvBridge, CvBridgeError
import matplotlib.pyplot as plt
import time
from sensor_msgs.msg import Joy


class Find_Path_and_Follow(object):
    def __init__(self):
        self.im = None
        self.bridge = CvBridge()
        self.transf_image = None
        self.cluster_color_features = None

        self.true_path = []
        self.actual_path_pts = []
        self.err = []

        self.errx = []
        self.erry = []
        self.wait_hold = None

        self.robot_pose = None
        self.robot_start_x = 0
        self.robot_start_y = 0
        self.yaw = 0
        self.scale = 1             #xx...resizing factor...if pix was resized to world size/xx

        # **** Initialize variables from handlers ****
        self.prev_time = None
        self.paths = None
        self.path = None
        self.est_pose = None
        self.vel = None
        self.vib_data = None
        self.c = 0
        self.flag = 0
        self.dist2points3 = None
        self.drivable_sized_cluster = 0

        # Initalize options
        # FBL controller params
        self.sat_omega_lim = 2      # Saturating steering rate at 2 rad/s
        # Operating velocities
        self.v_opt = 0.5
        self.v_min = 0.25
        # Controller Gain
        omegaCL = rospy.get_param('~omega_cl', 0.7)  # Closed-loop bandwidth for FBL controller
        zeta = rospy.get_param('~zeta', 1.5)  # Damping coeffient
        self.kp = -omegaCL ** 2  # Proportional gain
        self.kd = -2 * omegaCL * zeta  # Differential gain

        self.T = rospy.get_param('~rate', 0.1)
        self.timer = rospy.Timer(rospy.Duration(self.T), self.timer_callback)

        # Topics that controller is subscribing to
        rospy.Subscriber('label_no_and_IMU', Num, self.callback, queue_size=1)
        rospy.Subscriber('odometry/filtered', Odometry, self.get_pose, queue_size=1)      # To get estimate pose and velocity of vehicle
        rospy.Subscriber('cluster_acclns', Num, self.get_vib_data, queue_size=1)           #subscribe to IMU vib data from record_IMU_data.py
        self.transf_image_sub = rospy.Subscriber('trans_image', Image, self.transf_image_callback, queue_size=1)

        # Topics to that controller is publishing to
        self.pub_driveable = rospy.Publisher('driveable', Int8, queue_size=1, latch=True)           #latch? Number of driveable-sized cluters
        self.pub_path_points_followed = rospy.Publisher('path_points_followed', Int16, queue_size=1, latch=True)  # latch?
        self.pub_twist = rospy.Publisher('husky_velocity_controller/cmd_vel', Twist, queue_size=100)                  # Publish to command velocities (v, omega) to the vel controller
        self.pub_empty = rospy.Publisher('starter', Empty, queue_size=1)
        self.pub_actual_points = rospy.Publisher('actual_robot_path_pts', Num, queue_size=1, latch=True)
        self.lat_head_error = rospy.Publisher('lateral_heading_error', Num, queue_size=1, latch=True)
        self.pub_errx = rospy.Publisher('errxx', Num, queue_size=1, latch=True)
        self.pub_erry = rospy.Publisher('erryy', Num, queue_size=1, latch=True)
        self.pub_planned_paths_2 = rospy.Publisher('planned_paths_2', Num, queue_size=1, latch=True)

    def callback(self, data):
        self.im = copy.deepcopy(data.labelled)
        self.label_no_and_vib_data = copy.deepcopy(data.cluster_attributes)
        self.label_no_and_vib_data = np.reshape(self.label_no_and_vib_data, (-1, 2))

    def transf_image_callback(self, data):
        try:
            pass
        except Exception as err:
            print (err)
        self.transf_image = self.bridge.imgmsg_to_cv2(data, "bgr8")

    def get_vib_data(self, data):
        self.vib_data = data.cluster_attributes

    def timer_callback(self, event):

        if self.im is None:
            rospy.logwarn_throttle(2.5, "Controller waiting for labelled image...")
            return

        if self.est_pose is None:             # is None....no position data from Vicon or EKF...ceck appropriate topic
            rospy.logwarn_throttle(2.5, "... waiting for pose...")
            return

        if self.transf_image is None:
            rospy.logwarn_throttle(2.5, "... waiting for transformed image for visualization...")
            return

        im = copy.deepcopy(self.im)
        im = np.asarray(im)
        im = im.astype(np.uint8)
        im = np.reshape(im, (np.int(350/self.scale), np.int(200/self.scale)))   # gray     i.e maxheight by maxwidth

######################################################################################################################

        # A* Path Planner Starts

#######################################################################################################################

        # Next pass start and end for each section to Astar and concatenate...
        # i.e Astar[0], coors_for_IMU[0], Astar[], coords_for_IMU[1]... etc

        Current_Robot_Position = [im.shape[0]-1, np.int(0.5 *(im.shape[1]-1)+1)]
        Final_Position = [0, np.int(0.5*(im.shape[1]-1)+1)]

        start_end_coords = []

        start_end_coords.append(Current_Robot_Position)

        start_end_coords.append(Final_Position)

        # print(start_end_coords)
        # #######################################################################################################################
        # # Astar starts...
        # #######################################################################################################################
        OptimalPaths = []

        # Preallocation of Matrices

        Connecting_Distance = 1  # 8 possible directions....higher won't work unless you pre-declare obstacles
        MAP = im  # ld super_cluster image
        Start = start_end_coords[0]
        Goal = start_end_coords[1]  # intersection between heading and the zero row lne (i=0)
        # print(Goal)
        Goal =[Goal[0], Goal[1]]
        GoalRegister = np.zeros_like(MAP, dtype=np.int8)
        GoalRegister[Goal[0], Goal[1]] = 1
        StartX = Start[1]  # review if it should be StartY, STartX...from GPS or VIcon (at time Image was taken?)
        StartY = Start[0]

        # [Height,Width] = MAP.size              # Height and width of matrix
        GScore = np.full_like(MAP, np.inf, dtype=np.float32)  # Matrix keeping track of G-scores...iniialise with 1e10 so that any missed pixel is not zero
                            #change 1e10 appropritely to make unclassified pixels obstacles (e.g.1e10) or free pixels (e.g. 1e2) determine the better approach
        FScore = np.full_like(MAP, np.inf, dtype=np.float32)  # Matrix keeping track of F-scores [only open list]
        Hn = np.zeros_like(MAP, dtype=np.float32)  # Heuristic matrix
        OpenMAT = np.zeros_like(MAP, dtype=np.int8)  # Matrix keeping of open grid cells
        ClosedMAT = np.zeros_like(MAP, dtype=np.int8)  # Matrix keeping track of closed grid cells
        ParentX = np.zeros_like(MAP, dtype=np.int16)  # Matrix keeping track of X position of parent
        ParentY = np.zeros_like(MAP, dtype=np.int16)  # Matrix keeping track of Y position of parent

        label_no_and_vib = copy.deepcopy(self.label_no_and_vib_data.tolist())

        for i in range(0, len(label_no_and_vib)):
            GScore[MAP == label_no_and_vib[i][0]] = np.int(label_no_and_vib[i][1])        # TO DO: evaluate GSCORE costs based on IMU metric
        Test = GScore==np.inf
        GScore[Test] = np.amax(GScore)      #make noisy classes equal the min cost

        ### Setting up matrices representing neighbours to be investigated

        NeighboorCheck = np.ones((2 * Connecting_Distance + 1, 2 * Connecting_Distance + 1), dtype=np.int)
        Dummy = (2 * Connecting_Distance + 2) - 1
        # print(Dummy)
        Mid = Connecting_Distance
        # print(Mid)

        for i in range(0, Connecting_Distance - 1):
            NeighboorCheck[i, i] = 0
            NeighboorCheck[Dummy - i, i] = 0
            NeighboorCheck[i, Dummy - i] = 0
            NeighboorCheck[Dummy - i, Dummy - i] = 0
            NeighboorCheck[Mid, i] = 0
            NeighboorCheck[Mid, Dummy - i] = 0
            NeighboorCheck[i, Mid] = 0
            NeighboorCheck[Dummy - i, Mid] = 0

        NeighboorCheck[Mid, Mid] = 0

        # Get fortran style, column-major indices...like matlab
        non_zero = np.reshape(NeighboorCheck, (NeighboorCheck.shape[0] * NeighboorCheck.shape[1], 1), order='F')

        non_zero = np.nonzero(non_zero > 0)

        row = non_zero[0] % NeighboorCheck.shape[0]

        column = non_zero[0] / NeighboorCheck.shape[0]

        Neighboors = np.array([row, column]) - (Connecting_Distance)

        N_Neighboors = len(row)  # and check other as in StartX, startY!

        cell_size = 0.5
        # for a unit cell size: hori.and vert. distances are 1; and diag distance is sqrt.(2)....
        # report that it is taken from Howie Choset et al.'s book

        for k in range(0, MAP.shape[0]):
            for j in range(0, MAP.shape[1]):

                if np.abs(k - Goal[0]) == np.abs(j - Goal[1]):
                    Hn[k, j] = 1.414 * cell_size * np.abs(k - Goal[0])

                elif np.abs(k - Goal[0]) > np.abs(j - Goal[1]):
                    delta = np.abs(np.abs(k - Goal[0]) - np.abs(j - Goal[1]))
                    Hn[k, j] = 1.414 * cell_size * np.abs(j - Goal[1]) + 1 * cell_size * delta

                elif np.abs(k - Goal[0]) < np.abs(j - Goal[1]):
                    delta = np.abs(np.abs(k - Goal[0]) - np.abs(j - Goal[1]))
                    Hn[k, j] = 1.414 * cell_size * np.abs(k - Goal[0]) + 1 * cell_size * delta


        # Initializign start node with FValue and opening first node.
        FScore[StartY, StartX] = Hn[StartY, StartX]
        OpenMAT[StartY, StartX] = 1
        # print (OpenMAT)

        while True:  # Code will break when path found or when no path exist
            MINopenFSCORE = np.amin(FScore)  # ...check, use np.min(Fscore.flatten())...or use np.amax(a, axis=none)
            # print(MINopenFSCORE)
            if MINopenFSCORE == np.inf:
                # Failuere!
                OptimalPath = [np.inf]
                RECONSTRUCTPATH = 0
                break

            # Get fortran style, column-major indices...like matlab

            Current = FScore == MINopenFSCORE
            Current = np.reshape(Current, (FScore.shape[0] * FScore.shape[1], 1), order='F')

            Current = np.nonzero(Current > 0)

            row = Current[0] % FScore.shape[0]

            column = Current[0] / FScore.shape[0]

            CurrentY = row[0]  # check if the other way round... like StartY, StartX
            CurrentX = column[0]


            if GoalRegister[CurrentY, CurrentX] == 1:
                print('GOAL')
                RECONSTRUCTPATH = 1
                break

            # Removing node from OpenList to ClosedList
            OpenMAT[CurrentY, CurrentX] = 0
            FScore[CurrentY, CurrentX] = np.inf
            ClosedMAT[CurrentY, CurrentX] = 1

            for p in range(0, N_Neighboors):
                i = Neighboors[0, p]  # Y
                j = Neighboors[1, p]  # X

                if CurrentY + i < 0 or CurrentY + i > MAP.shape[0] - 1 or CurrentX + j < 0 or CurrentX + j > MAP.shape[
                    1] - 1:
                    # print('here')
                    continue

                if ClosedMAT[CurrentY + i, CurrentX + j] == 0:  # Neiboor is open

                    # tentative_gScore = GScore[CurrentY, CurrentX] + GScore[CurrentY + i, CurrentX + j]    #+ mt.sqrt(i ** 2 + j ** 2)  #old use for ICRA etc
                    tentative_gScore = GScore[CurrentY, CurrentX] + GScore[CurrentY + i, CurrentX + j] + cell_size * mt.sqrt(i ** 2 + j ** 2)  # new...plan to use this for thesis
                    # print(tentative_gScore)
                    # print(GScore[CurrentY + i, CurrentX + j])

                    if OpenMAT[CurrentY + i, CurrentX + j] == 0:
                        OpenMAT[CurrentY + i, CurrentX + j] = 1
                    elif tentative_gScore > GScore[CurrentY + i, CurrentX + j] or tentative_gScore == GScore[
                        CurrentY + i, CurrentX + j]:  # check condition >= correctly written?
                        # print('here again')
                        continue

                    ParentX[CurrentY + i, CurrentX + j] = CurrentX
                    ParentY[CurrentY + i, CurrentX + j] = CurrentY
                    GScore[CurrentY + i, CurrentX + j] = tentative_gScore
                    FScore[CurrentY + i, CurrentX + j] = tentative_gScore + Hn[CurrentY + i, CurrentX + j]
                    # print('here again again')

        # print(OptimalPath)

        if RECONSTRUCTPATH:
            OptimalPath = []
            OptimalPath.append([CurrentY,CurrentX])
            # print(CurrentY, CurrentX)
            while RECONSTRUCTPATH:
                CurrentXDummy = ParentX[CurrentY,CurrentX]
                CurrentY = ParentY[CurrentY,CurrentX]
                CurrentX = CurrentXDummy
                OptimalPath.append([CurrentY,CurrentX])

                if CurrentX==StartX and CurrentY==StartY:
                    break

        OptimalPaths.append(OptimalPath[::-1])          #flipped to give arrays as start to end


        for Paths in OptimalPaths:
            # OptimalPaths = OptimalPath[::-1]  # make it start to end...was returned as end to start
            for i in range(0, len(Paths)):
                if i == len(Paths) - 1:
                    continue
                cv2.line(self.transf_image, (Paths[i][1], Paths[i][0]), (Paths[i + 1][1], Paths[i + 1][0]),        #just testing a different color
                         (0, 255, ), 2)
                cv2.imshow("outline", self.transf_image)
                cv2.waitKey(5)
        cv2.imshow("outline2", self.transf_image)
        cv2.waitKey(10)
        # cv2.waitKey(0)
        cv2.destroyAllWindows()

        filename = str(time.clock())+'.png'
        cv2.imwrite(filename, self.transf_image)

        image_to_global = [self.to_global(x) for x in OptimalPaths]
        print(len(image_to_global))

        paths_to_follow = [self.add_heading(x) for x in image_to_global]

        # Now add initial path from vehice to start of image
        loc = self.est_pose
        self.robot_pose = self.current_pose(loc)

        # # global robot pose message from Vicon or  Odom_filtered_map EKF_map (ROS Robot_Localization package) in cm
        X_G_R = self.robot_pose[0]
        Y_G_R = self.robot_pose[1]
        angle = self.robot_pose[2]

        self.robot_start_x = X_G_R
        self.robot_start_y = Y_G_R

        robot_to_image_start = np.sqrt((paths_to_follow[0][0][0]-(self.robot_start_x*100))**2 + (paths_to_follow[0][0][1]-(self.robot_start_y*100))**2)

        x_y_completez = []

        for i in np.arange(np.ceil(robot_to_image_start), -1, -1):
            x = paths_to_follow[0][0][0] - ((i * 1.0 / (robot_to_image_start)) * (paths_to_follow[0][0][0]-(self.robot_start_x*100)))
            y = paths_to_follow[0][0][1] - ((i * 1.0 / (robot_to_image_start)) * (paths_to_follow[0][0][1]-(self.robot_start_y*100)))


            x_y_completez.append([x, y])

        x_y_completezz = [self.add_heading(x_y_completez)]

        final_paths_to_follow = x_y_completezz + paths_to_follow              # add the path list to start of paths list of list....see 'list concatenation'

        #publish path points
        planned = []
        for path in final_paths_to_follow:
            one_planned = np.asarray(path)
            one_planned  = one_planned.flatten()
            one_planned = one_planned.tolist()
            planned = planned + one_planned

        print(planned)

        pub_planned  = Num()
        pub_planned.cluster_attributes = planned
        self.pub_planned_paths_2.publish(pub_planned)

        self.wait_hold = [1]  # just for printing data...stop loop here so it doesn't keep going round with the playing nag's msg


        plt.style.use("ggplot")
        plt.figure()

        for n in range(0, len(final_paths_to_follow)):
            xs = [i[0] for i in final_paths_to_follow[n]]
            ys =[i[1] for i in final_paths_to_follow[n]]

            if n == len(final_paths_to_follow) - 1:  # only label last plot...to avoid multiple labels
                plt.plot(xs, ys, '-o', color="black", markeredgecolor="black", label="planned path")
            else:
                plt.plot(xs, ys, '-o', color="black", markeredgecolor="black")

        # plt.title("Final_map_paths")
        fontsize = 27
        plt.legend()
        plt.xlabel("x (cm)", fontsize=fontsize)
        plt.ylabel("y (cm)", fontsize=fontsize)
        # ax.set_title(title, fontsize=fontsize)
        plt.tick_params(labelsize=fontsize)
        plt.tight_layout()
        # plt.savefig("figs/final_map_paths.png")
        # plt.show()
    # #######################################################################################################################################
    #
    # #controller loop
    #
    # ######################################################################################################################################
        final_pathz_to_follow = copy.deepcopy(final_paths_to_follow)
        # ... for calculating steering commands
        for i in range(0, len(final_pathz_to_follow)):
            # print('newloop')
            # final_path_to_follow = final_paths_to_follow[i]
            # self.path = final_path_to_follow

            self.path = final_pathz_to_follow[i]

            # print(i)
            # print(len(final_pathz_to_follow)-1)
            # print(len(self.path))

            loc = self.est_pose
            pos = self.current_pose(loc)
            robot = pos[2]
            target_pos = self.path[0][2]
            twist = Twist()
            twist.linear.x = 0.0
            yaw_rate = 0.7

            if target_pos>=0 and robot>=0:
                if target_pos>robot:
                    twist.angular.z = yaw_rate         #+ve ccw
                else:
                    twist.angular.z = -yaw_rate

            if target_pos<=0 and robot<=0:
                if target_pos<robot:
                    twist.angular.z = -yaw_rate         #-ve ccw
                else:
                    twist.angular.z = yaw_rate

            if target_pos<=0 and robot>=0:
                Ntarget_pos = 360 - abs(target_pos)
                if (Ntarget_pos-robot)>=180:
                    twist.angular.z = -yaw_rate
                else:
                    twist.angular.z = yaw_rate

            if target_pos>=0 and robot<=0:
                Nrobot = 360 - abs(robot)
                if (Nrobot-target_pos)>=180:
                    twist.angular.z = yaw_rate
                else:
                    twist.angular.z = -yaw_rate

            # rotate Husky to new pose before
            while True:
                loc = self.est_pose
                q = self.current_pose(loc)
              #Husky's diff drive controller responds to this value (from using joystick and echoing Husky/cmd_vel topic)...try 0.8
                self.pub_twist.publish(twist)

                # if (q[2]/self.path[0][2] >= 0.85):              # try 0.9 or 0.95... and use subtraction instead to avoid division by zero
                if abs(q[2] - self.path[0][2]) < (3.142/36):    # i.e if the difference is within 5 deg either way...
                    twist.angular.z = 0.0                       # tomake sure 5 deg is not too small try smaller (than 1) twist.z values for smaller rotation/step
                    self.pub_twist.publish(twist)
                    break
            time.sleep(1)

                # choose which direction to rotate Husky

            while True:

                # ------------- Start polling data ------------
                # Define sensor data at the beginning of the callback function to be used throughout callback function

                # Define velocity of vehicle
                self.v = self.vel

                # Singularity prevention.
                # If speed is low, omega is nearing infinity. In this case, we just make it self.v_min
                if self.v < self.v_min:
                    self.v = self.v_min

                # Define location of the vehicle within timer function
                loc = self.est_pose

                # Establishing the current state of the unicycle vehicle as (x, y, theta)
                q = self.current_pose(loc)

                robot_actual = Num()
                robot_actual.cluster_attributes = q
                self.pub_actual_points.publish(robot_actual)

                self.actual_path_pts.append(q)

                # Calculate the lateral and heading error of the vehicle where e = [el, eh]
                e = self.path_errors(q)

                follow_error = Num()
                follow_error.cluster_attributes = e
                self.lat_head_error.publish(follow_error)

                # Calculate unicycle steering rate (omega) in rad/s
                self.omega = self.fbl_controller(q, e)

                # publish to check how closest points are spaced
                self.pub_path_points_followed.publish(self.c)


                if self.c == (len(self.path) - 1):

                    if i < (len(final_pathz_to_follow) - 1):
                        # if at the end of current path but not at the end of all paths, go to next path

                        # Build twist message to /cmd_vel
                        twist = Twist()
                        twist.linear.x = v_opt
                        twist.angular.z = self.omega
                        self.pub_twist.publish(twist)

                        rospy.logwarn_throttle(2.5, "At the end of the interim path! Pause!")

                        self.c = 0              # reset the count

                        break

                    else:
                        rospy.logwarn_throttle(2.5, "At the end of the final path! Stop!")

                        # Build twist message to /cmd_vel
                        twist = Twist()
                        twist.linear.x = 0.0
                        twist.angular.z = 0.0
                        self.pub_twist.publish(twist)

                        self.c = 0          # reset the count

                        print('sleeping for 1 sec')
                        time.sleep(1)

                        break  # check

            # ---------------- Publishing Space ---------------

                # Build twist message to /cmd_vel
                v_opt=self.v_opt
                twist = Twist()
                twist.linear.x = v_opt
                twist.angular.z = self.omega
                self.pub_twist.publish(twist)

        # publish errors

        errxx = np.asarray(self.errx)
        erryy = np.asarray(self.erry)

        errxxx = Num()
        errxxx.cluster_attributes = errxx
        self.pub_errx.publish(errxxx)

        erryyy = Num()
        erryyy.cluster_attributes = erryy
        self.pub_erry.publish(erryyy)

        # rotate Husky for reverse run...ONLY IN the LAB! Outdoors just keep moving

        while True:
            # Build twist message to /cmd_vel

            loc = self.est_pose
            q = self.current_pose(loc)
            twist = Twist()
            twist.linear.x = 0.0
            twist.angular.z = 0.4
            self.pub_twist.publish(twist)

            # if (q[2]/self.path[0][2] >= 0.85):
            # if q[2] - final_pathz_to_follow[0][0][2] < 3.142:  # facing reverse of starting position or...
            if q[2] < 0.04 and q[2] > -0.04:  # ... just use a small range around -175 and -179 deg with a small twist
                                                # command to avoid the transtion from -179.9999 to +179.999. (Husky stared at zero pose)....FOR INDOORS when u went zero to 180 start and reverse run start poses resp.
            # if abs(q[2] - start_pose) < (3.142 / 36):      # i.e if the difference is within 5 deg either way...
                twist.angular.z = 0.0
                self.pub_twist.publish(twist)
                break
        time.sleep(5)

    ##############################################################################################
        # Plot paths
    #############################################################################################
        plt.style.use("ggplot")
        plt.figure()

        for n in range(0, len(final_paths_to_follow)):
            xs = [i[0] for i in final_paths_to_follow[n]]
            ys = [i[1] for i in final_paths_to_follow[n]]

            if n == len(final_paths_to_follow) - 1:  # only label last plot...to avoid multiple labels
                plt.plot(xs, ys, '-o', color="black", markeredgecolor="black", label="planned path")
            else:
                plt.plot(xs, ys, '-o', color="black", markeredgecolor="black")

        # xs = [i[0] for i in self.true_path]
        # ys = [i[1] for i in self.true_path]
        #
        # plt.plot(xs, ys, '-o', color="green", markeredgecolor="darkred")

        xs = [i[0]*100 for i in self.actual_path_pts]
        ys = [i[1]*100 for i in self.actual_path_pts]

        plt.plot(xs, ys, '-o', color="red", markeredgecolor="green",label="actual path")

        fontsize = 22
        plt.legend(fontsize = 22)
        plt.xlabel("x (cm)", fontsize=fontsize)
        plt.ylabel("y (cm)", fontsize=fontsize)
        # ax.set_title(title, fontsize=fontsize)
        plt.tick_params(labelsize=fontsize)
        plt.tight_layout()
        plt.savefig("{}.png".format(time.clock()))
        # plt.show()

    ###############################################################################################
        self.errx = []
        self.erry = []
        self.actual_path_pts = []  # reinitialise...clear appended paths in list for next run
        self.im = None

    ######################################################################################################################################

    def to_global(self, Path_N):          # convert paths from image to global coordinates


        loc = self.est_pose
        self.robot_pose = self.current_pose(loc)

        # # global robot pose message from Vicon or  Odom_filtered_map EKF_map (ROS Robot_Localization package) in cm
        X_G_R = self.robot_pose[0]*100
        Y_G_R = self.robot_pose[1]*100
        angle = self.robot_pose[2]

        global_coords = []


        for i in range(0, len(Path_N)):

            one_global_coord = []

            U = self.scale*Path_N[i][1]                # back to pixel coordinates...if resized to half size
            V = self.scale*Path_N[i][0]

            image_to_robot = np.matmul(np.array([[1, 0, 588], [0, 1, 122.5], [0, 0, 1]]), np.array([[-V], [-U], [1]]))         #change 269, 42 from calibration

            robot_to_global = np.matmul(np.array([[np.cos(angle), -np.sin(angle), X_G_R ], [np.sin(angle), np.cos(angle), Y_G_R ], [0, 0, 1]]),image_to_robot)

            one_global_coord.append(robot_to_global[0])
            one_global_coord.append(robot_to_global[1])

            global_coords.append(one_global_coord)

        return global_coords


    def add_heading(self,Path_n):    # add heading

        complete_with_heading = []

        for i in range (0, len(Path_n)-1):

            with_heading = []
            with_heading.append(Path_n[i][0])                       # x in robot coordinates
            with_heading.append(Path_n[i][1])                       # y in robot coordinates
            with_heading.append(np.arctan2(Path_n[i+1][1]-Path_n[i][1], Path_n[i+1][0]-Path_n[i][0]))     # heading

            complete_with_heading.append(with_heading)

        # for the last path point, copy the previous heading...for now
        with_heading = []
        with_heading.append(Path_n[-1][0])  # x in robot coordinates
        with_heading.append(Path_n[-1][1])  # y in robot coordinates
        with_heading.append(copy.deepcopy(complete_with_heading[-1][2]))

        complete_with_heading.append(with_heading)

        return complete_with_heading

        # Initalization for Feedback Linearization Controller

    ########################################################################################################################

        #controller stuff
    ########################################################################################################################

    # -------------------------- Utility Functions --------------------------------
    # Function to calculate yaw angle from quaternion in Pose message
    def get_yaw(self, pose):
        # Build quaternion from orientation
        quaternion = (
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w)
        # Find all Euler angles from quaternion transformation
        euler = tf.transformations.euler_from_quaternion(quaternion)
        yaw = euler[2]  # Extract yaw angle from Euler angles in radians

        return yaw

    # Function to saturate steering rate (omega)
    def omega_sat(self, omega):
        # Saturating steering rate when not at steering limits
        if abs(omega) > self.sat_omega_lim:
            omega_fix = self.sat_omega_lim * np.sign(omega)
        else:
            omega_fix = omega

        return omega_fix

    # This function is to build an array of the unicycle vehicle state
    def current_pose(self, loc):
        # The state is [x, y, theta] in [m, m, rad]
        x = loc.pose.position.x
        y = loc.pose.position.y
        theta = self.get_yaw(loc.pose)
        q = np.array([x, y, theta])

        return q

    # Function to find lateral and heading errors with respect to the closest point on path to vehicle
    def path_errors(self, q):

        # Array declaration for closest_point_index function
        N_pp = len(self.path)
        x_dist = np.zeros(N_pp)
        y_dist = np.zeros(N_pp)
        self.dist2points = np.zeros(N_pp, dtype=float)           #why an object instance

        # Go through path points from previous closest point and find the new closest point to vehicle
        #... multiply by 0.01 to convert to metres
        for j in range(0, N_pp):
            x_dist[j] = np.array((q[0] - (.01*self.path[j][0])))
            y_dist[j] = np.array((q[1] - (.01*self.path[j][1])))
            self.dist2points[j] = np.array(np.sqrt(x_dist[j] ** 2 + y_dist[j] ** 2))

        # misan: above, rather than nonzero which makes the array lose its order(?),
        # try using a mask to set the zero value(s) or threshold value to 'inf',
        # then use: self.c = np.argmin(self.dist2points)

        mask = self.dist2points == 0                  # or '= self.dist2points < threshold'
        self.dist2points[mask] = np.inf                # or a very large number
        self.c = np.int(np.argmin(self.dist2points))

        # Local variable to represent closest point at current instance
        targetpoint = self.path[self.c]

        self.true_path.append(targetpoint)

        # Heading Error Calculations
        targetpoint_heading = targetpoint[2]  # Heading of the closest point at index c in radians

        # Heading error in radians
        eh = q[2] - targetpoint_heading

        # Lateral Error Calculation - considering x and y in N-E-S-W global frame to calculate lateral error
        del_x = q[0] - (0.01*targetpoint[0])
        del_y = q[1] - (0.01*targetpoint[1])
        self.err.append(np.sqrt(del_x ** 2 + del_y ** 2))
        self.errx.append(del_x)
        self.erry.append(del_y)
        el = -del_x * np.sin(targetpoint[2]) + del_y * np.cos(targetpoint[2])

        # Summary errors into an array for cleanliness
        tracking_errors = np.array([el, eh])

        return tracking_errors

    # -------------------------- Path Following Controller Functions --------------------------------
    # Function to calculate steering rate using Feedback Linearization Controller
    def fbl_controller(self, q, e):

        # Linearized control input using [z1, z2] where e = [el, eh]
        self.eta = self.kp * e[0] + self.kd * (self.v * np.sin(e[1]))

        # Calculate steering rate based on vehicle errors
        omega = self.eta / (self.v * np.cos(e[1]))

        # Correct / Saturate steering rate if needed
        omega_corr = self.omega_sat(omega)

        return omega_corr

    # -------------------------- Sensor Data Handlers --------------------------------

    # Handler #2 - For topic /odometry/filetered_map to get pose and velocities
    # Function to handle position estimate of the Husky from robot_localization
    def get_pose(self, data):

        # Define data from handler as estimate = odometry estimate
        self.estimate = data

        # Redefine new variable which is only pose component of Odom msg
        self.est_pose = copy.deepcopy(self.estimate.pose)

        # Redefine new variable which is absolute linear velocity from Odom msg
        self.vel = np.sqrt((self.estimate.twist.twist.linear.x) ** 2
                           + (self.estimate.twist.twist.linear.y) ** 2
                           + (self.estimate.twist.twist.linear.z) ** 2)


if __name__ == '__main__':
    rospy.init_node('find_and_follow_paths')
    Find_Path_and_Follow()
    rospy.spin()