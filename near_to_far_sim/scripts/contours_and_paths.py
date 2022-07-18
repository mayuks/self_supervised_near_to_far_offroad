#!/usr/bin/env python
from __future__ import print_function
import cv2
import imutils
import numpy as np
import math as mt
import tf
import copy
import rospy
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import Pose, PointStamped, PoseStamped, Twist
from std_msgs.msg import Empty, Bool, Int8, Int16, Float32MultiArray
from sensor_msgs.msg import Joy
from near_to_far_sim.msg import IMU_trigger, Num
import matplotlib.pyplot as plt
import time
from PIL import Image


class Contours_and_Paths(object):
    def __init__(self):
        self.im = None
        self.unknwon_cluster_color_features = None
        self.known_cluster_labels_and_vib = None
        self.labels_with_unknown_IMU = None
        self.robot_pose = None
        self.true_path = []
        self.actual_path_pts = []
        self.err =[]

        self.errx = []
        self.erry = []

        self.robot_start_x = 0
        self.robot_start_y = 0
        self.yaw = 0

        self.scale = 1            #xx...resizing factor...if pix was resized to world size/xx

        # **** Initialize variables from handlers ****
        self.prev_time = None
        self.paths = None
        self.path = None
        self.est_pose = None
        self.vel = None
        self.vib_data = None
        self.c = 0
        self.flag = 0
        self.count = 0

        # Initalize options

        # Define button that will be used as a deadman switch for path following
        self.estop_button = 1       # Set Deadman to button [1] which is the 'A' on the Logitech joysticks
        self.use_estop = rospy.get_param('~use_estop', True)
        self.estop = False

        # **** Get vehicle and controller parameters ****
        # Saturating steering rate to be no more than 2 rad/s (115 deg/sec)
        self.sat_omega_lim = rospy.get_param('~sat_omega_lim', 2)  # misan: find this parameter in launch file

        # Defining velocities for the vehicle (Desired and minimum)
        self.v_des = rospy.get_param('~const_vel', 0.50)  # [m/s] Desired vehicle speed
        self.v_min = rospy.get_param('~v_min',
                                     0.25)  # [m/s] Vehicle speed used in the control law if the vehicle is moving slower than this

        # Define parameter for Path Travelled
        self.path_traveled = Path()
        self.poi = PointStamped()

        # Setup to chose the correct controller based on the launch file
        self.controller_selector = rospy.get_param('~controller_selector', 'fbl')
        if self.controller_selector == 'fbl':
            # Call init function to set up fbl gains
            self.fbl_init()

        else:
            print('Select proper controller')


        self.T = rospy.get_param('~rate', 0.1)
        self.timer = rospy.Timer(rospy.Duration(self.T), self.timer_callback)

        # Topics that controller is subscribing to
        rospy.Subscriber('chatter', Num, self.callback, queue_size=1)

        # # To get estimate pose and velocity of vehicle from Gazebo_simuator EKF_Localization package
        rospy.Subscriber('odometry/filtered', Odometry, self.handle_estimate, queue_size=1)


        rospy.Subscriber('cluster_acclns', Num, self.get_vib_data, queue_size=1)           #subscribe to IMU vib data from record_IMU_data.py

        self.sub_joy = rospy.Subscriber('husky_b1/joy', Joy, self.handle_joy, queue_size=1)                         # To joystick information for deadman pendant


        # Topics to that controller is publishing to
        self.pub_driveable = rospy.Publisher('driveable', Int8, queue_size=1, latch=True)           #latch? Number of driveable-sized cluters
        # self.pub_paths = rospy.Publisher('paths_list', Num, queue_size=1, latch=True)           #latch?
        self.pub_path_points_followed = rospy.Publisher('path_points_followed', Int16, queue_size=1, latch=True)  # latch?
        self.pub_estop = rospy.Publisher('husky_b1/e_stop', Bool, queue_size=1)                      # Publish to estop topic to put husky in soft estop mode
        self.pub_twist = rospy.Publisher('husky_velocity_controller/cmd_vel', Twist, queue_size=100)                  # Publish to command velocities (v, omega) to the vel controller
        self.pub_IMU_record = rospy.Publisher('get_IMU_data', IMU_trigger, queue_size=1, latch=True)
        self.pub_features_store = rospy.Publisher('features_store', Num, queue_size=1, latch=True)
        self.pub_empty = rospy.Publisher('starter', Empty, queue_size=1)                            # Publish to dbscan_and_contours(.py) node for next run
        self.pub_actual_points = rospy.Publisher('actual_robot_path_pts', Num, queue_size=1, latch=True)
        self.lat_head_error = rospy.Publisher('lateral_heading_error', Num, queue_size=1, latch=True)
        self.pub_errx = rospy.Publisher('errx', Num, queue_size=1, latch=True)
        self.pub_erry = rospy.Publisher('erry', Num, queue_size=1, latch=True)
        self.pub_planned_paths_1 = rospy.Publisher('planned_paths_1', Num, queue_size=1, latch=True)



    def callback(self, data):
        self.im = copy.deepcopy(data.labelled)
        self.unknown_cluster_color_features = copy.deepcopy(data.no_IMU_cluster_attributes)
        self.known_cluster_labels_and_vib = copy.deepcopy(data.cluster_attributes)
        self.labels_with_unknown_IMU = copy.deepcopy(data.no_IMU_cluster_attributes_label)


    def get_vib_data(self, data):
        self.vib_data = data.cluster_attributes

    # For 'deadman'button...emergency stop
    def handle_joy(self, data):
        # Conditional statement to use estop or not...robot will only respond to ontrol signals when button is pressed i.e when 'True'
        if self.use_estop is True:
            # Read estop buttons and if they are on (1), then estop is off
            if data.buttons[self.estop_button] == 1:
                self.estop= False
                # self.pub_plotter.publish()    # This should start the plotter by publishing a to the starter topic
            else:
                self.estop = True
                rospy.logwarn_throttle(2.5,"Press A button to engage deadman")
            # Publish soft estop param
            self.pub_estop.publish(self.estop)
        else:
            self.estop = False                      #necessary?

        # #reverse of the above
        # if self.use_estop is True:
        #     # Read estop buttons and if they are on (1), then estop is off
        #     if data.buttons[self.estop_button] == 1:
        #         self.estop= True
        #         # self.pub_plotter.publish()    # This should start the plotter by publishing a to the starter topic
        #     else:
        #         self.estop = False
        #         rospy.logwarn_throttle(2.5,"Press A button to stop robot")
        #     # Publish soft estop param
        #     self.pub_estop.publish(self.estop)
        # else:
        #     self.estop = False                      #necessary?


    def timer_callback(self, event):

        if self.im is None:
            rospy.logwarn_throttle(2.5, "Controller waiting for labelled image...")
            return

        if self.est_pose is None:             # is None....no position data from Vicon or EKF...ceck appropriate topic
            rospy.logwarn_throttle(2.5, "... waiting for pose...")
            return

        # im = cv2.imread('/home/offroad/ld.png', 0)          #gray
        im = copy.deepcopy(self.im)


        im = np.asarray(im)
        im = im.astype(np.uint8)
        # print(im)


        im = np.reshape(im, (np.int(350/self.scale), np.int(200/self.scale)))   # gray     i.e maxheight by maxwidth
        # np.savetxt("all.csv", im, delimiter=",")

        # new_im = Image.fromarray(im)  # https://www.kite.com/python/examples/4887/PIL-convert-between-a-pil-%60image%60-and-a-numpy-%60array%60
        # new_im.save("ld.png")

        print(im.shape)

        Area_centrepts = []
        boxpts = []
        drivable_sized_clusters = []
        coords_for_IMU = []

        if len(self.labels_with_unknown_IMU) == 0:       #all clusters unknwon...list is empty
            labels_to_use = np.arange(1, np.amax(im)+1)
        else:
            labels_to_use = self.labels_with_unknown_IMU

        for n in labels_to_use:
            image = (im==n).astype(np.uint8)
            # np.savetxt("wood.csv", image, delimiter=",")

            cnt = cv2.findContours(image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            cnt = imutils.grab_contours(cnt)

            cnt = sorted(cnt, key = cv2.contourArea, reverse =True)[0]

            cnt=np.asarray(cnt, dtype=np.float32)


            cnt=np.asarray(cnt, dtype=np.int)

            rect = cv2.minAreaRect(cnt)     #(OPENCV image coords i.e. (col, row))
            print ("rect 1 is")
            print(rect)
            # print(len(rect))
            # print(type(rect))
            box = cv2.boxPoints(rect)
            box = np.int0(box)          #lists box edges for bottom left clockwise...pls recheck
            print(box)


            # handle the base terrain, it's contour is the full image; or any similar large clusters
            # if rect[1][0]>0.6*image.shape[1] and rect[1][1]>0.6*image.shape[0]:       #the indexing reflects opencv (column, row: x,y) vs python (conventional (row,column): y,x) shape definitions

            # Cover both cases to cover long strips of patches ie:
            # Large length narrow breadth or large breadth narrow length (See Gordon1.jpg result using check_contours_use.py in myslic.py, the sandy patch/cluster)
            if (rect[1][0]>0.3*image.shape[1] and rect[1][1]>0.1*image.shape[0]) or (rect[1][0]>0.3*image.shape[1] and rect[1][1]>0.1*image.shape[0]):
            # if rect[1][0] > 0.3 * image.shape[1] and rect[1][1] > 0.3 * image.shape[0]:
                # image = image==1
                # np.savetxt("check_imaage.csv", image, delimiter=",")
                # test_coords = [[50,400], [100,400], [150,400], [200,400], [50,300], [100,300], [150,300], [200,300],
                #                [50,200], [100,200], [150,200], [200,200], [50,100], [100,100], [150,100], [200,100],
                #                [50,50], [100,50], [150,50], [200,50]]      # an evenly-spaced list of test XY coordinates centres;
                                         # start close to robot then move horizontally and verically...not in image space(?)
                #                         # doesn't seem to matter if it exceeds image dimensions but still scale as below:

                row = np.arange(1, 0, -0.05)          #i.e. start populating list with centres close to the robot....cos maybe argmax will return the first instance
                col = np.arange(0, 1, 0.05)
                test_coords = []
                for i in row:       #col?
                    for j in col:  #row?
                        test_coords.append([i * image.shape[0], j * image.shape[1]])    #(np is in (row,col))

                test_coords = [[np.int(x) for x in y] for y in test_coords]             # make all integers
                # print(test_coords)
                # print(len(test_coords))
                max_test = np.zeros((len(test_coords), 1))
                # print(max_test.shape)
                half_box_size = np.int(46/self.scale)       #100cm is the size of terrain elements in Gazebo
                for i in range (0,len(test_coords)):
                    #functionally get the xys based on known centre, appropriate patch size based on size of robot and image scaling...
                    # box size of 46 used here i.e. approx. half same size as faux terrain boxes....

                    #TO DO: define more centres, i.e. closer spacing between centres, and make centres always fit image size (byparameterising?)
                    #cos it appears to ignore any centre that doesn't fit currently...will not warn you

                    # print(image.shape)
                    # print((image[(test_coords[i][1] - half_box_size):(test_coords[i][1] + half_box_size), (test_coords[i][0] - half_box_size):(test_coords[i][0] + half_box_size)]).shape)
                    max_test[i] = np.sum(image.copy()[test_coords[i][0]-half_box_size:test_coords[i][0]+half_box_size, test_coords[i][1]-half_box_size:test_coords[i][1]+half_box_size])
                    #numpy slicing so indices reversed
                    #instead of checking all centres, make it stop/break when the ratio of the np.sum/test_area is say 95, if not then continue
                print(len(max_test))
                # print(max_test)

                # print('a test test_coords is')
                # y = 500
                # print(image.copy()[test_coords[y][1] - half_box_size:test_coords[y][1] + half_box_size,
                #    test_coords[y][0] - half_box_size:test_coords[y][0] + half_box_size])

                x = np.argmax(max_test)
                # print(test_coords[x])
                # print(x)
                # print(max_test[x])

                box = np.array([[test_coords[x][1] - half_box_size, test_coords[x][0] + half_box_size],  # Bottom Left(BL); wrt test_centre: note image axis origin is TL and [col,row] to stay compatible with OPEN CV's rect and box functins that don't go through the 'if condition of this section
                                [test_coords[x][1] - half_box_size, test_coords[x][0] - half_box_size],  # Top Left(TL)
                                [test_coords[x][1] + half_box_size, test_coords[x][0] - half_box_size],  # Top Right(TR)
                                [test_coords[x][1] + half_box_size, test_coords[x][0] + half_box_size]])  # Bottom Right(BR)

                # rect = ((test_coords[x]), (box[2][1]-box[0][1], box[0][0]-box[2][0]), 0)
                # # i.e ((centreX, centreY), (W, H), angle))

                rect = (([test_coords[x][1], test_coords[x][0]]), (box[2][0] - box[0][0], box[0][1] - box[2][1]), 0)
                # i.e ((centreX, centreY), (W, H), angle))

            # TO DO!....DONE?
            Area_centrepts.append(rect)         # put a line above this that only appends if size threshold is met
            boxpts.append(box)                  # but maintain position on array to keep track of class_numbers
                                                # something like append 0 if size threshold not met...
            # print(len(Area_centrepts))
            # print(len(boxpts))
            #
            print("rect 2 is")
            print(rect)

            if rect[1][0]>=46/self.scale and rect[1][1]>=46/self.scale:
                #make a decison of size cut-off (to define noisy clusters), and change 46 to appropriate number based on robot size (overall width with some allowance)
                # TO DO: also check that the number of the class in the box meets a threshold, to weed out empty boxes

                box_area = rect[1][0]*rect[1][1]
                contour_area = cv2.contourArea(cnt)
                # print("cnt area is {}".format(contour_area))
                # print("cnt_box area is {}".format(box_area))
                print(contour_area/box_area)
                if (contour_area/box_area)>=0.5:   #solidity check...0.6 is my current choice, change reasonably
                    drivable_sized_clusters.append(1)
                else:
                    print('yes but no')
                    drivable_sized_clusters.append(0)

            else:
                print('no')
                drivable_sized_clusters.append(0)

            # if rect[1][0]>=23 and rect[1][1]>=23:  # rect[1][0]>=40 and rect[1][1]>=40:
            #     # make a decison of size cut-off, and change 40 to appropriate number based on robot size (overall width with some allowance)
            #     drivable_sized_clusters.append(1)
            # else:
            #     drivable_sized_clusters.append(0)

            # if rect[1][0]>=40/self.scale and rect[1][1]>=40/self.scale:
            #     #make a decison of size cut-off, and change 40 to appropriate number based on robot size (overall width with some allowance)
            #     # TO DO: also check that the number of the class in the box meets a threshold, to weed out empty boxes
            #     drivable_sized_clusters.append(1)
            #
            # else:
            #     drivable_sized_clusters.append(0)


            x_y = [] # centres of relevant edges...plus  box centre

            if rect[1][1] < rect[1][0]:                     #rect[1][1] == rect[1][0] or rect[1][1] < rect[1][0]:     # rect is in OPEN CV coords (col, row) so means if height is > width
                x = np.int (0.5 * (box[0][0] + box[1][0]))  # centre point of left edge on  X axis ... always start from the left edge i.e. BL, TL?
                y = np.int(0.5 * (box[0][1] + box[1][1]))

                x_y.append([x,y])

                x = np.int(rect[0][0])                      # midpoint of box
                y = np.int(rect[0][1])

                x_y.append([x,y])

                x = np.int(0.5 * (box[2][0] + box[3][0]))  # centre point of right edge on  X axis
                y = np.int(0.5 * (box[2][1] + box[3][1]))

                x_y.append([x,y])

                x_y_complete = []

                for i in np.arange(np.int(rect[1][0]), -1, -1):             # 'arange' width to zero in steps of 1 pixel, should be faster than letteing AStar get points between thetwo edges
                    x = x_y[2][0] - ((i*1.0/rect[1][0]) * (x_y[2][0] - x_y[0][0]))  # fit points in consistent steps of 1 pixel, using similar triangles,...
                    y = x_y[2][1] - ((i*1.0/rect[1][0]) * (x_y[2][1] - x_y[0][1]))  # i.e divide the hypotenuse(the width)in small steps and get corresponding x,y coordinates

                    x = np.int(np.around(x))
                    y = np.int(np.around(y))

                    if x<0 or y<0 or x>=image.shape[1] or y>=image.shape[0]:      # keep only (realistic) points within image... else path planner goes in death cycle....forever
                        print("don't")
                        continue            # remember x in image is width and y is height hence the shape correlations above wrt to numpy standard(the reverse
                                            # recall numpy arrays are [row,col], but cv outputs e.g fromcv2.box are [col,row]
                    x_y_complete.append([y,x])       # reversed this to be consistent with the Astar output...note how the cv2 parts are reversed when drawing and displaying
                                                    # i.e x_y_complete[i][1], x_y_complete[i][0])... and not x_y_complete[i][0], x_y_complete[i][1])...: this is the cause

                pose = abs(rect[2])

                x_y.append(90 + pose)

                x_y_complete.append(90 + pose)


            else:               # i.e if height <= width

                x = np.int(0.5*(box[0][0] + box[3][0]))         # centre point of bottom edge on  X axis ... always start from the bottom edge i.e. BL, BR?
                y = np.int(0.5 * (box[0][1] + box[3][1]))

                x_y.append([x,y])

                x = np.int(rect[0][0])
                y = np.int(rect[0][1])

                x_y.append([x,y])

                x = np.int(0.5*(box[1][0] + box[2][0]))          # centre point of top edge on  X axis
                y = np.int(0.5 * (box[1][1] + box[2][1]))

                x_y.append([x,y])

                x_y_complete = []

                for i in np.arange(np.int(rect[1][1]), -1, -1):
                    x = x_y[2][0] - ((i*1.0/rect[1][1]) * (x_y[2][0] - x_y[0][0]))
                    y = x_y[2][1] - ((i*1.0/rect[1][1]) * (x_y[2][1] - x_y[0][1]))

                    x = np.int(np.around(x))
                    y = np.int(np.around(y))

                    if x<0 or y<0 or x>=image.shape[1] or y>=image.shape[0]:
                        print("don't")
                        continue

                    x_y_complete.append([y,x])

                pose = abs(rect[2])

                x_y.append(pose)

                x_y_complete.append(pose)

            coords_for_IMU.append(x_y_complete)

        # print(boxpts)
        print(coords_for_IMU)

        # publish number of driveable clusters in a msg to record_IMU_data script

        driveable = (np.array(drivable_sized_clusters)==1).sum()
        print(driveable)
        print(drivable_sized_clusters)
        self.pub_driveable.publish(driveable)                #now publish it

######################################################################################################################

        #Path Planner Starts

#######################################################################################################################

        # Next pass start and end for each section to Astar and concatenate...
        # i.e Astar[0], coors_for_IMU[0], Astar[], coords_for_IMU[1]... etc

        # Current_Robot_Position = [312,120]                  # parameterise wrt image size
        # Final_Position = [0,120]                            # parameterise...

        Current_Robot_Position = [im.shape[0]-1, np.int(0.5*(im.shape[1]-1)+1)]                  # parameterise wrt image size
        Final_Position = [0, np.int(0.5*(im.shape[1]-1)+1)]                                 #numpy style....(row,col)

        # Current_Robot_Position = [im.shape[0] - 1, np.int(0.4 * (im.shape[1] - 1) + 1)]
        # Final_Position = [0, np.int(0.4*(im.shape[1]-1)+1)]

        start_end_coords = []

        start_end_coords.append(Current_Robot_Position)

        for drivable_sized_cluster, coord_for_IMU in zip(drivable_sized_clusters, coords_for_IMU):
            if drivable_sized_cluster==1:
                start_end_coords.append(coord_for_IMU[0])
                start_end_coords.append(coord_for_IMU[-2])              # -1 is pose/angle
            else:
                continue

        start_end_coords.append(Final_Position)

        print(start_end_coords)
        # #######################################################################################################################
        # # Astar starts...
        # #######################################################################################################################
        OptimalPaths = []

        for i in range(0, len(start_end_coords)):

            if 2*i+1 > len(start_end_coords) - 1:       # list limit: see Goal formula below
                continue                                    #check

            # Preallocation of Matrices

            Connecting_Distance = 1  # 8 possible directions....higher won't work unless you pre-declare obstacles
            MAP = im  # ld super_cluster image
            Start = start_end_coords[2*i]
            Goal = start_end_coords[2*i+1]  # intersection between heading and the zero row lne (i=0)
            # print(Goal)
            Goal =[Goal[0], Goal[1]]
            GoalRegister = np.zeros_like(MAP, dtype=np.int8)
            GoalRegister[Goal[0], Goal[1]] = 1
            StartX = Start[1]  # review if it should be StartY, STartX...from GPS or VIcon (at time Image was taken?)
            StartY = Start[0]

            # [Height,Width] = MAP.size              # Height and width of matrix
            GScore = np.full_like(MAP, 0, dtype=np.float32)  # Matrix keeping track of G-scores...iniialise with 1e2 so that any missedpixel is not zero
            if len(self.known_cluster_labels_and_vib) > 0:
                GScore = np.full_like(MAP, np.inf, dtype=np.float32)  # Matrix keeping track of G-scores...iniialise with 1e2 so that any missedpixel is not zero
            FScore = np.full_like(MAP, np.inf, dtype=np.float32)  # Matrix keeping track of F-scores [only open list]
            Hn = np.zeros_like(MAP, dtype=np.float32)  # Heuristic matrix
            OpenMAT = np.zeros_like(MAP, dtype=np.int8)  # Matrix keeping of open grid cells
            ClosedMAT = np.zeros_like(MAP, dtype=np.int8)  # Matrix keeping track of closed grid cells
            ParentX = np.zeros_like(MAP, dtype=np.int16)  # Matrix keeping track of X position of parent
            ParentY = np.zeros_like(MAP, dtype=np.int16)  # Matrix keeping track of Y position of parent

            # for i in range(np.amin(MAP), np.amax(MAP)+1):
            #     GScore[MAP == i] = 1e3*1e1          # ...= IMU_Cost[i]  i.e. 1e3 a placeholder...
            #     Link this to the IMU based costs for each super custer
            # the costs must have been pre-calculated in an array in the same order as the MAP
            # so just index the values...the costs must be appropriately designed to differ by orders of magnitude

            # GScore[MAP == 1] = 1e2
            # GScore[MAP == 2] = 1e2
            # GScore[MAP == 3] = 1e2
            # GScore[MAP == 4] = 1e5
            # GScore[MAP == 5] = 1e6
            # GScore[MAP == 6] = 1e7

            if len(self.known_cluster_labels_and_vib) > 0:
                self.known_cluster_labels_and_vib = np.reshape(self.known_cluster_labels_and_vib, (-1, 2))
                for i in range(0, len(self.known_cluster_labels_and_vib)):
                    GScore[MAP == self.known_cluster_labels_and_vib[i][0]] = np.int(self.known_cluster_labels_and_vib[i][1])  # TO DO: evaluate GSCORE costs based on IMU metric

                Test = GScore == np.inf
                GScore[Test] = np.amax(GScore)

            ### Setting up matrices representing neighboors to be investigated

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
            # report that it is taken from THAT book

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
                        tentative_gScore = GScore[CurrentY, CurrentX] + GScore[CurrentY + i, CurrentX + j] + cell_size * mt.sqrt(i ** 2 + j ** 2) # new...plan to use this for thesis
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

            OptimalPaths.append(OptimalPath)


        coordss_for_IMU_no_pose = []

        coordssss_for_IMU = copy.deepcopy(coords_for_IMU)

        for drivable_sized_cluster, coordsss_for_IMU in zip(drivable_sized_clusters, coordssss_for_IMU):
            if drivable_sized_cluster==1:
                del coordsss_for_IMU[-1]
                coordss_for_IMU_no_pose.append(coordsss_for_IMU)
            else:
                continue


        # image1 = cv2.imread('/home/offroad/Desktop/get-perspective-transform-example/transformed_resized.png')               #colour image
        # if self.count == 0:
        #     image1 = cv2.imread('/home/offroad/Desktop/get-perspective-transform-example/transformed_resized_two_forward.png')
        #
        # if self.count == 1:
        #     image1 = cv2.imread('/home/offroad/Desktop/get-perspective-transform-example/transformed_resized_three_forward.png')

        image1 = cv2.imread("/home/misan/figs/transformed_resized_in_use_sim.tiff")
        #
        # for x in coordss_for_IMU_no_pose:
        #     # x = x[::-1]
        #     for j in range(0, len(x)-1):
        #         cv2.line(image1, (x[j][1], x[j][0]), (x[j+1][1], x[j+1][0]), (0, 255, 0), 2)
        #
        # cv2.imshow("outline", image1)
        # cv2.waitKey(3000)
        # cv2.destroyAllWindows()


        # Add IMU_Paths to AStar Paths in the right order

        OptimalPathss = []
        # print(len(OptimalPaths))

        OptimalPathss.append(OptimalPaths[0][::-1])
        # first path from start to end....Note: OptimalPaths comes end to start, so flipped [;;-1]
        # print(len(OptimalPaths))
        del OptimalPaths[0]         #or just use .pop above?

        Final_Path = OptimalPaths.pop(-1)

        OptimalPathss.append(coordss_for_IMU_no_pose[0])               # first path for IMUto drive...IMU_path

        del coordss_for_IMU_no_pose[0]

        # print(len(OptimalPaths))
        # print(len(coordss_for_IMU_no_pose))


        for coords_for_IMU_no_pose, OptimalPathsss in zip(coordss_for_IMU_no_pose, OptimalPaths):

            OptimalPathss.append(OptimalPathsss[::-1])
            OptimalPathss.append(coords_for_IMU_no_pose)


        OptimalPathss.append(Final_Path[::-1])


        # image1 = cv2.imread('/home/offroad/Desktop/get-perspective-transform-example/transformed_resized.png')               #colour image

        # image1 = cv2.imread('/home/offroad/Desktop/get-perspective-transform-example/transformed_resized_two_backward.png')
        # print(image1.shape)

        for drivable_sized_cluster, box in zip(drivable_sized_clusters, boxpts):
            if drivable_sized_cluster == 1:
                cv2.drawContours(image1, [box], -1, (0, 255, 0), 4)
                cv2.imshow("outline", image1)
                cv2.waitKey(2000)
            else:
                continue

        for drivable_sized_cluster, coord_for_IMU in zip(drivable_sized_clusters, coords_for_IMU):
            if drivable_sized_cluster == 1:
                for j in range(0, len(coord_for_IMU)-2):                 #-1 is pose
                    #the Paths are in numpy format (row,col), so we flip to plot in CV2 format(col,row)
                    cv2.line(image1, (coord_for_IMU [j][1], coord_for_IMU [j][0]), (coord_for_IMU [j+1][1], coord_for_IMU [j+1][0]), (255, 0, 0), 2)
                else:
                    continue

        for Paths in OptimalPathss:
            # OptimalPaths = OptimalPath[::-1]  # make it start to end...was returned as end to start
            for i in range(0, len(Paths)):
                if i == len(Paths) - 1:
                    continue
                cv2.line(image1, (Paths[i][1], Paths[i][0]), (Paths[i + 1][1], Paths[i + 1][0]),
                         (0, 0, 255), 2)
                cv2.imshow("outline", image1)
                cv2.waitKey(5)
        print('stop')
        cv2.imshow("outline", image1)
        cv2.waitKey(10)
        # cv2.waitKey(0)
        cv2.destroyAllWindows()

        # cv2.imwrite("figs/image_traversal_paths_no_costs.tiff",image1)
        filename = 'figs/' + str(time.clock())+'.png'
        cv2.imwrite(filename, image1)                #for the record

        image_to_global = [self.to_global(x) for x in OptimalPathss]
        # print(len(image_to_global))
        # print(image_to_global[-2][0])

        paths_to_follow = [self.add_heading(x) for x in image_to_global]

        # print(len(paths_to_follow))
        # print(paths_to_follow[-2][0])

        # Now add initial path from vehice to start of image

        loc = self.est_pose
        self.robot_pose = self.state(loc)

        print('pose is')
        print(self.robot_pose)

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

        # print(planned)

        pub_planned  = Num()
        pub_planned.cluster_attributes = planned
        self.pub_planned_paths_1.publish(pub_planned)

    ########################################################################################################################

        #just some testing and printing stuff

    ########################################################################################################################

        # print(final_paths_to_follow[0])
        # print(len(final_paths_to_follow[0]))
        # print('done')
        # print(final_paths_to_follow[1])
        # print(len(final_paths_to_follow[1]))
        # print('done')
        # print(final_paths_to_follow[2])
        # print(len(final_paths_to_follow[2]))
        # print('done')
        # print(final_paths_to_follow[3])
        # print(len(final_paths_to_follow[3]))
        # print('done')
        # print(final_paths_to_follow[4])
        # print(len(final_paths_to_follow[4]))
        # print('done')
        # print(final_paths_to_follow[5])
        # print(len(final_paths_to_follow[5]))
        # print('done')
        # print(final_paths_to_follow[6])
        # print(len(final_paths_to_follow[6]))
        # print('done')
        # print(final_paths_to_follow[-1])
        # print(len(final_paths_to_follow[-1]))

        plt.style.use("ggplot")
        plt.figure()

        for n in range(0, len(final_paths_to_follow)):
            xs = [i[0] for i in final_paths_to_follow[n]]
            ys =[i[1] for i in final_paths_to_follow[n]]

            if n == len(final_paths_to_follow) - 1:  # only label last plot...to avoid multiple labels?
                plt.plot(xs, ys, '-o', color="black", markeredgecolor="black", label="planned path")
            else:
                plt.plot(xs, ys, '-o', color="black", markeredgecolor="black")

        # plt.title("Final map paths")
        fontsize = 26
        plt.legend()
        plt.xlabel("x (cm)", fontsize=fontsize)
        plt.ylabel("y (cm)", fontsize=fontsize)
        # ax.set_title(title, fontsize=fontsize)
        plt.tick_params(labelsize=fontsize )
        plt.tight_layout()
        plt.savefig("/home/misan/figs/{}.tiff".format(time.clock()))
        # plt.show()

        #
        # #check poses
        # plt.style.use("ggplot")
        # plt.figure()
        #
        # thethas = [i[2]*180/np.pi for i in final_paths_to_follow[3]]
        #
        # print("poses are")
        # print(thethas)
        #
        # plt.plot(thethas, '-o', color="black", markeredgecolor="black", label="planned path")
        #
        # # plt.title("Final map paths")
        # fontsize = 26
        # plt.legend()
        # plt.xlabel("point", fontsize=fontsize)
        # plt.ylabel("pose (rad)", fontsize=fontsize)
        # # ax.set_title(title, fontsize=fontsize)
        # plt.tick_params(labelsize=fontsize )
        # plt.tight_layout()
        # plt.savefig('figs/poses2.png')
    # #####################################################################################################################################
    #     # Save start pose
    # #####################################################################################################################################
    #
    # #     loc = self.est_pose
    # #     pos = self.state(loc)
    # #     # start_pose = pos[2]
    # #
    # #     start_pose = pos[2]+0.09
    # #
    # #     print('ok, first path point')
    # #     print(final_paths_to_follow[0][0])
    # #     print(final_paths_to_follow[0][1])
    # #     print(final_paths_to_follow[0][2])
    # #     print(final_paths_to_follow[0][3])
    # #     print(final_paths_to_follow[0][4])
    # #     print('ok, robot position')
    # #     print(pos)
    # #
    # #     # try a loop to pop out the first (and maybe second path-careful robot can't jump, closest path point must be 'close enough')
    # #     # point if it is negative just to check why map_EKF gives issues sometimes
    # #     # to check if the closest path point is actually behind the robot -- because at the instant when it wants to start path
    # #     # folling the map_ekf position might have drifted a bit putting the robot in front of the closest path ptor out of reach due to holonomic
    # #     # constraints i.e robot cannot jump!
    # # ######################################################################################################################################
    # #controller loop
    # ######################################################################################################################################
        final_pathz_to_follow = copy.deepcopy(final_paths_to_follow)
        # ... for calculating steering commands
        for i in range(0, len(final_pathz_to_follow)):
            print('newloop')
            # final_path_to_follow = final_paths_to_follow[i]
            # self.path = final_path_to_follow

            self.path = final_pathz_to_follow[i]

        #             print(i)
        #             print(len(final_pathz_to_follow)-1)
        #             print(len(self.path))

            # rotate Husky to start pose of path: define variables
            loc = self.est_pose
            pos = self.state(loc)
            robot = pos[2]
            target_pos = self.path[0][2]
            twist = Twist()
            twist.linear.x = 0.0
            yaw_rate = 0.7

            # rotate Husky to start pose of path: determine (shorter) direction of rotation to start pose i.e cw or ccw
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

            # rotate Husky to start pose of path: rotate to start pose in predetermined shorter direction
            while True:
                loc = self.est_pose
                q = self.state(loc)
              #Husky's diff drive controller responds to this value (from using joystick and echoing Husky/cmd_vel topic)...try 0.8
                self.pub_twist.publish(twist)

                # if (q[2]/self.path[0][2] >= 0.85):              # try 0.9 or 0.95... and use subtraction instead to avoid division by zero
                if abs(q[2] - self.path[0][2]) < (3.142/36):    # abs(q[2] - target_pos) i.e if the difference is within 5 deg either way...
                    twist.angular.z = 0.0                       # tomake sure 5 deg is not too small try smaller (than 1) twist.z values for smaller rotation/step
                    self.pub_twist.publish(twist)
                    break
            time.sleep(5)

            if i > 0 and not bool(i % 2):
                # path: zero is robot position to image start; 1 is image start to first cluster;
                # 2,4,6... are 1st,2nd... IMU path
                # publish msg to IMU record to start recording data (or keep recording data) in its timer_callback
                IMU_things = IMU_trigger()
                IMU_things.cluster_number = i
                IMU_things.start_stop = 'start'
                self.pub_IMU_record.publish(IMU_things)
            else:
                # publish msg to IMU record to stop recording data (or stay idle) i.e. in its timer_callback
                IMU_things = IMU_trigger()
                IMU_things.cluster_number = 0
                IMU_things.start_stop = 'stop'
                self.pub_IMU_record.publish(IMU_things)


            while True:

                # ------------- Start polling data ------------
                # Define sensor data at the beginning of the callback function to be used throughout callback function

                # Define velocity of vehicle
                self.v = self.vel
                # misan: self.v used in controller...so what's its relationship to self.v_des published to controller?
                # misan: my answer is that self.vel the absolute velocity in (3D) is used in controller calculations,
                # it is essentially the same as self.v_des most of the time because self.v_des is what we give Clearpath's controller
                # through self.pub_twist which publishes to Clearpath's supplied low level controller 'husky/cmd_vel'
                # for how invidual wheel speeds are calculated in Clearpath's supplied low level controller...
                # Husky-UGV-User-Manual.pdf: section 3.1....i.e knowing v=self.v and omega = self.omega solve the 2 eqns simultaneously to get
                # vr (right wheels speed) and vl(left wheels speed)? and the Clearpath controller will calculate and supply
                # the voltage required to achieve these speeds for each driven wheel...only the front wheels are powered by motors(?)

                # Singularity prevention.
                # If speed is low, omega is nearing infinity. In this case, we just make it self.v_min
                if self.v < self.v_min:
                    self.v = self.v_min

                # Define location of the vehicle within timer function
                loc = self.est_pose


                # Establishing the current state of the unicycle vehicle as (x, y, theta)
                q = self.state(loc)

                robot_actual = Num()
                robot_actual.cluster_attributes = q
                self.pub_actual_points.publish(robot_actual)

                self.actual_path_pts.append(q)

                # Calculate the lateral and heading error of the vehicle where e = [el, eh]
                e = self.current_path_errors_slow(q)

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

                        if IMU_things.start_stop == 'start':
                            IMU_things.start_stop = 'stop'
                            self.pub_IMU_record.publish(IMU_things)

                        v_des = 0.0
                        self.omega = 0.0

                        # Build twist message to /cmd_vel
                        twist = Twist()
                        twist.linear.x = v_des
                        twist.angular.z = self.omega
                        self.pub_twist.publish(twist)

                        rospy.logwarn_throttle(2.5, "At the end of the interim path! Pause!")

                        self.c = 0              # reset the count

                        break

                    else:

                        if IMU_things.start_stop == 'start':
                            IMU_things.start_stop = 'stop'
                            self.pub_IMU_record.publish(IMU_things)
                        # v_des = 0.0
                        # self.omega = 0.0
                        rospy.logwarn_throttle(2.5, "At the end of the final path! Stop!")

                        # Build twist message to /cmd_vel
                        twist = Twist()
                        twist.linear.x = 0.0
                        twist.angular.z = 0.0
                        self.pub_twist.publish(twist)

                        self.c = 0          # reset the count

                        print('sleeping for 1 sec')
                        time.sleep(5)

                        break  # check

            # ---------------- Publishing Space ---------------

                # Build twist message to /cmd_vel
                v_des=self.v_des
                twist = Twist()
                twist.linear.x = v_des
                twist.angular.z = self.omega
                self.pub_twist.publish(twist)


        # now, use driveable cluster array to associate the vib data to the correct colour features in cluster_attributes

        per_cluster_color_features = np.reshape(self.unknown_cluster_color_features, (-1, 3))       #i.e reshape to (number of clusters x 3 columns), each column holding L, a, b features
        per_cluster_color_features = per_cluster_color_features.tolist()             #make it a (nested) list so we can use list comprehension

        print(self.vib_data)

        self.vib_data = np.asarray(self.vib_data)

        print(type(self.vib_data))

        self.vib_data = self.vib_data.tolist()

        print(self.vib_data)
        print(type(self.vib_data))


        feature_store = []

        for drivable_sized_cluster, per_cluster_color_feature in zip(drivable_sized_clusters, per_cluster_color_features):

            if drivable_sized_cluster == 1:
                temp = self.vib_data.pop(0)         #or just append self.vib_data.pop[0] directly?
                feature_store.append([per_cluster_color_feature[0], per_cluster_color_feature[1], per_cluster_color_feature[2], temp])
            else:
                continue

        # then publish to self.features_store in dbscan_and_contours.py for future comparison in its next cycle

        self.features_store = np.array(feature_store)           #change to array, then ravel for publishing as 1D array
        feat = Num()
        feat.cluster_attributes = np.ravel(self.features_store)  # self.features_store is size (driveable, 4)
                                                                # column is fixed as four for each custer...i.e L,a,b, IMU
                                                                # so can always be resized as ...np.reshape(xx, (-1,4)) at the receiving end
        self.pub_features_store.publish(feat)

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
            q = self.state(loc)
            twist = Twist()
            twist.linear.x = 0.0
            twist.angular.z = 0.4
            self.pub_twist.publish(twist)

            # if (q[2]/self.path[0][2] >= 0.85):
            # if q[2] - final_pathz_to_follow[0][0][2] < 3.142:     # facing reverse of starting position or...
            if q[2] < -3.054 and q[2] > -3.124:       # ... just use a small range around -175 and -179 deg with a small twist
                             # command to avoid the transtion from -179.9999 to +179.999. (Husky stared at zero pose)...FOR INDOORS

            # if abs(q[2] - start_pose) < (3.142 / 36):       #Approx. 5 deg. difference both ways
                twist.angular.z = 0.0
                self.pub_twist.publish(twist)
                break
        time.sleep(5)

    ##########################################################################################
        ## TO DO: OUTDOORS here calculate heading to final goal then rotate to it
    ##########################################################################################
        #     loc = self.est_pose
        #     q = self.state(loc)

        # theta = np.arctan2(self.final_goal[1]-q[1], self.final_goal[0]-q[0])

        # CHECK: this is the angular difference, between current and goal position:
        # add it 'vectorially' (notice atan2 used) to the current angular position to get the final position angle at this instant
        # OR: Just rotate Husky by this angle in the correct direction (this seems simpler)

        # angle = q[2] + theta

        # So if you know your FIXED goal in GPScoordinates,how do you get the position of the goal in Euclidean (or map) coordinates
        # i.e. self.final_goal[1], self.final_goal[0] in the theta equation above?
        # The NavSatTransform node in ROS Localization package will give you the GPS--Map transformation...get it and use it!

        # while True:
        #     # Build twist message to /cmd_vel
        #
        #     loc = self.est_pose
        #     q = self.state(loc)
        #     twist = Twist()
        #     twist.linear.x = 0.0
        #     twist.angular.z = 0.3
        #     self.pub_twist.publish(twist)
        #
        #     if (angle < 0 and q[2] < 0.98 * angle and q[2] > 1.02 * angle) or (if angle > 0 and q[2] > 0.98 * angle and q[2] < 1.02 * angle):
        #     if abs(q[2] - angle) < (3.142 / 36):  # i.e if the difference is within 5 deg either way...
        #
        #     #trouble if theta is +/-180!!!
        #
        #         twist.angular.z = 0.0
        #         self.pub_twist.publish(twist)
        #         break

        # time.sleep(2)

        # self.pub_empty.publish()        #start up next cycle...dbscan...
        # time.sleep(2)  # just in case

        self.pub_empty.publish()            #re-trigger dbscan_and_contours.py node for next run
        time.sleep(2)                       #just in case

    ##############################################################################################
        # Plot paths
    #############################################################################################
        plt.style.use("ggplot")
        plt.figure()

        for n in range(0, len(final_paths_to_follow)):
            xs = [i[0] for i in final_paths_to_follow[n]]
            ys =[i[1] for i in final_paths_to_follow[n]]

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

        plt.plot(xs, ys, '-o', color="red", markeredgecolor="green", label="actual path")

        fontsize = 22
        plt.legend(fontsize = 22)
        plt.xlabel("x (cm)", fontsize=fontsize)
        plt.ylabel("y (cm)", fontsize=fontsize)
        # ax.set_title(title, fontsize=fontsize)
        plt.tick_params(labelsize=fontsize)
        plt.tight_layout()
        plt.savefig("/home/misan/figs/{}.tiff".format(time.clock()))
        # plt.show()

    # ############################################################################

        self.actual_path_pts =[]            # reinitialise...clear appended paths in list for next run
        self.count = self.count + 1
        self.im = None

    ######################################################################################################################################

    def to_global(self, Path_N):          # convert paths from image to global coordinates

        loc = self.est_pose
        self.robot_pose = self.state(loc)

        # # global robot pose message from Vicon or  Odom_filtered_map EKF_map (ROS Robot_Localization package) in cm
        X_G_R = self.robot_pose[0]*100
        Y_G_R = self.robot_pose[1]*100
        angle = self.robot_pose[2]

        global_coords = []

        for i in range(0, len(Path_N)):

            one_global_coord = []

            # U = Path_N[i][1]                # back to pixel coordinates
            # V = Path_N[i][0]

            U = self.scale*Path_N[i][1]                # back to pixel coordinates...if resized to half size
            V = self.scale*Path_N[i][0]

            image_to_robot = np.matmul(np.array([[1, 0, 450], [0, 1, 100], [0, 0, 1]]), np.array([[-V], [-U], [1]]))         #change ... from calibration

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

    def fbl_init(self):

        # Lookahead Parameters
        self.enable_lookahead = rospy.get_param('~enable_lookahead',
                                                False)  # [bool] Set true to get heading from a path point ahead of the vehicle
        self.path_point_spacing = rospy.get_param('~path_point_spacing', 0.05)  # [m] Spacing between path points
        self.t_lookahead = rospy.get_param('~t_lookahead',
                                           0.5)  # [s] Amount of time to look ahead of vehicle when lookahead is enabled

        # Controller Gain Parameters and Calculations
        omegaCL = rospy.get_param('~omega_cl', 0.7)  # Closed-loop bandwidth for FBL controller
        zeta = rospy.get_param('~zeta', 1.5)  # Damping coeffient
        self.kp = -omegaCL ** 2  # Proportional gain
        self.kd = -2 * omegaCL * zeta  # Differential gain

    # -------------------------- Utility Functions --------------------------------
    # Function to calculate yaw angle from quaternion in Pose message
    def headingcalc(self, pose):
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
    def state(self, loc):
        # The state is [x, y, theta] in [m, m, rad]
        curr_x = loc.pose.position.x
        curr_y = loc.pose.position.y
        curr_theta = self.headingcalc(loc.pose)
        q = np.array([curr_x, curr_y, curr_theta])

        return q

    # Function to find lateral and heading errors with respect to the closest point on path to vehicle
    def current_path_errors_slow(self, q):

        # Smart path planner - only look at path points from the previous closest point ahead
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

        # Find the smallest non-zero distance to a point, and find its index
        # self.c = np.argmin(self.dist2points[np.nonzero(self.dist2points)])            #Michael's

        # misan: above, rather than nonzero which makes the array lose its order(?),
        # try using a mask to set the zero value(s) or threshold value to 'inf',
        # then use: self.c = np.argmin(self.dist2points)

        mask = self.dist2points == 0                  # or '= self.dist2points < threshold'
        self.dist2points[mask] = np.inf                # or a very large number
        # self.c = np.int(np.argmin(self.dist2points))
        self.c = np.argmin(self.dist2points)


        # print(self.c)
        # Local variable to represent closest point at current instant
        targetpoint = self.path[self.c]

        self.true_path.append(targetpoint)

        # Heading Error Calculations

        targetpoint_heading = targetpoint[2]  # Heading of the closest point at index c in radians

        # Heading error in radians
        eh = q[2] - targetpoint_heading

        # Lateral Error Calculation - considering x and y in N-E-S-W global frame to calculate lateral error
        del_x = q[0] - (0.01*targetpoint[0])
        del_y = q[1] - (0.01*targetpoint[1])
        self.err.append(np.sqrt(del_x**2 + del_y**2))
        self.errx.append(del_x)
        self.erry.append(del_y)
        el = -del_x * np.sin(targetpoint[2]) + del_y * np.cos(targetpoint[2])

        # Summary errors into an array for cleanliness
        pf_errors = np.array([el, eh])

        return pf_errors

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
    def handle_estimate(self, data):

        # Define data from handler as estimate = odometry estimate
        self.estimate = data

        # Redefine new variable which is only pose component of Odom msg
        self.est_pose = copy.deepcopy(self.estimate.pose)

        # Redefine new variable which is absolute linear velocity from Odom msg
        self.vel = np.sqrt((self.estimate.twist.twist.linear.x) ** 2
                           + (self.estimate.twist.twist.linear.y) ** 2
                           + (self.estimate.twist.twist.linear.z) ** 2)

if __name__ == '__main__':
    rospy.init_node('get_paths')
    Contours_and_Paths()
    rospy.spin()