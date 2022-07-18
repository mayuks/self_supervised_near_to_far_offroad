#!/usr/bin/env python

import rospy
from std_msgs.msg import Empty, String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from near_to_far_sim.msg import Num

import imutils
import numpy as np
import cv2
import math as mt
from scipy import sparse
from scipy import ndimage as ndi
from skimage import color
import copy

# rostopic pub starter std_msgs/Empty "{}" --once
#Heshan Poster advice: powerpoint 4 ft by 3 ft...Don't scale, he says

class Dbscan_and_Contours(object):
    def __init__(self):
        self.start = False
        self.count = 0
        self.cluster_attributes = 0
        self.labelled = 0
        self.bridge = CvBridge()
        self.cv_image = None

        self.scale = 1

        self.size = [np.int(200/self.scale), np.int(350/self.scale)]      # scaled real world to image size...W x H

        self.features_store = []

        # self.features_store = [[9.96372223, 1.33045149, -1.40226817, 1e1],


        rospy.Subscriber('starter', Empty, self.commence_prog, queue_size=1)
        rospy.Subscriber('features_store', Num, self.store_features, queue_size=1)
        rospy.Subscriber('realsense/color/image_raw', Image, self.callback, queue_size=1)

        self.T = rospy.get_param('~rate', 0.1)
        self.timer = rospy.Timer(rospy.Duration(self.T), self.timer_callback)
        self.pub = rospy.Publisher('chatter', Num, queue_size=1, latch=True)           #from tutorial: funny...if latch is
                                                                                # omitted (False) the subsciber misses the first message
                                                                           # careful though something about it not being copied
                                                                        # http://wiki.ros.org/rospy/Overview/Publishers%20and%20Subscribers
                                                                        # copy.deepcopy() the variable before yourself sending? Just in case you need to
                                                                        # edit it in subsequent lines after publishing?
        self.pub_label_vib = rospy.Publisher('label_no_and_IMU', Num, queue_size=1, latch=True)

    def commence_prog(self, data):
        self.start = True

    def store_features(self, data):

        temp  = copy.deepcopy(data.cluster_attributes)
        temp = np.reshape(temp, (-1, 4))
        temp = temp.tolist()
        self.features_store = self.features_store + temp

    def callback(self, data):
        try:
            pass
        except Exception as err:
            print (err)
        self.cv_image = self.bridge.imgmsg_to_cv2(copy.deepcopy(data), "rgb8")


    def my_slic(self, image, k, m, seRadius, nItr):

        [rows, cols, chan] = image.shape

        im = color.rgb2lab(image)       #better for tiles vs black mat


        S = mt.sqrt(rows * cols / (k * mt.sqrt(3) / 2))

        nodeCols = np.around(cols / S - 0.5)

        S = cols / (nodeCols + 0.5)

        nodeRows = np.around(rows / (mt.sqrt(3) / 2 * S))

        vSpacing = rows / nodeRows

        k = nodeRows * nodeCols

        C = np.zeros((6, int(k)))

        l = -np.ones((rows, cols))

        d = np.full((rows, cols), np.inf)

        kk = 0

        r = vSpacing / 2.0

        for ri in range(1, int(nodeRows + 1)):

            if bool(ri % 2):
                c = S / 2
            else:
                c = S

            for ci in range(0, int(nodeCols)):
                cc = int(np.around(c))
                rr = int(np.around(r))
                C[0:5, kk] = [im[rr - 1, cc - 1, 0], im[rr - 1, cc - 1, 1], im[rr - 1, cc - 1, 2], cc - 1, rr - 1]
                # or C[0:5, kk] = np.concatenate([np.squeeze(im[rr,cc,:])], [cc, rr]) see also hstack and vstack
                c = c + S
                kk = kk + 1

            r = r + vSpacing

        S = np.around(S)

        for n in range(0, nItr):

            for kk in range(0, int(k)):
                rmin = int(max(C[4, kk] - S, 0))
                rmax = int(min(C[4, kk] + S, rows - 1))
                cmin = int(max(C[3, kk] - S, 0))
                cmax = int(min(C[3, kk] + S, cols - 1))

                subim = im[rmin:rmax + 1, cmin:cmax + 1, :]

                # make a copy of subim...confrim why using subim as is gives a huge value in the subim loop in 'dist'

                assert (subim.size > 0), 'subim has a zero dimension at(%d)' % kk

                D = self.dist(C[:, kk], subim.copy(), rmin, cmin, S, m)

                subd = d[rmin:rmax + 1, cmin:cmax + 1]
                subl = l[rmin:rmax + 1, cmin:cmax + 1]
                updateMask = D < subd
                subd[updateMask] = D[updateMask]
                subl[updateMask] = kk + 1

                d[rmin:rmax + 1, cmin:cmax + 1] = subd
                l[rmin:rmax + 1, cmin:cmax + 1] = subl

            C[:] = 0

            for r in range(0, int(rows)):

                for c in range(0, int(cols)):
                    tmp = np.array([im[r, c, 0], im[r, c, 1], im[r, c, 2], c, r, 1])

                    C[:, int(l[r, c] - 1)] = C[:, int(l[r, c] - 1)] + tmp

            for kk in range(0, int(k)):
                C[0:5, kk] = np.around(C[0:5, kk] / C[5, kk])

        l, Am = self.mcleanupregions(l, seRadius)

        N = Am.shape[0]

        L = np.zeros((N, 1), dtype=np.float32)
        a = np.zeros((N, 1), dtype=np.float32)
        b = np.zeros((N, 1), dtype=np.float32)

        for n in range(1, N):           #for mean
            mask = l == (n)

            nm = np.sum(mask)

            L[n] = sum(im[:, :, 0][mask]) / nm
            a[n] = sum(im[:, :, 1][mask]) / nm
            b[n] = sum(im[:, :, 2][mask]) / nm

        ###########################################################################################################
        #
        # SLIC ENDS
        #
        # SPDBSCAN starts here
        ###########################################################################################################

        Np = len(L)

        # Ec = 3.5    #back full
        Ec = 2.3    #webcam
        # Ec = 1.04   #icra/Lab
        # Ec = 1.01   #Gordon #many days up toMay 17
        # Ec = 1.5     #try
        # Ec = 0.75

        regionsC = np.zeros((Np, 1))

        C = []

        Nc = 0

        Pvisit = np.zeros((Np, 1))

        # check indexing issues as always is 0 equi to 1 ...

        for n in range(0, Np):  # starting from 0 gives problems, adjust regionsC, Pvisit, accordingly

            if Pvisit[n] == 0:

                Pvisit[n] = 1

                neighbours = self.regionQueryM(L, a, b, Am, n, Ec)

                Nc = Nc + 1

                C.append([n])

                regionsC[n] = Nc

                ind = 1

                while ind <= len(neighbours):

                    nb = neighbours[ind - 1]

                    if Pvisit[nb] == 0:
                        Pvisit[nb] = 1

                        neighboursP = self.regionQueryM(L, a, b, Am, nb, Ec)

                        if len(neighboursP) > 1:

                            for neighbors in neighboursP:
                                neighbours.append(neighbors)
                        else:
                            neighbours.append(neighboursP)

                    # problem with the above line cos neighboursP can be a list, so when appended to neigbours in the next iteration, nb becomes a list, not an integer....
                    # flatten neighboursP before appending?
                    # or loop through and append each integer in the list
                    # see also:https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-list-of-lists

                    if regionsC[nb] == 0:
                        regionsC[nb] = Nc

                        Nz = Nc - 1  # this is the current length of the list C
                        # i.e. append neighbours (nb) to n
                        C[Nz].append(nb)

                    ind = ind + 1

        lc = np.zeros(l.shape)

        # np.savetxt("regionsC_before", regionsC, delimiter=",")

        # regionsC = regionsC + 1
        for n in range(0, (len(regionsC))):
            lc[l == n + 1] = regionsC[n]

        # np.savetxt("regionsC_after", regionsC, delimiter=",")

        # ###################################################################################################
        #
        # Testcomparison starts here
        #
        # ###################################################################################################

        # E = 8  #webcam               # keep somewhat lowto keep individual clusters as pure as possible...
        # E = 6.3        #Lab/icra            # e.g the tire burn marks separate from wood in the rough wood faux terrain...
                            # test the effect by comparing dbscan labelled image to testcomparison.m labelled image
        E = 10  #Gordon
        cluster_features = []

        for i in range(0, len(C)):

            featureq = 0

            for q in range(0, len(C[i])):
                featuresq = np.array((L[C[i][q]], a[C[i][q]], b[C[i][q]]))

                featureq = featureq + featuresq

            featureq = featureq / len(C[i])

            cluster_features.append(featureq)

        print(cluster_features[0])

        A = np.ones((len(cluster_features), 1))
        similar_cluster_indices = []
        similar = np.zeros((len(A), len(A)))
        E2 = E ** 2

        for i in range(0, len(A)):
            for j in range(0, len(A)):

                # for L, use 0.15 for test4.png, else try 0
                v = np.array((0 * cluster_features[i][0], cluster_features[i][1], cluster_features[i][2])) - \
                    np.array((0 * cluster_features[j][0], cluster_features[j][1], cluster_features[j][2]))

                dist2 = np.matmul(v.T, v)
                if dist2 < E2:
                    similar[i, j] = 1
                else:
                    similar[i, j] = 0

        for k in range(0, len(A)):
            if A[k] == 0:
                similar_cluster_indices.append(-1)
            else:
                z = np.nonzero(similar[k, :])
                similar_cluster_indices.append(z)
                for p in range(0, len(z)):
                    A[z[p]] = 0

        for x in range(similar_cluster_indices.count(-1)):
            similar_cluster_indices.remove(-1)  # ...cos the 'remove' method removes the element one at at time
            # https://www.includehelp.com/python/remove-all-occurrences-a-given-element-from-the-list.aspx

        D = []


        for i in range(0, len(similar_cluster_indices)):
            z = similar_cluster_indices[i]

            B = []
            # Q = []

            for k in range(0, len(z[0])):
                B.append(C[z[0][k]])
                # print(C[z[0][k]])
                # Q.append(len(C[z[0][k]]))

            # flatten out the list of lists before appending to D (see also neighboursP to compare neighbour appending in spdbscan)
            # https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-list-of-lists
            B_flat = []
            for sublist in B:
                for item in sublist:
                    B_flat.append(item)
            D.append(B_flat)


        regionsD = np.zeros_like(regionsC)

        for i in range(0, len(D)):
            z = D[i]

            for k in range(0, len(z)):
                regionsD[z[k]] = i + 1  # start numbering from 1
                # regionsD[z[k]] = i

        # regionsD = regionsD + 1  # start numbering from 1


        ld = np.zeros_like(l)

        for n in range(1, len(regionsD) + 1):
            ld[l == n] = regionsD[n - 1]  # labelled image numbering starts at 1, regionsD indexing starts at zero

        for n in range(1,len(D) + 1):
            lv = np.zeros_like(l)
            lv[ld == n] = n * 20
            lv = (lv).astype(np.uint8)
            cv2.imshow("lv", lv)
            cv2.waitKey(1000)
            # cv2.waitKey(0)
        cv2.destroyAllWindows()

        # lb = (ld).astype(np.uint8)
        # new_im = Im.fromarray(
        #     lb)  # https://www.kite.com/python/examples/4887/PIL-convert-between-a-pil-%60image%60-and-a-numpy-%60array%60
        # new_im.save("lb.png")

        super_cluster_features = []

        # np.savetxt("labeled_imaage.csv", ld, delimiter=",")

        for i in range(0, len(D)):

            featureQ = 0

            for q in range(0, len(D[i])):
                featuresQ = np.array((L[D[i][q]], a[D[i][q]], b[D[i][q]]))

                featureQ = featureQ + featuresQ

            featureQ = featureQ / len(D[i])

            super_cluster_features.append(featureQ)
        print(len(D))
        print(len(super_cluster_features))
        print(np.amax(ld))

        # make it an array... to be reshaped as ID array for ROS message
        super_cluster_features_matrix = np.zeros((len(super_cluster_features), super_cluster_features[0].shape[
            0]))  # each row covers one setof cluster features

        for i in range(0, super_cluster_features_matrix.shape[0]):
            for j in range(0, super_cluster_features_matrix.shape[1]):
                super_cluster_features_matrix[i, j] = super_cluster_features[i][j]


        print(super_cluster_features_matrix)
        print(super_cluster_features_matrix.shape)

        return super_cluster_features_matrix, ld
    #################################################################################################

    # Testcomparison ends

    ##################################################################################################


    def dist(self, C, subim, rmin, cmin, S, m):

        [rows, cols, chan] = subim.shape

        x, y = np.meshgrid(np.arange(cmin, (cmin+cols)), np.arange(rmin, (rmin+rows)))

        x = x-C[3]
        y = y-C[4]
        ds2 = x**2 + y**2

        for n in range(0, 3):
            subim[:, :, n] = (subim[:, :, n] - C[n])**2

        # dc2 = subim.sum(axis=2)
        dc2 = np.sum(subim, axis =2)
        D = (dc2 + ((ds2/(S**2))*(m**2)))**0.5

        return D

    #################################################################################################

    def makeregionsdistinct(self, seg, connectivity=4):

        labels = np.unique(seg.flatten())               #or use: np.amx(a, axis=none)
        maxlabel = max(labels)
        labels = np.setdiff1d(labels, 0)

        for l in labels:
            bl = (seg==l).astype(np.uint8)
            output = cv2.connectedComponentsWithStats(bl, connectivity, cv2.CV_32S)

            if output[0] > 1:
                for n in range(2, output[0]+1):
                    maxlabel = maxlabel+1
                    seg[bl==n] = maxlabel

        return seg, maxlabel

    ###############################################################################################

    def finddisconnected(self, l):

        Am, Al = self.regionadjacency(l)

        N = np.max(l).astype(int)

        print(N)

        visited = np.zeros((N+1,1))     #python is row major so ravel ops are consistent for sparse csr matrix (Am) and
        list = []                       #visited array  (see A and B below)...just in case the nonzero ravel numbering
        listNo = -1                     #is diff for row vs column matrices (checked in ndimage_things script doesn't matter, can u PK's)

        for n in range (0, N):    #check indexing...fromzero instead?
            # print(visited[n])
            if  not visited[n]:
                # print(n)
                listNo = listNo + 1
                list.append([n])
                visited[n] = 1

                A_chk =  np.logical_not((Am[n,:]).toarray())

                A  = np.nonzero(A_chk)
                A = np.ravel_multi_index(A, A_chk.shape)                  # to linear indexing

                B = np.nonzero(visited)
                B = np.ravel_multi_index(B, visited.shape)

                notConnected = np.setdiff1d(A, B)

                for m in notConnected:

                    if not np.any([np.intersect1d(Al[m], list[listNo])]):
                        (list[listNo]).append(m)

                        visited[m] = 1
        return list

    ###################################################################################

    def renumberregions(self, L):
        # doesn't appear to work as advertised...i.e to renumber regions sequentially
        nL = L
        # labels = np.unique(L.flatten())
        labels = np.unique(L.ravel())     #faster?
        N =len(labels)

        if labels[0] == 0:

            # check this slice
            labels = labels[1:]
            minLabel = 0
            maxLabel = N-1                 #not used but check python matlab indexing issue
        else:
            minLabel =1
            maxLabel = N                #not used but check

        count = 1
        for n in labels:
            nL[L==n] = count
            count = count + 1

        return nL

    ###################################################################################

    def circularstruct(self, radius):

        dia= mt.ceil(2*radius)

        if bool(dia % 2) == 0:              # or if not bool...
            dia = dia + 1

        r = np.fix(dia/2)

        x, y = np.meshgrid(np.arange(-r, r+1), np.arange(-r, r+1))

        rad = (x**2 + y**2)**0.5

        strel = rad <= radius

        # strel = np.min(rad, radius)           # not used?
        return strel

    ###############################################################################################

    def mcleanupregions(self, seg, seRadius):

        seg, maxlabel = self.makeregionsdistinct(seg)
        se = self.circularstruct(seRadius)

        mask=np.zeros_like(seg)

        lst  = self.finddisconnected(seg)

        for n in range(0, len(lst)):
            b = np.zeros_like(seg)

            for m in range(0,len(lst[n])):
                b = np.logical_or(b, (seg==lst[n][m]))
            # print(type(b))
            b = b.astype(np.uint8)
            se = se.astype(np.uint8)
            # b - cv2.morphologyEx(b, cv2.MORPH_OPEN, se)
            mask= np.logical_or(mask, (b - cv2.morphologyEx(b, cv2.MORPH_OPEN, se)))

        # open cv's and scikit image transform are inverse of matlab"s so don't use mask inverse as in PK's

        edt, inds = ndi.distance_transform_edt(mask, return_indices=True)

        inds = np.ravel_multi_index(inds, mask.shape, order = 'C')

        seg[mask] = seg.flat[inds[mask]]


        seg, maxlabel = self.makeregionsdistinct(seg)

        seg = self.renumberregions(seg)

        Am, Al = self.regionadjacency(seg)

        return seg, Am
        # return se
    ##################################################################################################

    def regionadjacency(self, L, connectivity=4):

        [rows, cols] = L.shape

        labels = np.unique(L.flatten())

        labels = np.setdiff1d(labels, 0)

        i = np.zeros((2*(rows-1)*(cols-1), 1))
        j = np.zeros((2*(rows-1)*(cols-1), 1))
        s = np.zeros((2*(rows-1)*(cols-1), 1))

        n = 0

        for r in range(0, rows-1):
            for c in range(0, cols-1):

                i[n] = L[r,c]
                j[n] = L[r, c+1]
                s[n] = 1
                n = n+1

                i[n] = L[r,c]
                j[n] = L[r+1, c]
                s[n] = 1
                n = n+1
        i = i.flatten().astype(np.int)
        j = j.flatten().astype(np.int)
        s = s.flatten().astype(np.int)


        Am = sparse.csr_matrix((s,(i,j)),shape=None)

        #lil matrix advise in terminal comes from here, check...

        for r in range (0, Am.shape[0]):
            Am[r,r] = 0

        Am = Am + Am.T          #same as logical_or

        Al = []
        for r in range(0, Am.shape[0]):
            Al.append((np.ravel_multi_index(np.nonzero(Am[r,:]), Am[r,:].shape)))

        Al = np.asarray(Al)
        return Am, Al

    #############################################################################################################################

    def regionQueryM(self, L, a, b, Am, n, Ec):

        E2 = Ec ** 2

        neighbours = []

        ind = np.nonzero(Am[n,:])           #if a vector why is ind 2D?
        # print(Am[0,:])                      #this ans the above Q...each element is 2D(?), so nonzero returns a 2-tuple
        ind = ind[1]

        for i in ind:
            # v = np.array((L[i], a[i], b[i])) - np.array((L[n], a[n], b[n]))

            # suppress illumination
            v = np.array((0*L[i], a[i], b[i])) - np.array((0*L[n], a[n], b[n]))

            dist2 = np.matmul(v.T, v)

            if dist2 < E2:
                neighbours.append(i)

        return neighbours

############################################################################################

    def four_point_transform(self, img):
        # obtain a consistent order of the points and unpack them
        # individually

        tl = [298, 489]  # Calibration 05 May 2021 ends 13 May 2021
        tr = [503, 489]
        br = [791, 739]
        bl = [9, 739]

        rect = np.array(([tl, tr, br, bl]), dtype="float32")

        # now that we have the dimensions of the new image, construct
        # the set of destination points to obtain a "birds eye view",
        # (i.e. top-down view) of the image, again specifying points
        # in the top-left, top-right, bottom-right, and bottom-left
        # order

        maxWidth = self.size[0]
        maxHeight = self.size[1]

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")

        # compute the perspective transform matrix and then apply it
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

        store = cv2.cvtColor(warped, cv2.COLOR_RGB2BGR)

        # filename = 'figs/' + str(time.clock())+'.png'
        # cv2.imwrite(filename, store)                #for the record
        #
        cv2.imwrite("/home/misan/figs/transformed_resized_in_use_sim.tiff", store)

        # return the warped image
        return warped
    ############################################################################################################################

    # Get Driveable-sized clusters starts here

    #############################################################################################################################

    def get_driveble_sized_clusters(self, attributes, labelled_img):

        im = labelled_img.astype(np.uint8)
        driveable_features_plus_label_no = []
        attributes = attributes.tolist()

        Area_centrepts = []
        boxpts = []
        drivable_sized_clusters = []

        # half_box_size = np.int(23 / self.scale)  # 23 is half size of faux terrain tile

        for n in range(1, np.amax(im) + 1):
            image = (im == n).astype(np.uint8)

            cnt = cv2.findContours(image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            cnt = imutils.grab_contours(cnt)

            cnt = sorted(cnt, key=cv2.contourArea, reverse=True)[0]

            # test = cv2.contourArea(cnt)
            #
            # print("cnt area is {}".format(test))

            cnt = np.asarray(cnt, dtype=np.float32)

            cnt = np.asarray(cnt, dtype=np.int)

            rect = cv2.minAreaRect(cnt)

            # append if size threshold is met and1 otherwise,to maintain position on array to keep track of class_numbers
            if rect[1][0]>=46/self.scale and rect[1][1]>=46/self.scale:
                #make a decison of size cut-off, and change 46 to appropriate number based on robot size (overall width with some allowance)
                # TO DO: also check that the number of the class in the box meets a threshold
                box_area = rect[1][0]*rect[1][1]
                contour_area = cv2.contourArea(cnt)
                # print("cnt area is {}".format(contour_area))
                # print("cnt_box area is {}".format(box_area))
                # print(contour_area/box_area)

                #check contour solidity meets a theshold...specify the threshold based on noise...e.g grass with leaves in the fall will have more voids in clusters
                #note tht the contourArea won't work correctly for self intersecting contours...see docs
                if (contour_area/box_area)>=0.2:   #0.6
                    drivable_sized_clusters.append(1)
                    # driveable_features_plus_label_no.append([attributes[n-1][0], attributes[n-1][1], attributes[n-1][2], n])
                    driveable_features_plus_label_no.append(attributes[n - 1][0:3] + [n])
                    # or do this outside loop with enumerate(ziP())...see below before 'return' (commented)
                    # last one is the cluster label which starts from '1', but the corresponding attribute is n-1 as n starts from 1
                    # i.e. Label 1 has attribute[0]
                else:
                    drivable_sized_clusters.append(0)
            else:
                drivable_sized_clusters.append(0)

        return driveable_features_plus_label_no

    ###########################################################################################################################

    #compare_superclusters strarts here

    ##########################################################################################################################

    def compare_superclusters(self, current_superclusters):

        features_bank = copy.deepcopy(self.features_store)
        known_features_with_IMU_added = []
        label_no_and_vib_data = []
        unknown_features_store = []
        unknown_supercluster = []
        unknown_supercluster_label = []

        E = 12.5
        similar = np.zeros((len(current_superclusters), len(features_bank)))        #i.e without the label no, and IMU data columns respectively
        dist = np.zeros_like(similar)

        E2 = E ** 2

        for i in range(0, len(current_superclusters)):
            for j in range(0, len(features_bank)):

                v = np.array((0 * features_bank[j][0], features_bank[j][1], features_bank[j][2])) - \
                    np.array((0 * current_superclusters[i][0], current_superclusters[i][1], current_superclusters[i][2]))
                # print(v.shape)

                dist2 = np.matmul(v.T, v)
                dist[i, j] = dist2

                if dist2 < E2:
                    similar[i, j] = 1
                else:
                    similar[i, j] = 0

            if not np.any(similar[i, :]):
                # unknown_features_store.append([current_superclusters[j][0], current_superclusters[j][1], current_superclusters[j][2], current_superclusters[j][3], 0])
                # more like:
                # current_superclusters[i] = current_superclusters[i][0:4] + [0]
                # self.features_store.append([current_superclusters[i][0], current_superclusters[i][1], current_superclusters[i][2], 0])
                # zero is a placeholder for the IMU data to be collected

                unknown_supercluster.append(current_superclusters[i][0:3])                      # a list of lists
                unknown_supercluster_label.append(np.int(current_superclusters[i][3]))          # just a list
                #send to contours and paths, to plan a path to this cluster(s), drive over, get IMU data...
                # and send back here to festures_store here for appending

            else:
                k = np.argmin(dist[i, :])       # cos more than one feature_store cluster might meet the threshold (E), why it needs careful choice
                # known_features_with_IMU_added.append(current_superclusters[i][0:4] + features_bank[k][3])   #see 'list concatenation'
                # OR maybe only append the cluster index in current_superclusters i.e. index 3, and the IMU data and send to ASTAR with labelled Image ...
                label_no_and_vib_data.append([current_superclusters[i][3]] + [features_bank[k][3]])
                # at ASTAR copy the IMU data to labelled image using masks and any non-driveable label is given an optimistic or pessimistic cost
                # i.e highest or lowest
        print("similar is")
        print(similar)

        print("dist is")
        print(dist)
        #FOR LATER
        # for i in range(0, similar.shape[0]):
        #     if not np.any(similar[i, :]):
        #         self.features_store.append([current_superclusters[i][0], current_superclusters[i][1], current_superclusters[i][2], 0] )
        #         #zero is a placeholder for the IMU data to be collected
        return unknown_supercluster, unknown_supercluster_label, label_no_and_vib_data

    ############################################################################################################################

    #Timer Callback starts heere
    ############################################################################################################################

    def timer_callback(self, event):

        if self.start is False:
            rospy.logwarn_throttle(2.5, "Send empty message to start program...")
            return
        # elif self.cv_image is None:
        #     rospy.logwarn_throttle(2.5, "Waiting for Gazebo image...")
        #     return

        # get transformed top-down view of image
        image = self.four_point_transform(self.cv_image)

        # image = cv2.imread('/home/offroad/figs/transformed_resized_in_use_sim.tiff')
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        #get dbscan clusters
        # subim = self.my_slic(image, np.int(6000/self.scale), 15, 1, 1) #icralab
        # subim = self.my_slic(image, np.int(3000/self.scale), 30, 1, 1) #webcam
        # subim = self.my_slic(image, np.int(6000 / self.scale), 50, 1.5, 10)  # Gordon1
        # subim = self.my_slic(image, 3000, 50, 1.5, 1)      #20/04

        subim = self.my_slic(image, 6000/self.scale, 30, 1.5, 1)

    ####################################################################################################
        #send msg to correct node
    ########################################################################################################

        if self.count == 0:

            # self.cluster_attributes = subim[0]
            # self.labelled = subim[1]

            cluster_attributes = np.ravel(subim[0])
            print(cluster_attributes)
            print(np.reshape(cluster_attributes, (-1, 3)))  # test 'unravelling' i.e. reshaping

            print(subim[1].shape)
            labelled = (np.ravel(subim[1])).astype(np.uint8)

            msg = Num()
            msg.labelled = copy.deepcopy(labelled)
            msg.no_IMU_cluster_attributes = copy.deepcopy(cluster_attributes)


            rospy.loginfo(msg)
            self.pub.publish(msg)

        if self.count > 0:

            self.cluster_attributes = subim[0]
            self.labelled = subim[1]

            clusters_of_interest = self.get_driveble_sized_clusters(subim[0], subim[1])
            print('clusters of interest are:')
            print(clusters_of_interest)

            unknown, unknown_label, known = self.compare_superclusters(clusters_of_interest)

            if len(unknown) == 0:     #if not unknown...i.e all clusters known

                print('ALL CLUSTERS KNOWN')
                print(known)

                known = np.array(known)         # might not be necessary to cast to array first, ravel does that

                print(known)

                known = np.ravel(known)

                print(known)

                labelled = (np.ravel(subim[1])).astype(np.uint8)

                msg = Num()
                msg.labelled = copy.deepcopy(labelled)
                msg.cluster_attributes = copy.deepcopy(known)


                rospy.loginfo(msg)
                self.pub_label_vib.publish(msg)         # note the publishing object used...i.e. all known so straght to ASTAR.py

            else:
                print('AT LEAST ONE CLUSTER UNKNOWN')

                labelled = (np.ravel(subim[1])).astype(np.uint8)

                known = np.array(known)         # might not be necessary to cast to array first, ravel does that
                known = np.ravel(known)

                unknown = np.array(unknown)
                unknown = np.ravel(unknown)

                unknown_label = np.array(unknown_label)

                msg = Num()
                msg.labelled = copy.deepcopy(labelled)
                msg.cluster_attributes = copy.deepcopy(known)
                msg.no_IMU_cluster_attributes = copy.deepcopy(unknown)
                msg.no_IMU_cluster_attributes_label = copy.deepcopy(unknown_label)


                rospy.loginfo(msg)
                self.pub.publish(msg)  # note the publishing object used...i.e. at least one unknown so go to get contours_and_path to get IMU data

        self.count = self.count + 1

        self.start = False


if __name__ == '__main__':

    rospy.init_node('clusters_and_contours')

    Dbscan_and_Contours()

    rospy.spin()

