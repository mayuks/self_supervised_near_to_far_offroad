A set of ROS/Gazebo packages that together simulate, in closed-loop, autonomous terrain data acquisition for training a semantic segmentation network towards terrain-adaptive autonomous off-road driving. See the demonstration videos and https://youtu.be/aOfmmZdM8kI.

REF: 

O. Mayuku, B. F. Surgenor, and J. A. Marshall. A self-supervised near-to-far approach for terrain-adaptive off-road autonomous driving.  In Proceedings of the 2021 IEEE Conference on Robotics and Automation (ICRA), Xi’an, China, May 30, 2021. DOI: 10.1109/ICRA48506.2021.9562029

ALgorithms:

Note: For proprietary reasons, I used openly available versions for this repository; although my implementation was in Python and C++, I retained the name of the methods to give the authors credit. This applies to the first two algorithms only.

Image clustering:
SLIC and DBSCAN: See Matlab implementation at https://www.peterkovesi.com/projects/segmentation/

Path Planning:
AStar: See Matlab implementation at https://www.mathworks.com/matlabcentral/profile/authors/3020423

Controlller:
Feedback Linearized Controller
