# Object Detection

![ROS1](https://img.shields.io/badge/ROS1-blue)

![fag1](../images/section_2/object_detection/header_object_detection.png)

## Overview

Object detection is a fundamental capability in autonomous vehicle systems, enabling the vehicle to perceive and understand its surrounding environment. This workshop focuses on **3D object detection** using raw LiDAR data within the **Robot Operating System (ROS)** framework. Specifically, we will work with data recorded from a test vehicle equipped with a [Velodyne VLP-32C](https://icave2.cse.buffalo.edu/resources/sensor-modeling/VLP32CManual.pdf) LiDAR sensor. Utilizing a state-of-the-art 3D object detection model, participants will learn to predict bounding boxes around detected objects, facilitating tasks such as navigation, obstacle avoidance, and path planning.

## Learning Objectives

By the end of this workshop, participants will be able to:

- **Understand ROS Object Definitions:** Comprehend how objects are defined and structured within ROS messages.
- **Visualize LiDAR Point Clouds in RViz:** Utilize RViz to visualize and interpret raw LiDAR data.
- **Launch and Operate ROS Nodes:** Execute ROS nodes that apply detection algorithms to raw sensor data.
- **Visualize Detected Objects in RViz:** Use RViz to display and analyze detected objects with bounding boxes.

## Prerequisites

- **Basic Knowledge of ROS:** Familiarity with ROS concepts, including nodes, topics, and messages.
- **C++ Programming Skills:** Ability to read and modify C++ code.
- **Understanding of LiDAR Technology:** Basic comprehension of LiDAR sensors and point cloud data.
- **Experience with RViz:** Prior experience using RViz for visualization is beneficial but not mandatory.

## Setup Instructions

### 1. Clone the Repository and Navigate to the Workspace

Ensure you have access to the necessary ROS workspace. Navigate to your workspace directory:

```bash
cd ~/catkin_workspace/src
```

### 2. Clone the Necessary Packages

Clone the `lidar_detection` package along with its dependencies:

```bash
git clone https://github.com/ika-rwth-aachen/acdc.git
cd acdc/catkin_workspace/src
git clone https://github.com/ika-rwth-aachen/acdc.git
```

### 3. Download the Required Bag File

Download the LiDAR data recording from Campus Melaten in Aachen:

```bash
wget -O lidar_campus_melaten.bag https://rwth-aachen.sciebo.de/s/udlMYloXpCdVtyp/download
```

Alternatively, access the bag file directly [**here**](https://rwth-aachen.sciebo.de/s/udlMYloXpCdVtyp) (approx. 1.5 GB). Save the file to the local directory `${REPOSITORY}/bag` on your host machine, which is mounted to `~/bag` in the Docker container.

### 4. Build the Workspace

Navigate to your workspace and build the packages:

```bash
cd ~/catkin_workspace
catkin build
source devel/setup.bash
```

*Note:* If you encounter a compilation error similar to `g++: internal compiler error: Killed (program cc1plus)`, it indicates excessive resource consumption. To resolve this, disable parallel building:

```bash
catkin build -j 1
source devel/setup.bash
```

### 5. Launch the ROS Environment with RViz and Rosbag

To streamline the process of launching `rosbag play` and `RViz` simultaneously, utilize the provided launch file:

```bash
roslaunch lidar_detection start_rosbag_play_rviz.launch
```

This command performs the following actions:

- **Rosbag Playback:** Plays the `lidar_campus_melaten.bag` file.
- **RViz Visualization:** Launches RViz with a pre-configured display for point clouds.

#### Expected Terminal Output:

```bash
Initialization of Trajectory Planner done!
Trajectory optimization SUCCESSFUL after [...]s.
Trajectory optimization SUCCESSFUL after [...]s.
...
```

#### Expected RViz Visualization:

![fag1](../images/section_2/object_detection/rviz.png)

- **PointCloud2 Display:** Shows the raw LiDAR point cloud data.
- **Adjust Visualization Settings:** Enhance visibility by modifying parameters such as `Size`, `Style`, `Decay Time`, and `Color Transformer` in the `PointCloud2` tab.

#### RViz Navigation Controls:

- **Left Mouse Button:** Rotate the view around the Z-axis.
- **Middle Mouse Button:** Pan the camera along the XY plane.
- **Right Mouse Button:** Zoom in and out.
- **Scroll Wheel:** Zoom in and out incrementally.

**Congratulations!** You have successfully visualized the raw LiDAR data. Proceed to the object detection algorithms in the subsequent sections.

## Definitions

### ROS Object Definitions

Understanding how objects are defined and structured within ROS is crucial for effective communication between nodes and for processing detection results.

#### ika ROS Object Definition

The **ika** definitions for ROS messages and internal utilities are located in the [*definitions*](https://github.com/ika-rwth-aachen/acdc/blob/main/catkin_workspace/src/dependencies/definitions) package. Specifically:

- **ROS Message Files:** Located in `definitions/msg`.
- **Internal Definitions:** Found in `~/ws/catkin_workspace/src/dependencies/definitions/include/definitions/utility`.

##### IkaObject.msg

Defines the structure of a single 3D object detected by the LiDAR sensor.

```bash
float32[] fMean               # State vector, containing attributes based on the chosen motion model
float32[] fCovariance         # Covariance matrix, representing uncertainties in the state vector
```

- **fMean:** Represents the object's bounding box attributes such as position, velocity, acceleration, and orientation.
- **fCovariance:** Captures the uncertainties associated with each attribute in `fMean`, essential for tasks like object fusion.

*Note:* This assignment focuses solely on 3D object detection, thus only the `fMean` vector is utilized to describe an object's bounding box.

##### object_definitions.h

Defines the enumeration of object types recognized by the system.

```c++
enum ika_object_types {
  UNCLASSIFIED = 0,
  PEDESTRIAN = 1,
  BICYCLE = 2,
  MOTORBIKE = 3,
  CAR = 4,
  TRUCK = 5,
  VAN = 6,
  BUS = 7,
  ANIMAL = 8,
  ROAD_OBSTACLE = 9,
  TRAILER = 10,
  TYPES_COUNT = 11
};
```

- **Key Object Classes:** For simplicity, this workshop focuses on the following classes:
  - `CAR` (ID: 4)
  - `PEDESTRIAN` (ID: 1)
  - `TRUCK` (ID: 5)
  - `BICYCLE` (ID: 2)

*Example Usage:* Access the `CAR` type in code using `definitions::ika_object_types::CAR`.

#### ika ROS Object Lists Definition

Handling multiple objects efficiently is essential for real-time applications. The `IkaObjectList.msg` facilitates this by aggregating multiple `IkaObject` messages into a single list.

##### IkaObjectList.msg

```bash
std_msgs/Header header

# List meta information
uint8 IdSource    # See definitions/utility/object_definitions.h for enum of sensors

# Actual objects
IkaObject[] objects
```

- **header:** Contains timestamp and frame information.
- **IdSource:** Identifies the sensor source of the object detections.
- **objects:** An array of `IkaObject` messages, allowing the transmission of multiple detected objects simultaneously.

*Benefits:*

- **Efficiency:** Reduces the overhead of publishing individual object messages.
- **Organization:** Simplifies the management and processing of multiple detections from various sources.

## Measurement Data

For this workshop, we utilize real sensor data captured from our institute's test vehicle equipped with a **Velodyne VLP-32C** LiDAR sensor. This sensor provides high-resolution 3D point clouds at a rate of 10 Hz, making it ideal for dynamic object detection tasks.

### Downloading the LiDAR Bag File

Retrieve the recorded LiDAR data:

```bash
wget -O lidar_campus_melaten.bag https://rwth-aachen.sciebo.de/s/udlMYloXpCdVtyp/download
```

*Alternatively,* access the bag file directly [**here**](https://rwth-aachen.sciebo.de/s/udlMYloXpCdVtyp) (approx. 1.5 GB). Save the file to the local directory `${REPOSITORY}/bag` on your host machine, which is mounted to `~/bag` within the Docker container.

### Rosbag Inspection

Inspect the contents of the bag file using the `rosbag info` command:

```bash
rosbag info lidar_campus_melaten.bag 
```

#### Expected Output:

```bash
path:        lidar_campus_melaten.bag 
version:     2.0
duration:    1:59s (119s)
start:       Feb 05 2020 15:25:31.41 (1580916331.41)
end:         Feb 05 2020 15:27:31.37 (1580916451.37)
size:        1.5 GB
messages:    1200
compression: none [1199/1199 chunks]
types:       sensor_msgs/PointCloud2 [1158d486dd51d683ce2f1be655c3c181]
             tf2_msgs/TFMessage      [94810edda583a504dfda3829e70d7eec]
topics:      /points2     1199 msgs    : sensor_msgs/PointCloud2
             /tf_static      1 msg     : tf2_msgs/TFMessage
```

- **Topics:**
  - `/points2`: Contains `sensor_msgs/PointCloud2` messages representing raw LiDAR point clouds.
  - `/tf_static`: Contains `tf2_msgs/TFMessage` messages for static transformations.

*Note:* `sensor_msgs/PointCloud2` is a standard ROS message type for point cloud data, documented [here](http://docs.ros.org/noetic/api/sensor_msgs/html/msg/PointCloud2.html).

### Rosbag Visualization

Visualizing the LiDAR point clouds provides insights into the sensor data and facilitates debugging.

#### Using a Launch File for Visualization

Instead of manually starting `roscore`, `rviz`, and `rosbag play`, utilize the provided launch file to execute these commands simultaneously.

##### Launch File: `start_rosbag_play_rviz.launch`

Located in the `lidar_detection` package, this launch file orchestrates the playback of the bag file and the visualization in RViz.

```xml
<launch>
    <!-- Rosbag Playback -->
    <param name="use_sim_time" value="true"/>
    <node 
        pkg="rosbag"
        type="play"
        args="--clock -l -r 0.5 -d 1 /home/rosuser/bag/lidar_campus_melaten.bag"
        name="player"
        output="screen">
    </node>

    <!-- RViz Visualization -->
    <node
        type="rviz"
        name="rviz"
        pkg="rviz"
        args="-d $(find lidar_detection)/rviz/point_cloud.rviz">
    </node>    
</launch>
```

- **Parameters:**
  - `use_sim_time`: Synchronizes ROS time with simulation time.
  - `rosbag play` arguments:
    - `--clock`: Publishes simulated clock time.
    - `-l`: Loops the bag file indefinitely.
    - `-r 0.5`: Plays the bag file at half speed.
    - `-d 1`: Delays the start by 1 second.

#### Launching the Visualization

Execute the launch file to start playback and visualization:

```bash
roslaunch lidar_detection start_rosbag_play_rviz.launch
```

#### Expected RViz Window:

![fag1](../images/section_2/object_detection/rviz.png)

- **PointCloud2 Display:** Visualizes the raw LiDAR point clouds.
- **Customization:** Enhance visualization by adjusting settings such as `Size`, `Style`, `Decay Time`, and `Color Transformer` within the `PointCloud2` tab.

*Tip:* Modify RViz settings to improve clarity and highlight specific features of the point cloud data.

## Lidar Detection

The `lidar_detection` ROS package facilitates the detection of 3D objects from raw LiDAR data. It leverages the **PointPillars** deep learning model to infer bounding boxes around detected objects, which are then published as `IkaObjectList` messages for further processing.

### Package Structure

```bash
section_2/
└── lidar_detection
    ├── CMakeLists.txt
    ├── include
    │   ├── definitions.h
    │   ├── detector.h
    │   ├── lidar_detection.h
    │   ├── list_creator.h
    │   ├── pillar_utils.h
    ├── launch
    │   ├── static_params.yaml
    │   ├── start_all.launch
    │   ├── start_lidar_detection.launch
    │   └── start_rosbag_play_rviz.launch
    ├── model
    │   ├── lidar_detection.yml
    │   └── FrozenGraphs
    │       └──lidar_detection
    │           └──lidar_detection.pb
    ├── nodelet_plugins.xml
    ├── package.xml
    ├── rviz
    │   └── point_cloud.rviz
    └── src
        ├── definitions.cpp
        ├── detector.cpp
        ├── lidar_detection.cpp
        ├── list_creator.cpp
        └── pillar_utils.cpp
```

- **Key Directories and Files:**
  - `include/`: Header files defining classes and utilities.
  - `launch/`: ROS launch files for starting nodes and configurations.
  - `model/`: Contains the PointPillars model configuration and frozen graph.
  - `rviz/`: RViz configuration files.
  - `src/`: Source code implementing detection algorithms and message handling.

### Understanding the Detection Pipeline

1. **Point Cloud Ingestion:**
   - Raw LiDAR data is received from the `/points2` topic as `sensor_msgs/PointCloud2` messages.

2. **PointPillars Inference:**
   - The PointPillars deep neural network processes the point cloud to identify and classify objects.
   - Outputs include bounding boxes with associated class probabilities.

3. **Bounding Box and Object List Creation:**
   - Detected objects are encapsulated into `IkaObject` messages, detailing their state vectors and classifications.
   - These objects are aggregated into `IkaObjectList` messages for efficient transmission.

4. **Visualization:**
   - Detected objects are visualized in RViz with bounding boxes, color-coded by class type.

*Note:* Detailed understanding of the PointPillars model is beyond the scope of this workshop. Interested participants are encouraged to refer to the original [PointPillars paper](https://arxiv.org/abs/1812.05784) for in-depth knowledge.

### Launching the Detection Node

To initiate the object detection process, execute the combined launch file that starts both the `rosbag play` and the `lidar_detection` nodes:

```bash
roslaunch lidar_detection start_all.launch
```

*Note:* The bag file playback rate is set to `0.1` to accommodate the computational demands of the neural network inference. Depending on your system's performance, you may adjust this rate for optimal processing.

### Visualizing Detected Objects in RViz

Once the detection node is running, configure RViz to display the detected objects:

1. **Add IkaObjectList Display:**
   - Click on **ADD** in RViz.
   - Select **By Topic**.
   - Choose `/lidar_detection/object_list/IkaObjectList`.
   - Click **OK**.

   ![fag1](../images/section_2/object_detection/rviz_select.png)

2. **Resulting Visualization:**
   - Detected objects are displayed with bounding boxes.
   - Objects are color-coded based on their classified type (e.g., cars, pedestrians).

   ![fag1](../images/section_2/object_detection/task.png)

*Issue Encountered:* Initially, bounding boxes may not align accurately, and all objects might be classified as `UNKNOWN`. These issues will be addressed in the subsequent tasks.

## Tasks

### Task 1: Correcting Bounding Box Dimensions and Orientation

**Objective:** Ensure that detected objects have accurate bounding box dimensions and correct heading angles by properly assigning values from the detection model to the `IkaObject` message.

#### Steps:

1. **Navigate to `list_creator.cpp`:**

   Open the `list_creator.cpp` file located in `~/ws/catkin_workspace/src/workshops/section_2/lidar_detection/src/`.

2. **Locate the Code Snippet:**

   Identify the section responsible for setting the object position and dimensions:

   ```cpp
   // set object position
   object.fMean[(int)definitions::ca_model::posX] = bounding_box.center(0);
   object.fMean[(int)definitions::ca_model::posY] = bounding_box.center(1);

   ...  

   // START TASK 1 CODE  

   // set object dimensions and fHeading
   object.fMean[(int)definitions::ca_model::length] = 2;
   object.fMean[(int)definitions::ca_model::width] = 2;
   object.fMean[(int)definitions::ca_model::height] = 2;

   ...
        
   // set yaw angle
   object.fMean[(int)definitions::ca_model::heading] = 0;

   // END TASK 1 CODE 
   ```

3. **Understand the Variables:**

   - **`object`:** Instance of `IkaObject.msg`.
   - **`bounding_box`:** Instance of `BoundingBoxCenter` struct defined in `definitions.h`:

     ```cpp
     struct BoundingBoxCenter
     {
       Eigen::Vector2d center;
       float z;
       float length;
       float width;
       float height;
       float yaw;

       float score;
       int class_idx;
       std::string class_name;
     };
     ```

   - **Issue:** The bounding box dimensions (`length`, `width`, `height`) and the `heading` angle are hardcoded and do not reflect the values from `bounding_box`.

4. **Implement the Corrections:**

   Modify the code between the `// START TASK 1 CODE` and `// END TASK 1 CODE` comments to assign the correct values from `bounding_box`:

   ```cpp
   // START TASK 1 CODE  

   // Set object dimensions and heading using bounding_box data
   object.fMean[(int)definitions::ca_model::length] = bounding_box.length;
   object.fMean[(int)definitions::ca_model::width] = bounding_box.width;
   object.fMean[(int)definitions::ca_model::height] = bounding_box.height;

   // Set yaw angle from bounding_box
   object.fMean[(int)definitions::ca_model::heading] = bounding_box.yaw;

   // END TASK 1 CODE 
   ```

5. **Rebuild the `lidar_detection` Package:**

   After saving the changes, execute the following commands to rebuild the package:

   ```bash
   cd ~/catkin_workspace
   catkin build lidar_detection
   source devel/setup.bash
   ```

6. **Launch the Detection Node:**

   Restart the detection node to apply the changes:

   ```bash
   roslaunch lidar_detection start_all.launch
   ```

7. **Visualize the Updated Bounding Boxes:**

   - If RViz is not already running, execute the launch file again.
   - Observe that the bounding boxes now reflect accurate dimensions and orientations based on the detection model's output.

   *Expected Visualization:*

   ![fag1](../images/section_2/object_detection/result.png)

   *Note:* The bounding boxes should align more accurately with the detected objects, and classes should start to populate correctly.

### Task 2: Assigning Correct Object Classes

**Objective:** Ensure that detected objects are correctly classified by assigning the appropriate `class_idx` based on the highest class probability from the detection model.

#### Steps:

1. **Navigate to `detector.cpp`:**

   Open the `detector.cpp` file located in `~/ws/catkin_workspace/src/workshops/section_2/lidar_detection/src/`.

2. **Locate the Code Snippet:**

   Find the section marked for Task 2, responsible for determining the object's class index:

   ```cpp
   // START TASK 2 CODE
   
   int class_idx = -1;
   
   // END TASK 2 CODE
   ```

3. **Understand the Objective:**

   The goal is to assign `class_idx` to the index of the class with the highest probability score from the detection model's output.

4. **Implement the Class Index Assignment:**

   Replace the placeholder code with logic that identifies the class with the maximum score using `std::max_element`. Here's how to implement it:

   ```cpp
   // START TASK 2 CODE
   
   // Find the index of the class with the highest score
   auto max_it = std::max_element(bounding_box.class_scores.begin(), bounding_box.class_scores.end());
   class_idx = std::distance(bounding_box.class_scores.begin(), max_it);
   
   // END TASK 2 CODE
   ```

   - **Explanation:**
     - `bounding_box.class_scores` is assumed to be a `std::vector<float>` containing the probability scores for each class.
     - `std::max_element` locates the iterator pointing to the highest score.
     - `std::distance` calculates the index of this maximum score, which corresponds to the `class_idx`.

5. **Rebuild the `lidar_detection` Package:**

   After saving the changes, execute the following commands to rebuild the package:

   ```bash
   cd ~/catkin_workspace
   catkin build lidar_detection
   source devel/setup.bash
   ```

6. **Launch the Detection Node:**

   Restart the detection node to apply the changes:

   ```bash
   roslaunch lidar_detection start_all.launch
   ```

7. **Visualize the Correctly Classified Objects:**

   - If RViz is not already running, execute the launch file again.
   - Detected objects should now display their correct classifications (e.g., `CAR`, `PEDESTRIAN`, `TRUCK`, `BICYCLE`) instead of `UNKNOWN`.

   *Expected Visualization:*

   ![fag1](../images/section_2/object_detection/result.png)

   *Observation:* Correct class assignments reduce false positives and enhance the overall detection performance.

## Wrap-up

In this workshop, you have:

- **Explored 3D Object Detection Basics:**
  - Understood how LiDAR data is utilized for object detection in ROS.
  
- **Mastered ROS Object Definitions:**
  - Learned about `IkaObject.msg` and `IkaObjectList.msg` structures.
  
- **Visualized LiDAR Point Clouds:**
  - Utilized RViz to display and interpret raw LiDAR point cloud data.
  
- **Launched and Operated ROS Nodes:**
  - Executed ROS nodes for point cloud playback and object detection.
  
- **Visualized Detected Objects:**
  - Configured RViz to display detected objects with accurate bounding boxes and classifications.
  
- **Enhanced Detection Accuracy:**
  - Corrected bounding box dimensions and orientations.
  - Implemented class index assignment to ensure accurate object classifications.

These skills form the foundation for advanced perception tasks in autonomous systems, enabling vehicles to navigate safely and efficiently by accurately detecting and classifying surrounding objects.

## References

- **PointPillars Paper:** [https://arxiv.org/abs/1812.05784](https://arxiv.org/abs/1812.05784)
  
  The original research paper detailing the PointPillars model used for efficient and accurate object detection in LiDAR point clouds.