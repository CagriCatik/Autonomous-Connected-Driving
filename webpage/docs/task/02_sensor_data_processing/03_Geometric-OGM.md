# Geometric Point Cloud Occupancy Grid Mapping

![ROS1](https://img.shields.io/badge/ROS1-blue)

You will now get some hands-on experience with inverse sensor models that create occupancy grid maps from lidar point clouds. In this task, you will complete an algorithm for a **geometric inverse sensor model** and in the next tasks you will train and use a deep learning-based model.

You will

- start multiple ROS nodes using a launch file to run and visualize a geometric inverse sensor model using recorded lidar sensor data from a Rosbag,
- learn how to use an existing ROS package to separate ground from obstacle points in the lidar point cloud and
- implement a simple geometric inverse sensor model.

## Task 1: Set up and test the ROS workspace

We will use the same ROS environment and sensor data that was already used in the previous object detection task. So, make sure to have the **ROS framework** set up and the **Rosbag** containing recorded lidar data at `acdc/bag/lidar_campus_melaten.bag`. If you skipped the object detection task, download the Rosbag [here](https://rwth-aachen.sciebo.de/s/udlMYloXpCdVtyp).

If not running yet, start the Docker container by executing the `docker/run.sh` script. Open a shell in the container by executing the script again from another terminal. In the container, build the ROS package and source the ROS workspace:

```bash
catkin build pointcloud_ogm
source devel/setup.bash
```

Start a roscore in the background and start playing the recorded data:

```bash
roscore&
rosbag play --loop ../bag/lidar_campus_melaten.bag
```

You will see this output in the terminal:

```bash
rosuser@******:~/ws/catkin_workspace$ roscore&
[1] 14791
rosuser@******:~/ws/catkin_workspace$ ... logging to /home/rosuser/.ros/log/52caca3c-4495-11ec-82b7-b49691b9ac50/roslaunch-I2102656-linux-14791.log
Checking log directory for disk usage. This may take a while.
Press Ctrl-C to interrupt
Done checking log file disk usage. Usage is <1GB.

started roslaunch server http://******:34069/
ros_comm version 1.15.11
...
started core service [/rosout]
rosbag play --loop ../bag/lidar_campus_melaten.bag
[ INFO] [1636816879.584949638]: Opening ../bag/lidar_campus_melaten.bag

Waiting 0.2 seconds after advertising topics... done.

Hit space to toggle paused, or 's' to step.
 [RUNNING]  Bag Time: 1580916332.230592   Duration: 0.820823 / 119.955245
```

Open another shell in the running container by executing the `docker/run.sh` script again from another terminal window. In the container, source the workspace and execute the launch file that starts the geometric inverse sensor model:

```bash
source devel/setup.bash
roslaunch pointcloud_ogm GeometricISM.launch
```

Rviz will open and visualize lidar point clouds. The points in the **originally recorded point cloud** are colored by the intensity of the reflection. A second **obstacle point cloud** is colored in purple. You can (de-)activate the visualization of the point clouds by (un-)ticking them in the left menu.


<img src="../images/rviz_points2.PNG" alt="Description of image" />

## Task 2: Separate obstacles from ground

The recorded lidar point clouds are published as ROS messages of type [sensor_msgs/PointCloud2](http://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/PointCloud2.html). Each message contains a set of points where each point is described by multiple fields. In our case each point has the fields `x`, `y` and `z` for the position as well as fields describing the `intensity` of reflection and number of the vertical sensor `ring` that detected the reflection.

<img src="../images/rqt_points2_fields.PNG" alt="Description of image" />


At first step of the geometric algorithm, we have to **extract all ground points from the lidar point cloud**. For that, we can use the `PassThrough` filter that is [provided by the PointCloudLibrary](https://wiki.ros.org/pcl_ros/Tutorials/filters#PassThrough). It allows setting parameters to determine which points should be kept in the point cloud. The filter is already launched by the file `workshops/section_2/pointcloud_ogm/launch/GeometricISM.launch`:

```xml
  <node pkg="nodelet" type="nodelet" name="GroundExtraction" args="load pcl/PassThrough $(arg nodelet_manager)" output="screen">
    <remap from="~input" to="/points2" />
    <remap from="~output" to="/points2_obstacles" />
    <rosparam>
      filter_limit_negative: False
      filter_field_name: x
      filter_limit_min: -50
      filter_limit_max: 50
    </rosparam>
  </node>
```

The parameters are already set according to the [specification in the ROS wiki](https://wiki.ros.org/pcl_ros/Tutorials/filters#PassThrough), but they do not fulfill the task. Your task is now to find a configuration that extracts most ground points from the point cloud but (as far as possible) no obstacle points. After changing the parameters, kill the running launch file by pressing `CTRL-C` in the terminal and restart it to see the effects.

## Task 3: Complete the geometric ISM

The ROS nodelet `pointcloud_ogm/GeometricISM` receives lidar point cloud and shall create occupancy grid maps using a geometric inverse sensor model and a binary Bayes filter. You can find the code in the file `workshops/section_2/pointcloud_ogm/src/GeometricISM.src`. The following steps are performed by the code:

1. A point cloud is received and the `messageCallback()` method is executed.
   1. A new grid map is initialized with occupancy probability of all cells set to `0.5` (or `50%`).
   2. All points in the point cloud are evaluated sequentially
      1. A line is formed from each point to the sensor.
      2. The inverse sensor model determines occupancy probabilities of all grid cells on this line.
      3. A binary Bayes filter is used to combine the occupancy probabilities derived from the inverse sensor model with the previous state of the grid map.
   3. The complete measurement grid map is published and can be visualized in Rviz.

The following code iterates over all cells on the line between a reflection point and the sensor. The current occupancy probability of the cell can be accessed with the variable `occupancy_probability`. A binary Bayes filter combines this value with the occupancy probability derived by the inverse sensor model `p_ism`. The developer has already added comment describing the desired behavior but you will have to **add the code for this simple inverse sensor model** in the marked section.

```cpp
    int cell = 0;
    for (grid_map::LineIterator iterator(grid_map_measurement, start_position, end_position); !iterator.isPastEnd(); ++iterator) {
      auto& occupancy_probability = grid_map_measurement.at("occupancy_probability", *iterator);

      /* inverse sensor model:
          - cell containing reflection point: 90% occupancy probability
          - next two cells towards sensor: 80%, 50% occupancy probability
          - remaining cells towards sensor: 10% occupancy probability
      */
      double p_ism;
      // TASK 2 BEGIN
      // ADD YOUR CODE HERE...

      // TASK 2 END
      
      // combine probability from ism with previous probability in cell using binary Bayes filter
      occupancy_probability = (p_ism*occupancy_probability) /
                              (p_ism*occupancy_probability + (1-p_ism)*(1-occupancy_probability));

      cell++;
    }
```

Once you have added your code, compile the ros workspace and restart the launch file. Enable the checkbox next to "Grid Map (GeometricISM)" in the Rviz window to see the published occupancy grid maps.

```bash
catkin build pointcloud_ogm
roslaunch pointcloud_ogm GeometricISM.launch
```

## Wrap-up

- You have learned how to **use a filter from the PointCloudLibrary** to preprocess point clouds.
- You wrote an **algorithm for a simple geometric inverse sensor model** to create occupancy grid maps from point clouds.
- You should notice that this approach has some noticeable **deficiancies**.
- In the next Python task, you will train a neural network to perform occupancy grid mapping from lidar point clouds
