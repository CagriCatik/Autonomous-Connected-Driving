# ROS2 Outlook

![ROS1](https://img.shields.io/badge/ROS1-blue)
![ROS2](https://img.shields.io/badge/ROS2-red)


This tutorial provides you with a quick look into ROS2. ROS2 is the successor of ROS1 (we have previously called it ROS for simplicity) and will replace it in future. 
As of 2022, ROS1 is still more popular and many packages like e.g. *flatland*, which is used in this course, are still being ported to ROS2. 

While the transition is in progress, some packages will have been already ported to ROS2, while some others are only available in ROS1. 
For such scenarios, you learn how to connect them to a single system using the  so-called *ROS bridge*.

In this exercise you will
* learn the rough concepts about ROS2, ROS2 launch and config files
* be able to build, source and use ROS1 and ROS2 workspaces, both individual and in parallel
* start the racing application with the simulation running in ROS1 and the vehicle controller in ROS2


## Installation of ROS2

We already installed all necessary dependencies for ROS2 and the ROS bridge inside of the provided docker image. Have a look into the official [ROS2 Installation Tutorial](https://docs.ros.org/en/foxy/Installation/Ubuntu-Install-Debians.html) for further information.


## Build the ROS2 workspace
In this step we will build the racing controller in ROS2 using `colcon`, which replaces the `catkin` build system of ROS1.

:information_source: With the `--symlink-install` parameter, the files of the package such as configurations and launch files are linked to the corresponding `src` directory instead of copied. This allows us to make changes in the source files (config, launch files) without the need to execute `colcon build` again.

```bash
cd ~/ws/colcon_workspace
source /opt/ros/foxy/setup.bash
colcon build --packages-select racing --symlink-install 
```
:information_source: We will not add the new `source` commands to the `.bashrc` script for now, since it may result in problems when both ROS1 and ROS2 are sourced. 

## Start a ROS2 node
In your fist terminal, `source` the colcon workspace.
```bash
source ~/ws/colcon_workspace/install/setup.bash
```

Then, start the controller node using `ros2 launch`, which replaces the ROS1 command `roslaunch`. 
```bash
ros2 launch racing racing_controller.launch.py
```
### Note about ROS2 launch files
The code for the demo can be found in the directory `colcon_workspace/src/section_1/racing`. The structure of this __Cpp package__ is illustrated in the following:
```
racing/
├── CMakeLists.txt
├── config
|   └── racing.yaml
├── include
|   └── VehicleController.h
├── launch
|   └── racing_controller.launch.py
├── package.xml
└── src
    ├── VehicleController.cpp
    ├── vehicle_controller_node.cpp
    └── vehicle_timer_node.cpp
```
ROS2 launch file ([racing_controller.launch.py](https://github.com/ika-rwth-aachen/acdc/blob/ce9b1e8b2e4396a8438aa1a692daf0fcffcf4ed4/colcon_workspace/src/section_1/racing/launch/racing_controller.launch.py)):
- Python script instead of XML file. 
- Mostly describes the node, while the configuration is in a separate file.
- See how the `<node .../>` declaration of ROS1 is similar to `Node(...)` code of ROS2.

```python
# Importing necessary modules.
import os
# 'get_package_share_directory' is used to find a package's shared directory in a ROS2 environment.
from ament_index_python.packages import get_package_share_directory
# 'LaunchDescription' helps describe the nodes to be launched in a ROS2 system.
from launch import LaunchDescription
# 'Node' is an action that represents a node in the ROS2 graph.
from launch_ros.actions import Node

def generate_launch_description():
    # Fetching the shared directory path for the 'racing' package.
    racing_dir = get_package_share_directory('racing')
    # Constructing the path to the configuration file 'params.yaml'.
    config = os.path.join(racing_dir, 'config', 'params.yaml')

    # Defining a node for the vehicle_controller.
    vehicle_controller_node = Node(package='racing', 
                                   executable='vehicle_controller_node',
                                   name='vehicle_controller_node', 
                                   output='screen',
                                   parameters=[config])
    
    # Defining a node for the vehicle_timer.
    vehicle_timer_node = Node(package='racing', 
                              executable='vehicle_timer_node',
                              name='vehicle_timer_node',
                              output='screen',
                              parameters=[config])

    # Creating a LaunchDescription object to store the nodes.
    ld = LaunchDescription()

    # Adding both nodes to the launch description.
    ld.add_action(vehicle_controller_node)
    ld.add_action(vehicle_timer_node)

    # Returning the launch description, which is used by the ROS2 launch system.
    return ld
```

The default configuration ([racing.yaml](https://git.rwth-aachen.de/ika/acdc/-/blob/00eb696faed7b831f77683758a46247d915f5cfe/colcon_workspace/src/section_2/racing/config/racing.yaml)) is held in an external `.yaml` file instead of the launch file itself. This would also be possible in ROS1.

```yaml
vehicle_controller_node:
    ros__parameters:
        vehicle:
            sensor_topic: "/vehicle/lidar_measurements"
            actuator_topic: "/vehicle/actuator_commands"

vehicle_timer_node:
    ros__parameters:
        vehicle:
            position_topic: "/odometry/ground_truth"

```

## Start a ROS1 node
In your second terminal, start the ROS1 flatland simulation:

```bash
cd ~/ws/catkin_workspace
source /opt/ros/noetic/setup.bash
source ~/ws/catkin_workspace/devel/setup.bash # this may be already included in your .bashrc
roslaunch racing flatland_simulation.launch
```


## Start the ROS bridge
Now the racing controller is started in ROS2 and the flatland simulation in ROS1. We can use the bridge to exchange data between ROS1 and ROS2 in both ways. 

The ROS2 racing controller subscribes to the sensor data from the ROS1 flatland server over the ROS bridge, computes actuator commands and publishes them back to the ROS1 flatland server over the ROS bridge. 

In your third terminal, source both the ROS1 and the ROS2 version. 

```bash
source /opt/ros/foxy/setup.bash # source ros2 version
source /opt/ros/noetic/setup.bash # source ros1 version
```

Now, start the bridge using `ros2 run`, which replaces the ROS1 command `rosrun`. 
```bash
ros2 run ros1_bridge dynamic_bridge
```
After this, the data of the ROS1 node are linked over the `ros1_bridge` and the vehicle should start moving.


## Wrap-up
* You learned that both ROS versions can be used in parallel within one docker environment.
* You learned that we can use the ROS bridge to transfer information between ROS1 and ROS2 nodes.
