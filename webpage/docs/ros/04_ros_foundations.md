# ROS Foundations

The Robot Operating System (ROS) is an open-source framework designed for developing robotic software. ROS1 (the original version of ROS) provides tools, libraries, and conventions to simplify the task of creating complex robot behavior across various robotic platforms.

### Core Concepts
ROS1 is built around several key concepts that enable modular and distributed robotics software development:

1. **Nodes**: A node is a process that performs computation. In ROS1, nodes communicate with each other using messages.

2. **Master**: The ROS Master provides name registration and lookup to the nodes in the ROS network. It allows nodes to find each other and establish communication.

3. **Topics**: Topics are named buses over which nodes exchange messages. Publishers send messages to a topic, and subscribers receive messages from a topic.

4. **Messages**: Messages are data structures used to exchange information between nodes. These are predefined or custom formats defined in `.msg` files.

5. **Services**: Services allow synchronous communication between nodes. A service is defined by a pair of request and response messages.

6. **Parameters**: Parameters are configuration values that can be set at runtime and accessed by nodes. They are managed by the parameter server.

7. **ActionLib**: The ActionLib library is used for long-running tasks that require feedback, such as navigation or manipulation.

8. **Bags**: Bag files are used to record and playback ROS topics for debugging and analysis.

---

## Setting Up ROS1

### Prerequisites
- A compatible operating system (ROS1 is primarily supported on Ubuntu-based distributions).
- Basic knowledge of Linux commands and shell scripting.

### Installation
1. **Add the ROS Repository**:
   ```bash
   sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
   sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
   ```

2. **Update the Package Index**:
   ```bash
   sudo apt update
   ```

3. **Install ROS1 (e.g., ROS Melodic)**:
   ```bash
   sudo apt install ros-melodic-desktop-full
   ```

4. **Initialize rosdep**:
   ```bash
   sudo rosdep init
   rosdep update
   ```

5. **Set Up the Environment**:
   ```bash
   echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc
   source ~/.bashrc
   ```

6. **Install Additional Dependencies**:
   ```bash
   sudo apt install python-rosinstall python-rosinstall-generator python-wstool build-essential
   ```

---

## Creating and Managing Workspaces

### What is a ROS Workspace?
A ROS workspace is a directory where you can build and modify ROS packages. The workspace follows a specific structure:
- `src/`: Contains source code for ROS packages.
- `devel/`: Contains built files and setup scripts for the workspace.
- `build/`: Contains intermediate build files.

### Steps to Create a Workspace
1. **Create a Directory for the Workspace**:
   ```bash
   mkdir -p ~/catkin_ws/src
   cd ~/catkin_ws/
   ```

2. **Initialize the Workspace**:
   ```bash
   catkin_make
   ```

3. **Source the Workspace**:
   ```bash
   source devel/setup.bash
   ```

4. **Add Packages to the Workspace**:
   Clone or create ROS packages inside the `src/` directory.

5. **Build the Workspace**:
   ```bash
   catkin_make
   ```

---

## Writing Your First ROS1 Node

### Node Structure
A simple ROS1 node typically includes:
1. **Initialization**: Initialize the ROS node with a name.
2. **Topic Subscription/Publication**: Define publishers and/or subscribers.
3. **Callback Functions**: Handle incoming messages.
4. **Main Loop**: Keep the node alive.

### Example: Writing a Publisher Node (Python)

#### File: `talker.py`
```python
#!/usr/bin/env python
import rospy
from std_msgs.msg import String

def talker():
    rospy.init_node('talker', anonymous=True)
    pub = rospy.Publisher('chatter', String, queue_size=10)
    rate = rospy.Rate(10) # 10 Hz

    while not rospy.is_shutdown():
        hello_str = "Hello ROS! %s" % rospy.get_time()
        rospy.loginfo(hello_str)
        pub.publish(hello_str)
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
```

#### Steps to Run:
1. Make the file executable:
   ```bash
   chmod +x talker.py
   ```

2. Run the ROS Master:
   ```bash
   roscore
   ```

3. Run the Node:
   ```bash
   rosrun <package_name> talker.py
   ```

---

## ROS1 Tools

### Commonly Used Tools
1. **rqt_graph**: Visualize the ROS computation graph.
   ```bash
   rqt_graph
   ```

2. **rosnode**: Manage and debug nodes.
   - List nodes:
     ```bash
     rosnode list
     ```
   - Get information about a node:
     ```bash
     rosnode info <node_name>
     ```

3. **rostopic**: Inspect topics.
   - List topics:
     ```bash
     rostopic list
     ```
   - Echo topic messages:
     ```bash
     rostopic echo <topic_name>
     ```

4. **rviz**: Visualize robot state and sensor data.
   ```bash
   rviz
   ```

5. **rosbag**: Record and play back topics.
   - Record:
     ```bash
     rosbag record -a
     ```
   - Play:
     ```bash
     rosbag play <bag_file>
     ```

---

## Advanced Topics

### Custom Messages
1. Create a `msg` file in the `msg/` directory of a package.
2. Define the message fields.
3. Modify `CMakeLists.txt` and `package.xml` to include the message.
4. Build the package:
   ```bash
   catkin_make
   ```

### Launch Files
Launch files are XML files that start multiple nodes with a single command. Example:

#### File: `example.launch`
```xml
<launch>
    <node pkg="turtlesim" type="turtlesim_node" name="sim"/>
    <node pkg="turtle_teleop" type="turtle_teleop_key" name="teleop"/>
</launch>
```

Run the launch file:
```bash
roslaunch <package_name> example.launch
```

---

## Conclusion
This document provides a foundational understanding of ROS1, enabling you to set up and develop basic nodes and applications. For further exploration, delve into advanced concepts like navigation, manipulation, and multi-robot systems.

