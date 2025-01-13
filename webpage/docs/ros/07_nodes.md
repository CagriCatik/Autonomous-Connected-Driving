# ROS Node

In ROS (Robot Operating System) 1, a **node** is a fundamental building block of the ROS computational graph. Nodes are processes that perform computation and communicate with other nodes in a distributed system. This modularity allows developers to build complex robotic systems by composing multiple nodes, each responsible for a specific task, such as perception, control, or actuation.

## Characteristics of a ROS1 Node
- **Uniqueness**: Each node has a unique name within the ROS graph. Naming conflicts can be avoided using namespaces or remapping.
- **Communication**: Nodes communicate with one another using topics, services, and actions.
- **Distributed**: Nodes can run on different machines as long as they are part of the same ROS master.
- **Lifecycle**: A node's lifecycle is typically managed by the user—launch, runtime, and shutdown are explicit steps in the workflow.

---

## Creating a ROS1 Node

### Prerequisites
Before creating a ROS1 node, ensure the following:
1. ROS1 is installed on your system.
2. A ROS workspace is set up (e.g., `catkin_ws`).
3. Basic understanding of programming in Python or C++.

### Steps to Create a Node

#### 1. Set Up Your Package
Create a package that will house your node:
```bash
cd ~/catkin_ws/src
catkin_create_pkg <package_name> roscpp rospy std_msgs
cd ~/catkin_ws
catkin_make
```
Replace `<package_name>` with your desired package name.

#### 2. Write the Node Code
Nodes can be written in Python or C++. Below is an example for each language.

##### Python Node Example
**File:** `~/catkin_ws/src/<package_name>/scripts/simple_node.py`
```python
#!/usr/bin/env python

import rospy
from std_msgs.msg import String

def talker():
    pub = rospy.Publisher('chatter', String, queue_size=10)
    rospy.init_node('simple_node', anonymous=True)
    rate = rospy.Rate(10)  # 10 Hz

    while not rospy.is_shutdown():
        hello_str = "Hello ROS World %s" % rospy.get_time()
        rospy.loginfo(hello_str)
        pub.publish(hello_str)
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
```
Make the script executable:
```bash
chmod +x ~/catkin_ws/src/<package_name>/scripts/simple_node.py
```

##### C++ Node Example
**File:** `~/catkin_ws/src/<package_name>/src/simple_node.cpp`
```cpp
#include <ros/ros.h>
#include <std_msgs/String.h>

int main(int argc, char **argv) {
    ros::init(argc, argv, "simple_node");
    ros::NodeHandle nh;

    ros::Publisher pub = nh.advertise<std_msgs::String>("chatter", 10);

    ros::Rate rate(10);
    while (ros::ok()) {
        std_msgs::String msg;
        msg.data = "Hello ROS World";

        ROS_INFO("%s", msg.data.c_str());
        pub.publish(msg);

        rate.sleep();
    }
    return 0;
}
```

#### 3. Add Dependencies
Update the `CMakeLists.txt` and `package.xml` files in your package to include required dependencies.

**CMakeLists.txt**
```cmake
add_executable(simple_node src/simple_node.cpp)
target_link_libraries(simple_node ${catkin_LIBRARIES})
add_dependencies(simple_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
```

**package.xml**
```xml
<depend>roscpp</depend>
<depend>std_msgs</depend>
```

#### 4. Build the Node
Build your package using `catkin_make`:
```bash
cd ~/catkin_ws
catkin_make
```

#### 5. Run the Node
Launch the node using `rosrun` or `roslaunch`:
```bash
rosrun <package_name> simple_node
```

---

## Communicating with Other Nodes

### Topics
Topics provide a publisher-subscriber mechanism for nodes to communicate. Use `rostopic` commands to inspect topics:
```bash
rostopic list
rostopic echo /chatter
```

### Services
Services allow synchronous communication between nodes. Define a service in a `.srv` file and generate the required headers.

### Parameters
Nodes can read and write parameters from the ROS parameter server. Example:
```python
my_param = rospy.get_param('/param_name', default_value)
```

---

## Debugging and Monitoring
Use ROS tools to debug and monitor your nodes:
- `rosnode list`: List active nodes.
- `rosnode info <node_name>`: Get information about a node.
- `rqt_graph`: Visualize the node graph.
- `roslog`: Access log files.

---

## Best Practices
- Use meaningful and unique names for your nodes.
- Handle exceptions and errors gracefully.
- Test nodes individually before integration.
- Use `roslaunch` for complex setups.
- Follow ROS coding standards and guidelines.

---

## Conclusion
ROS1 nodes are essential components in building modular, distributed robotic systems. By understanding their lifecycle, communication mechanisms, and best practices, you can effectively design and deploy robust robotic applications.

