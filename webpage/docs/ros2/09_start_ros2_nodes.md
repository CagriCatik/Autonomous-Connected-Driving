# How to Start ROS2 Nodes

This guide provides a comprehensive understanding of starting ROS2 nodes, verifying their connectivity, and debugging communication. Designed for both beginners and advanced users, this documentation ensures clarity and technical depth for users at all levels.

---

## **Overview**

ROS2 nodes are the fundamental building blocks of a ROS2 application. Each node is a distinct process that performs computation and communicates with other nodes using topics, services, or actions. Starting and managing nodes is a critical skill for any ROS2 developer.

---

## **Starting ROS2 Nodes**

### **Using the Command Line**

1. **Navigate to the Package**: Locate the ROS2 package containing the node you want to run.
2. **Run the Node**:
   ```bash
   ros2 run <package_name> <node_name>
   ```
   Example:
   ```bash
   ros2 run demo_nodes_cpp talker
   ```

### **Using a Launch File**

Launch files simplify running multiple nodes and configuring parameters.
1. **Navigate to the Launch Directory**:
   ```bash
   cd ~/ros2_ws/src/<package_name>/launch
   ```
2. **Execute the Launch File**:
   ```bash
   ros2 launch <package_name> <launch_file_name>.launch.py
   ```
   Example:
   ```bash
   ros2 launch turtlesim turtlesim_node.launch.py
   ```

### **Starting Nodes Programmatically**

For advanced users, nodes can be started programmatically within Python scripts:
```python
import rclpy
from rclpy.node import Node

class SimpleNode(Node):
    def __init__(self):
        super().__init__('simple_node')
        self.get_logger().info('Node has started!')

rclpy.init()
node = SimpleNode()
rclpy.spin(node)
node.destroy_node()
rclpy.shutdown()
```

---

## **Verifying Node Connectivity**

### **List Active Nodes**

Use the following command to verify active nodes:
```bash
ros2 node list
```
Example Output:
```
/talker
/listener
```

### **Inspect Node Information**

View details about a specific node:
```bash
ros2 node info <node_name>
```
Example:
```bash
ros2 node info /talker
```

### **Check Topics**

List all active topics:
```bash
ros2 topic list
```
View the details of a specific topic:
```bash
ros2 topic info <topic_name>
```

### **Test Topic Communication**

Publish and subscribe to topics to verify communication:
1. Publish a test message:
   ```bash
   ros2 topic pub /test_topic std_msgs/String "data: 'Hello, ROS2'"
   ```
2. Subscribe to the topic:
   ```bash
   ros2 topic echo /test_topic
   ```

---

## **Debugging Communication**

### **Using ROS2 CLI**

1. **Debug Node Logs**:
   ```bash
   ros2 run <package_name> <node_name> --ros-args --log-level debug
   ```
2. **Check Parameter Values**:
   ```bash
   ros2 param list <node_name>
   ros2 param get <node_name> <parameter_name>
   ```

### **Using RQT Tools**

Install RQT plugins for debugging:
```bash
sudo apt install ros-<your_ros2_distro>-rqt
sudo apt install ros-<your_ros2_distro>-rqt-graph
```
- **Graph Visualization**:
  ```bash
  rqt_graph
  ```
- **Topic Monitoring**:
  ```bash
  rqt_topic
  ```

### **ROS2 Bag for Debugging**

Record and replay data:
1. Record messages:
   ```bash
   ros2 bag record -a
   ```
2. Replay messages:
   ```bash
   ros2 bag play <bag_file_name>
   ```

### **Common Issues and Solutions**

| Issue                                | Solution                                          |
|-------------------------------------|--------------------------------------------------|
| Node not found                      | Check if the package and node names are correct. |
| Topic communication failing         | Verify topic names and message types.           |
| Parameters not loading              | Ensure parameter file paths are correct.         |
| Unexpected shutdowns                | Inspect node logs for detailed error messages.   |

---

## **Best Practices**

1. **Use Descriptive Names**: Name your nodes and topics clearly to avoid confusion.
2. **Leverage Launch Files**: Use launch files to manage complex configurations.
3. **Automate Tests**: Write automated tests to validate node communication and behavior.
4. **Enable Logging**: Use appropriate logging levels (info, warn, error, debug) for better insights.

---

## **Conclusion**

Understanding how to start and manage ROS2 nodes is essential for efficient robotics development. By mastering the techniques outlined in this guide, you can build robust ROS2 applications and troubleshoot issues effectively.

For further learning, explore the [ROS2 documentation](https://docs.ros.org/en/).

