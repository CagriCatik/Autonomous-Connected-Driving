# Creating Your First ROS2 Subscriber Node

In ROS 2 (Robot Operating System 2), the Subscriber node is an essential component used to receive and process messages published by a Publisher node. This documentation provides a detailed guide for setting up and implementing a ROS 2 Subscriber node, catering to both beginners and advanced users. By the end of this guide, you will understand how to create a Subscriber node, process incoming messages, and integrate it into a larger ROS 2 application.

---

## Prerequisites

Before implementing a Subscriber node, ensure you have:
- ROS 2 installed on your system (e.g., Humble, Galactic).
- Basic understanding of ROS 2 concepts (nodes, topics, messages).
- A functioning Publisher node to provide data.
- Python or C++ programming experience, depending on your implementation language of choice.

---

## Step 1: Create a ROS 2 Package

Use the following command to create a new ROS 2 package if you don’t already have one:

```bash
ros2 pkg create --build-type ament_python my_package
```

Replace `my_package` with the desired package name.

### Add Dependencies
Update the `package.xml` and `setup.py` files to include required dependencies. Add `rclpy` for Python or `rclcpp` for C++ as needed.

---

## Step 2: Define the Subscriber Node

### Python Implementation

1. Create the Node File: Navigate to the `my_package` directory and create a new Python file, e.g., `subscriber_node.py`.

2. Write the Subscriber Code:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String  # Replace with your custom message type if needed

class MinimalSubscriber(Node):

    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            String,
            'topic_name',  # Replace with your topic name
            self.listener_callback,
            10
        )
        self.subscription  # Prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info(f'Received message: {msg.data}')


def main(args=None):
    rclpy.init(args=args)

    subscriber = MinimalSubscriber()

    rclpy.spin(subscriber)

    subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

3. Make the File Executable:

```bash
chmod +x subscriber_node.py
```

### C++ Implementation

1. Create the Node File: Navigate to the `src` directory of your package and create a new C++ file, e.g., `subscriber_node.cpp`.

2. Write the Subscriber Code:

```cpp
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"  // Replace with your custom message type if needed

class MinimalSubscriber : public rclcpp::Node {
public:
    MinimalSubscriber()
    : Node("minimal_subscriber") {
        subscription_ = this->create_subscription<std_msgs::msg::String>(
            "topic_name", 10,
            std::bind(&MinimalSubscriber::listener_callback, this, std::placeholders::_1)
        );
    }

private:
    void listener_callback(const std_msgs::msg::String::SharedPtr msg) {
        RCLCPP_INFO(this->get_logger(), "Received message: '%s'", msg->data.c_str());
    }

    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr subscription_;
};

int main(int argc, char *argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MinimalSubscriber>());
    rclcpp::shutdown();
    return 0;
}
```

3. Update CMakeLists.txt: Add the new node to your build system.

```cmake
add_executable(subscriber_node src/subscriber_node.cpp)
target_link_libraries(subscriber_node ${rclcpp_LIBRARIES})
install(TARGETS
    subscriber_node
    DESTINATION lib/${PROJECT_NAME})
```

---

## Step 3: Build and Run the Subscriber Node

1. Build the Package:

```bash
colcon build --packages-select my_package
```

2. Source the Workspace:

```bash
source install/setup.bash
```

3. Run the Subscriber Node:

For Python:

```bash
ros2 run my_package subscriber_node
```

For C++:

```bash
ros2 run my_package subscriber_node
```

---

## Debugging Tips

- Ensure the topic name matches between the Publisher and Subscriber.
- Use `ros2 topic list` to verify available topics.
- Use `ros2 topic echo <topic_name>` to inspect topic data.

---

## Advanced Features

### Quality of Service (QoS)
Customize the QoS settings to control message delivery reliability, latency, and durability:

```python
from rclpy.qos import QoSProfile
qos_profile = QoSProfile(depth=10)
```

### Using Custom Messages
Define your custom message type and include it in your Subscriber. Ensure the `msg` file is in your package and built properly.

---

## Conclusion
This guide has demonstrated how to set up and run a Subscriber node in ROS 2 using Python or C++. For more advanced applications, consider using QoS policies, custom message types, or integrating the node into a complex system. Use debugging tools like `ros2 topic list` and `ros2 topic echo` to verify node functionality.

