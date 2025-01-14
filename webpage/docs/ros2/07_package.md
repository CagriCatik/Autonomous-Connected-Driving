# Understanding ROS 2 Packages

ROS 2 (Robot Operating System 2) packages are fundamental units of software organization and deployment in ROS 2 applications. They allow developers to group together nodes, launch files, message and service definitions, and other resources required for robotic applications. This documentation provides a comprehensive guide to understanding the structure of ROS 2 packages and creating your own packages, catering to both beginners and advanced users.

---

## Structure of a ROS 2 Package
A ROS 2 package is a directory with a predefined structure. This structure ensures compatibility with ROS 2 tools such as `colcon` for building and `ros2` CLI commands for interaction. Below is the typical structure of a ROS 2 package:

```
my_package/
├── package.xml
├── setup.py (for Python packages)
├── CMakeLists.txt (for C++ packages)
├── src/
│   └── my_package/
│       └── __init__.py (optional for Python packages)
├── include/
│   └── my_package/ (for C++ header files)
├── msg/
│   └── MyMessage.msg (custom message definitions)
├── srv/
│   └── MyService.srv (custom service definitions)
├── action/
│   └── MyAction.action (custom action definitions)
├── launch/
│   └── my_launch_file.py (launch files)
└── test/
    └── test_my_package.py (test scripts)
```

### Key Components
1. `package.xml`: Describes the package’s metadata, including its name, version, dependencies, and maintainers.
2. `setup.py` / `CMakeLists.txt`: Specifies how the package is built (Python or C++).
3. `src/`: Contains the source code.
4. `include/`: Used for C++ header files.
5. `msg/`, `srv/`, and `action/`: Define custom messages, services, and actions respectively.
6. `launch/`: Stores launch files for starting nodes and setting configurations.
7. `test/`: Contains test cases for unit and integration testing.

---

## Creating a ROS 2 Package

### Step 1: Setting Up the Workspace
Create a new ROS 2 workspace if not already done:
```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws
colcon build
```
Source the workspace:
```bash
source install/setup.bash
```

### Step 2: Creating the Package
Use the `ros2 pkg create` command to scaffold a new package:
#### For Python:
```bash
ros2 pkg create --build-type ament_python my_python_package
```
#### For C++:
```bash
ros2 pkg create --build-type ament_cmake my_cpp_package
```
This command generates a package directory with the necessary files, including `package.xml` and `setup.py` (or `CMakeLists.txt`).

### Step 3: Adding Code

#### Python Node Example
Create a Python node under `src/my_python_package/`:
```python
# src/my_python_package/simple_node.py
import rclpy
from rclpy.node import Node

class SimpleNode(Node):
    def __init__(self):
        super().__init__('simple_node')
        self.get_logger().info('Node has started!')

if __name__ == '__main__':
    rclpy.init()
    node = SimpleNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
```

#### C++ Node Example
Create a C++ node under `src/`:
```cpp
// src/simple_node.cpp
#include "rclcpp/rclcpp.hpp"

class SimpleNode : public rclcpp::Node {
public:
    SimpleNode() : Node("simple_node") {
        RCLCPP_INFO(this->get_logger(), "Node has started!");
    }
};

int main(int argc, char *argv[]) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<SimpleNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
```
Update `CMakeLists.txt` to include the source file:
```cmake
add_executable(simple_node src/simple_node.cpp)
target_link_libraries(simple_node ${rclcpp_LIBRARIES})
install(TARGETS simple_node DESTINATION lib/${PROJECT_NAME})
```

### Step 4: Building the Package
Navigate to the workspace root and build the package:
```bash
cd ~/ros2_ws
colcon build --packages-select my_python_package
```

### Step 5: Running the Node
Source the workspace and run the node:
```bash
source install/setup.bash
ros2 run my_python_package simple_node
```

---

## Best Practices
1. Consistent Naming: Use descriptive and consistent naming for packages and files.
2. Version Control: Use a version control system like Git for managing package changes.
3. Documentation: Provide clear and detailed documentation in `README.md`.
4. Testing: Implement thorough test cases and include them in the `test/` directory.
5. Dependencies: Clearly specify all dependencies in `package.xml`.

---

## Advanced Topics

### Custom Messages and Services
Create custom message and service definitions in `msg/` and `srv/` directories. Update `CMakeLists.txt` or `setup.py` to build them.

#### Example: Custom Message
`msg/MyMessage.msg`:
```plaintext
string data
int32 count
```
Update `package.xml`:
```xml
<build_depend>rosidl_default_generators</build_depend>
<exec_depend>rosidl_default_runtime</exec_depend>
```
Update `CMakeLists.txt`:
```cmake
rosidl_generate_interfaces(${PROJECT_NAME} "msg/MyMessage.msg")
```

### Launch Files
Launch multiple nodes using Python-based launch files. Example:
```python
# launch/my_launch_file.py
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='my_python_package',
            executable='simple_node',
            name='my_node'
        )
    ])
```
Run the launch file:
```bash
ros2 launch my_python_package my_launch_file.py
```

---

## Conclusion
This guide provides a comprehensive overview of ROS 2 packages, from basic structure to advanced usage. By adhering to these principles, you can create well-structured, maintainable, and scalable ROS 2 packages.

