# CMakeLists.txt in ROS2

In ROS2, `CMakeLists.txt` plays a pivotal role in defining the build configuration of your package. It is a CMake script used by the `ament` build system to compile and link your code, manage dependencies, and define installation rules. A well-configured `CMakeLists.txt` ensures smooth development and deployment of your ROS2 package.

This documentation provides an in-depth guide to understanding, creating, and configuring `CMakeLists.txt` for ROS2, catering to both beginners and advanced users.

---

## Basics of `CMakeLists.txt`

1. Purpose:
   - Specify the minimum CMake version.
   - Declare the package name and dependencies.
   - Define targets (e.g., executables, libraries).
   - Configure installation paths.
   - Set up testing and documentation generation.

2. Structure:
   A typical `CMakeLists.txt` file consists of the following sections:
   - Project setup.
   - Dependencies and packages.
   - Build targets.
   - Installation instructions.
   - Testing configurations (optional).

---

## Step-by-Step Configuration

### 1. Setting Up the Project
Start by specifying the minimum required CMake version and project name:

```cmake
cmake_minimum_required(VERSION 3.5)
project(my_ros2_package)
```

### 2. Finding Dependencies
Use `find_package` to locate dependencies. For example:

```cmake
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
```

### 3. Adding Targets
Define targets such as executables or libraries.

For an executable:

```cmake
add_executable(my_node src/my_node.cpp)
ament_target_dependencies(my_node rclcpp sensor_msgs)
```

For a library:

```cmake
add_library(my_library src/my_library.cpp)
ament_target_dependencies(my_library rclcpp)
```

### 4. Installing Targets
Specify installation rules to ensure the built binaries are placed in the appropriate location:

```cmake
install(TARGETS my_node
  DESTINATION lib/${PROJECT_NAME})
```

### 5. Exporting Dependencies
Export dependencies so that other packages using this package can link against them:

```cmake
ament_export_dependencies(rclcpp sensor_msgs)
ament_package()
```

### 6. Optional: Adding Tests
Configure tests using `ament_add_gtest`:

```cmake
ament_add_gtest(test_my_node test/test_my_node.cpp)
if(TARGET test_my_node)
  target_link_libraries(test_my_node my_library)
endif()
```

---

## Full Example
Below is a complete example of `CMakeLists.txt` for a simple ROS2 package:

```cmake
cmake_minimum_required(VERSION 3.5)
project(my_ros2_package)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)

# Add executable
add_executable(my_node src/my_node.cpp)
ament_target_dependencies(my_node rclcpp sensor_msgs)

# Install executable
install(TARGETS my_node
  DESTINATION lib/${PROJECT_NAME})

# Export dependencies
ament_export_dependencies(rclcpp sensor_msgs)
ament_package()
```

---

## Advanced Topics

### 1. Custom Build Options
Add custom options to enable or disable features at build time:

```cmake
option(USE_SIMULATION "Enable simulation mode" ON)
if(USE_SIMULATION)
  add_definitions(-DUSE_SIMULATION)
endif()
```

### 2. Using Python Nodes
For Python-based nodes, include:

```cmake
ament_python_install_package(${PROJECT_NAME})
install(PROGRAMS scripts/my_python_node.py
  DESTINATION lib/${PROJECT_NAME})
```

### 3. Generating Messages and Services
To include custom message or service files:

```cmake
find_package(rosidl_default_generators REQUIRED)
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/MyMessage.msg"
  "srv/MyService.srv"
)
ament_export_dependencies(rosidl_default_runtime)
```

---

## Best Practices

- Consistency: Follow a consistent structure for all `CMakeLists.txt` files in your project.
- Comments: Add comments to explain each section, especially for complex configurations.
- Validation: Regularly test your `CMakeLists.txt` by building the package from scratch.
- Avoid Hardcoding: Use variables like `${PROJECT_NAME}` and `${CMAKE_INSTALL_PREFIX}` to avoid hardcoding paths.

---

## Common Errors and Debugging

1. Missing Dependencies:
   Ensure all required dependencies are included using `find_package` and `ament_target_dependencies`.

2. Incorrect Install Paths:
   Verify the `install` directives and ensure they match the expected directory structure.

3. Unresolved Symbols:
   Check if all linked libraries and exported dependencies are correctly configured.

---

## Conclusion
The `CMakeLists.txt` file is the backbone of your ROS2 package build process. A well-structured and optimized file not only simplifies development but also ensures compatibility and scalability. By following this guide, you can create robust `CMakeLists.txt` configurations tailored to your project needs.

For further details, refer to the [ROS2 Documentation](https://docs.ros.org/en/rolling/index.html) and the [CMake Documentation](https://cmake.org/documentation/).

