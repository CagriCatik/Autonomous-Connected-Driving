# ROS Nodes in C++

![ROS1](https://img.shields.io/badge/ROS1-blue)

This tutorial delves into **implementing your own ROS node in C++**, building upon the foundational knowledge from the [official ROS tutorial on writing a simple publisher and subscriber](https://wiki.ros.org/ROS/Tutorials/WritingPublisherSubscriber%28c%2B%2B%29). By the end of this exercise, you will have a solid understanding of creating, configuring, and managing ROS nodes using C++.

## Learning Objectives

In this exercise, you will learn to:

- Grasp fundamental concepts of C++ relevant to ROS development.
- Set up a ROS node with necessary includes, initialization, and `NodeHandle`.
- Utilize ROS subscribers, publishers, and callback functions effectively.
- Access and manipulate the ROS parameter server.
- Implement ROS logging functionalities.
- Understand the concept and mechanics behind ROS spinning.

## Introduction to C++

A robust understanding of a high-level programming language is essential before diving into this tutorial. While hands-on programming in C++ will be limited to small tasks within this exercise, comprehending the underlying concepts is crucial.

### Prerequisites

- **Basic Programming Knowledge:** Familiarity with programming concepts is necessary.
- **C++ Familiarity:** Although deep expertise in C++ isn't mandatory, understanding its syntax and core principles is beneficial.

If you're new to C++, consider exploring the following resources:

- **External Documentation:** [ACDC Wiki - External Documentation](https://github.com/ika-rwth-aachen/acdc/wiki/External-Documentation)
- **C++ for Python Programmers:** [C++ Tutorial for Python Programmers](https://blue.cs.sonoma.edu/~tiawatts/UsefulStuff/C++ForPythonProgrammers.pdf)

### Key Differences Between C++ and Other Languages

1. **Pointer Variables:**
   - **Declaration:** Use the `*` operator to declare a pointer variable.
     ```cpp
     int *ptr;
     ```
   - **Dereferencing:** Use the `*` operator to access the value at the memory address pointed to by the pointer.
     ```cpp
     int value = *ptr;
     ```
   
2. **Memory Management:**
   - **Dynamic Allocation:** Use `new` to allocate memory and `delete` to deallocate.
     ```cpp
     int *ptr = new int;
     // ... use ptr ...
     delete ptr;
     ```
   - **Modern Alternatives:** Utilize smart pointers like `std::unique_ptr` or `std::shared_ptr` to manage memory automatically.

### Practical C++ Examples from `vehicle_controller_node.cpp`

```cpp
#include "VehicleController.h"
```
- **Purpose:** Includes code from another file, similar to importing a package in Python.

```cpp
/**
 * @brief Callback function that is automatically triggered when a new Lidar scan is available
 * @param msg A pointer to message object that contains the new Lidar scan
 */
```
- **Purpose:** A multi-line comment formatted for Doxygen, enabling automatic generation of code documentation.

```cpp
distances[i] = msg->ranges[i];
```
- **Explanation:**
  - `[]` operator accesses array elements.
  - `->` operator accesses members of a class instance via a pointer.

```cpp
ros::spin();
```
- **Explanation:** The `spin()` function, defined within the `ros` namespace, keeps the node alive and responsive to callbacks.

### Recommendation

For those intending to develop C++ ROS nodes beyond academic exercises, it's highly recommended to learn C++ comprehensively to leverage its full potential within ROS projects.

## Code Elements of a ROS Node in C++

This section examines the ROS-specific components of the vehicle controller node implemented in `~/ws/catkin_workspace/src/workshops/section_1/racing/src/vehicle_controller_node.cpp`. Open this file in your preferred editor and explore the highlighted code snippets to understand their roles and interactions.

### Code Outside Functions

#### Includes

```cpp
#include <ros/ros.h>
#include <sensor_msgs/LaserScan.h>
#include <geometry_msgs/Twist.h>
```

- **Purpose:**
  - `ros/ros.h`: Provides access to common ROS functionalities.
  - `sensor_msgs/LaserScan.h` & `geometry_msgs/Twist.h`: Define message types for Lidar scans and velocity commands, respectively.

#### Publisher and Subscriber Objects

```cpp
ros::Publisher *publisher_actions = nullptr;
ros::Subscriber *subscriber_sensor_data = nullptr;
```

- **Explanation:**
  - Declares pointer variables to `ros::Publisher` and `ros::Subscriber` objects.
  - Initialized to `nullptr`, they will later point to dynamically allocated memory.

#### Vehicle Controller Object

```cpp
VehicleController *vehicle_controller = nullptr;
```

- **Explanation:**
  - Declares a pointer to a `VehicleController` class object.
  - The control algorithm resides within this class, promoting modularity and reusability.

### Callback Function of the Subscriber

```cpp
/**
 * @brief Callback function that is automatically triggered when a new Lidar scan is available
 * @param msg A pointer to message object that contains the new Lidar scan
 */
void callbackLaserSensor(const sensor_msgs::LaserScanPtr &msg) {
  // function body removed
}
```

- **Purpose:**
  - Custom function executed upon receiving new Lidar scan data.
  - Requires correct topic subscription and callback registration to function as intended.

#### Execution of Control Algorithms via VehicleController Interface

```cpp
// Interface calls to the VehicleController instance
vehicle_controller->overwriteLidarDistances(distances);
vehicle_controller->computeTargetValues();
double linear_velocity = vehicle_controller->getTargetVelocity();
double steering_angle = vehicle_controller->getTargetSteeringAngle();
```

- **Explanation:**
  - Interacts with the `VehicleController` object to process sensor data and compute actuator commands.
  - Promotes separation of concerns:
    - **ROS-Specific Code:** Managed in `vehicle_controller_node.cpp`.
    - **Control Algorithm:** Encapsulated within `VehicleController.h` / `VehicleController.cpp`.

#### Conversion of Native C++ Variables to ROS Messages

```cpp
// Convert local variables to a geometry_msgs::Twist message for publishing.
geometry_msgs::Twist new_action;
geometry_msgs::Vector3 steering;
geometry_msgs::Vector3 velocity;
steering.z = steering_angle;
velocity.x = linear_velocity;
new_action.linear = velocity;
new_action.angular = steering;
```

- **Purpose:**
  - Transforms computed velocity and steering values into a `geometry_msgs::Twist` message format for publishing.

#### Publish Commands to Topic

```cpp
// Publish the newly computed actuator command to the topic
publisher_actions->publish(new_action);
```

- **Explanation:**
  - Utilizes the `publisher_actions` object to send the `new_action` message to the designated topic.

### Main Function

#### Initialize ROS Node

```cpp
ros::init(argc, argv, "vehicle_controller");
```

- **Purpose:**
  - Initializes the ROS node named `"vehicle_controller"`.
  - Must be called before any other ROS functionalities.

```cpp
ros::NodeHandle node_handle;
```

- **Explanation:**
  - `NodeHandle` serves as the primary interface for communicating with the ROS system.
  - Manages subscriptions, publications, and parameter interactions.

#### Access the ROS Parameter Server

```cpp
// Declare local variables for subscribe and publish topics
std::string subscribe_topic_sensors;
std::string publish_topic_actuators;

// Write publish and subscribe topics from parameter server into local variables
node_handle.getParam("vehicle/sensor_topic", subscribe_topic_sensors);
node_handle.getParam("vehicle/actuator_topic", publish_topic_actuators);
```

- **Purpose:**
  - Retrieves topic names from the ROS parameter server, allowing dynamic configuration of communication channels.

#### Log Messages

```cpp
ROS_INFO("Vehicle controller subscribes to: %s", subscribe_topic_sensors.c_str());
```

- **Explanation:**
  - Outputs informative log messages to the terminal or `rqt_console`.
  - Facilitates debugging and monitoring by displaying runtime information.

#### Allocate Dynamic Memory

```cpp
// Initialize / allocate dynamic memory
vehicle_controller = new VehicleController;
subscriber_sensor_data = new ros::Subscriber;
publisher_actions = new ros::Publisher;
```

- **Purpose:**
  - Allocates memory for the `VehicleController`, `Subscriber`, and `Publisher` objects.
  - Ensures that these objects are properly instantiated before use.

#### Configure Publisher and Subscriber

```cpp
// Connect subscriber and publisher to their respective topics and callback function
*subscriber_sensor_data = node_handle.subscribe(subscribe_topic_sensors, 10, callbackLaserSensor);
```

- **Explanation:**
  - Subscribes to the sensor topic with a queue size of 10.
  - Registers `callbackLaserSensor` as the callback function for incoming messages.

```cpp
*publisher_actions = node_handle.advertise<geometry_msgs::Twist>(publish_topic_actuators, 10);
```

- **Purpose:**
  - Advertises a publisher on the actuator topic with a message queue size of 10.
  - Enables the node to publish `geometry_msgs::Twist` messages to control actuators.

#### Event Loop

```cpp
// Enter a loop to keep the node running while looking for messages on the subscribed topic [...]
ros::spin();
```

- **Explanation:**
  - Enters an infinite loop, processing callbacks as messages arrive.
  - Ensures the node remains active and responsive until terminated manually (e.g., via <kbd>Ctrl</kbd>+<kbd>C</kbd>).

## Practical Tasks

### Task 1: Adjust the `ros::spin()` Event Loop

**Objective:** Modify the event loop in the `main` function of `vehicle_controller_node.cpp` to process incoming messages every **20ms**.

**Steps:**

1. **Open `vehicle_controller_node.cpp`:**
   - Locate the `ros::spin();` statement within the `main` function.

2. **Replace `ros::spin()` with a Custom Loop:**
   - Implement a `while` loop that calls `ros::spinOnce()` and sleeps for 20ms.

3. **Implement the Loop Using `ros::Rate`:**
   - Utilize `ros::Rate` to maintain a consistent loop rate.

**Sample Implementation:**

```cpp
// Define a loop rate of 50 Hz (20ms)
ros::Rate loop_rate(50);

while (ros::ok()) {
    ros::spinOnce(); // Process incoming messages
    loop_rate.sleep(); // Sleep to maintain loop rate
}
```

**Explanation:**

- **`ros::Rate loop_rate(50);`:** Sets the loop to run at 50 Hz, corresponding to 20ms intervals.
- **`ros::spinOnce();`:** Processes any incoming messages and invokes callbacks.
- **`loop_rate.sleep();`:** Sleeps for the necessary duration to maintain the desired loop rate.

**Benefits:**

- **Control Over Loop Frequency:** Allows for precise timing of message processing.
- **Flexibility:** Enables additional operations within the loop if needed in the future.

### Task 2: Enhance Logging in the Callback Function

**Objective:** Improve the `callbackLaserSensor` function in `vehicle_controller_node.cpp` by adding log messages that indicate when a message is received and display the measured Lidar distances.

**Steps:**

1. **Open `vehicle_controller_node.cpp`:**
   - Navigate to the `callbackLaserSensor` function.

2. **Add Logging Statements:**
   - Insert `ROS_INFO` statements to log message reception and Lidar distances.

**Sample Implementation:**

```cpp
void callbackLaserSensor(const sensor_msgs::LaserScanPtr &msg) {
    ROS_INFO("Received a new Lidar scan.");

    // Assuming there are five Lidar measurements
    ROS_INFO("Lidar Distances:");
    for (size_t i = 0; i < 5; ++i) {
        ROS_INFO("Distance %zu: %.2f meters", i, msg->ranges[i]);
    }

    // Existing processing code...
    distances[i] = msg->ranges[i];
    // ...
}
```

**Explanation:**

- **`ROS_INFO("Received a new Lidar scan.");`:** Logs the reception of a new message.
- **Loop Logging Distances:**
  - Iterates through the first five Lidar measurements.
  - Logs each distance with its index and value in meters.
  
**Notes:**

- **Adjusting the Number of Measurements:**
  - Ensure that the loop does not exceed the actual number of measurements in `msg->ranges`.
  
- **Formatting:**
  - `%.2f` formats the distance to two decimal places for readability.
  - `%zu` is used for `size_t` indices.

### Building the Source Code

After making changes to the source files, remember to:

1. **Save Your Changes:**
   - Ensure all modified files are saved in your editor.

2. **Build the Workspace:**
   - Open a terminal and navigate to your catkin workspace.
   - Execute the build command:
     ```bash
     catkin build
     ```
   - **Note:** Ensure there are no compilation errors before proceeding.

## Wrap-up

In this chapter, you have gained foundational knowledge and practical skills essential for developing ROS nodes in C++. Here's a summary of what you've accomplished:

- **C++ Fundamentals for ROS:**
  - Understood key differences between C++ and other programming languages.
  - Explored practical C++ code snippets relevant to ROS development.

- **Setting Up a ROS Node:**
  - Included necessary headers and initialized the node.
  - Utilized `NodeHandle` for interacting with the ROS system.

- **ROS Publishers and Subscribers:**
  - Configured publishers and subscribers to communicate via ROS topics.
  - Implemented callback functions to handle incoming messages.

- **ROS Parameter Server:**
  - Accessed and utilized parameters from the ROS parameter server for dynamic configuration.

- **Logging Mechanisms:**
  - Enhanced logging within callback functions to facilitate debugging and monitoring.

- **Event Loop Management:**
  - Modified the ROS spinning mechanism to control the frequency of message processing.

By mastering these concepts, you are well-equipped to develop more complex and efficient ROS nodes in C++, catering to both beginner and advanced applications within the ROS ecosystem.