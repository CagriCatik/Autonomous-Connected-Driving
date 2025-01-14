# Cloud-Based Object Fusion

![ROS1](https://img.shields.io/badge/ROS1-blue)

This documentation provides a comprehensive guide to receiving and processing V2X (Vehicle-to-Everything) messages using ROS1 and MQTT. Whether you're a beginner or an advanced user, this guide will help you navigate through configuring MQTT bridges, visualizing data in RVIZ, and implementing object fusion algorithms. In this workshop, you will learn to receive V2X messages and process them effectively using ROS1 and MQTT protocols. By adapting existing ROS nodes, you'll handle received messages to implement two primary use cases:

- **Cloud-Based Object Fusion** (Section 3)
- **SPaT/MAP Processing for Trajectory Planning** (Section 4)

Both use cases leverage the MQTT protocol and the Mosquitto broker for message transmission. This guide focuses on the **Cloud-Based Object Fusion** use case, guiding you through configuring MQTT bridges, visualizing object lists, and applying object fusion algorithms.

---

## Prerequisites

Before diving into the workshop, ensure you have the following:

- **ROS1 Installed**: Make sure ROS1 is correctly installed and configured on your system.
- **Catkin Workspace**: Familiarity with Catkin workspaces and basic ROS package management.
- **Docker Image**: Access to the provided Docker image containing necessary ROS packages, including `mqtt_client`.
- **Completed Section 3 Tasks**: Ensure you've completed tasks from Section 3 or pulled the corresponding solution.

---

## Overview

The **Cloud-Based Object Fusion** task aims to merge object lists from two infrastructure sensors located at the intersection of Campus-Boulevard and Forckenbeckstraße. These sensors, Station A and Station B, publish their detected objects as `ikaObjectList` messages via MQTT to a broker. Using the `mqtt_client` ROS package, you'll bridge MQTT messages to ROS topics, visualize them in RVIZ, and apply an object fusion algorithm to consolidate the data.

---

## Task 1: Configure the MQTT-ROS Bridge

The first task involves setting up a bridge between MQTT and ROS to receive object lists from the infrastructure sensors.

### Building the Workspace

Begin by building your Catkin workspace to ensure all packages are correctly compiled.

```bash
catkin build
source devel/setup.bash  # Execute this line in each new terminal session
```

**Explanation:**

- `catkin build`: Compiles all packages in the workspace.
- `source devel/setup.bash`: Sets up the environment variables for the workspace. This command must be run in every new terminal where you intend to use ROS commands.

### Creating the Parameter File

The `mqtt_launchpack` package includes an example launch file with a demo configuration located at `section_5/mqtt_launchpack/launch/example.launch`. Your task is to create a new parameter file tailored to receive object lists from Station A and Station B.

**Steps:**

1. **Locate the Example Parameter File:**

   Use the example configuration as a reference:
   ```
   section_5/mqtt_launchpack/config/example_params.yaml
   ```

2. **Create a New Parameter File:**

   Create a new YAML file, e.g., `cloud_object_fusion_params.yaml`, and populate it with the following content:

   ```yaml
   bridge_type: mqtt2ros  # Ensures the bridge converts MQTT messages to ROS

   mqtt:
     broker_address: broker.hivemq.com
     broker_port: 1883
     keepalive: 60
     # No credentials required for this broker

   topics:
     subscribe:
       - topic: ika_acdc_22/objectList_a
         type: ika_msgs/IkaObjectList
       - topic: ika_acdc_22/objectList_b
         type: ika_msgs/IkaObjectList
     publish:
       # No topics to publish in this configuration
   ```

   **Explanation:**

   - **bridge_type**: Specifies the direction of data flow (`mqtt2ros` converts MQTT messages to ROS topics).
   - **mqtt**: Contains connection details for the MQTT broker.
     - `broker_address`: The address of the MQTT broker.
     - `broker_port`: The port number (default for HiveMQ is `1883`).
     - `keepalive`: Interval in seconds for the keepalive ping.
   - **topics**:
     - `subscribe`: Lists the MQTT topics to subscribe to, along with their message types.
     - `publish`: Defines ROS topics to publish MQTT messages if needed (not required here).

### Launching the MQTT Client

With the parameter file ready, launch the MQTT client to start receiving messages.

1. **Modify the Launch File:**

   Create a new launch file or modify the existing one to use your parameter file. For example, create `cloud_object_fusion.launch`:

   ```xml
   <launch>
     <arg name="load_params" default="true" />
     <arg name="params_file" default="$(find mqtt_launchpack)/config/cloud_object_fusion_params.yaml" />

     <node pkg="nodelet" type="nodelet" name="mqtt_client" args="standalone mqtt_client/MqttClient" output="screen">
       <rosparam command="delete" param="" if="$(arg load_params)" />
       <rosparam command="load" file="$(arg params_file)" if="$(arg load_params)" />
     </node>
   </launch>
   ```

2. **Launch the MQTT Client:**

   Execute the launch file:

   ```bash
   roslaunch mqtt_launchpack cloud_object_fusion.launch
   ```

   **Expected Output:**

   ```
   [INFO] [timestamp]: Connected to broker at 'tcp://broker.hivemq.com:1883'
   ```

3. **Verify Message Reception:**

   Open `rqt` to monitor ROS topics:

   ```bash
   rqt
   ```

   Navigate to the **Topics** section to ensure messages are arriving on the subscribed topics (`ika_acdc_22/objectList_a` and `ika_acdc_22/objectList_b`). You should see object lists with six objects each.

   ![Object Lists in rqt](../images/rqt_object_list.png)

---

## Task 2: Visualize the Received Object Lists

Visualization is crucial for verifying that the object lists are correctly received and processed. This task guides you through using RVIZ to visualize the object data.

### Starting RVIZ

An RVIZ configuration has been prepared for this task. Launch it using the following command:

```bash
roslaunch acdc_launchpack cloud_processing_vizu.launch
```

### Configuring RVIZ Topics

To visualize the object lists:

1. **Open RVIZ Panels:**

   In the RVIZ interface, locate the **Displays** panel on the left side.

2. **Add or Modify Topics:**

   - **Option 1: Directly in RVIZ**
     - Add a new display type that matches your data (e.g., `Marker`, `PointCloud2`).
     - Set the **Topic** field to your subscribed ROS topics (`ika_acdc_22/objectList_a` and `ika_acdc_22/objectList_b`).

   - **Option 2: Edit RVIZ Configuration File**
     - Locate the RVIZ configuration file:
       ```
       dependencies/acdc_launchpack/rviz/cloud_processing.rviz
       ```
     - Open the file in a text editor and replace `/topicA` and `/topicB` with your actual ROS topics:
       ```yaml
       Display:
         - Name: ObjectListA
           Topic: /ika_acdc_22/objectList_a
         - Name: ObjectListB
           Topic: /ika_acdc_22/objectList_b
       ```

3. **Verify Visualization:**

   If configured correctly, RVIZ should display the object lists as shown below:

   ![Cloud Processing RVIZ](../images/cloud_processing_rviz.gif)

   Each topic should display six objects corresponding to the data from Station A and Station B.

---

## Task 3: Fuse the Object Lists

The final task involves fusing the object lists from both infrastructure sensors using the object fusion algorithm developed in Section 3.

### Modifying the Fusion Configuration

To integrate the object fusion algorithm with your received object lists:

1. **Locate the Configuration File:**

   ```
   workshops/section_3/object_fusion_wrapper/param/config_inout.yaml
   ```

2. **Edit the Configuration:**

   Update the `input_topics` and `output_topics` sections to match your setup:

   ```yaml
   input_topics:
     object_lists: [
       /ika_acdc_22/objectList_a,
       /ika_acdc_22/objectList_b
     ]

     # Remove ego_motion as there is no ego vehicle
     # ego_motion: /sensors/vehicleCAN/ikaEgoMotion

   output_topics:
     object_list_fused:
       publish: true
       topic: /sensors/fusion/ikaObjectList
       frame_id: map  # Use 'map' frame instead of 'base_link'
   ```

   **Explanation:**

   - **input_topics.object_lists**: Updated to your subscribed topics from Station A and B.
   - **ego_motion**: Removed since there's no ego vehicle in this use case.
   - **output_topics.object_list_fused.frame_id**: Changed to `map` to publish the fused list in the global coordinate system.

3. **Save the Configuration:**

   Ensure the modified `config_inout.yaml` is saved correctly.

### Launching the Object Fusion Node

With the configuration updated, proceed to launch the object fusion node.

1. **Build the Workspace:**

   Rebuild the workspace to incorporate configuration changes:

   ```bash
   catkin build
   ```

2. **Launch the Fusion Node:**

   Start the object fusion process:

   ```bash
   roslaunch object_fusion_wrapper fusion.launch
   ```

   **Expected Output:**

   ```
   [INFO] [timestamp]: Object fusion node started, publishing to '/sensors/fusion/ikaObjectList'
   ```

### Visualizing Fused Data

To verify that object fusion is functioning correctly:

1. **Add Fused Object List to RVIZ:**

   - In the RVIZ **Displays** panel, add a new display for the fused object list.
   - Set the **Topic** to your chosen output topic (`/sensors/fusion/ikaObjectList`).

2. **Update RVIZ Configuration:**

   Alternatively, edit the RVIZ configuration file to include the fused topic:

   ```yaml
   Display:
     - Name: FusedObjectList
       Topic: /sensors/fusion/ikaObjectList
       Color: Green  # Fused objects displayed in green
   ```

3. **Observe the Visualization:**

   The fused object list should now appear in RVIZ, highlighted in green:

   ![Cloud Processing Fusion RVIZ](../images/cloud_processing_fusion_rviz.gif)

   You may also hide the individual object lists by unchecking their respective checkboxes in RVIZ to focus solely on the fused data.

---

## Wrap-up

In this workshop, you have successfully:

- **Received MQTT Messages in ROS**: Configured a bridge to subscribe to MQTT topics and convert them into ROS messages.
- **Visualized Object Lists in RVIZ**: Utilized RVIZ to monitor and visualize incoming object data.
- **Applied Object Fusion Algorithms**: Integrated an object fusion algorithm to merge data from multiple infrastructure sensors, enhancing the robustness of object detection.

These skills are foundational for developing advanced V2X communication systems and autonomous vehicle perception modules.

---

## References

- [mqtt_client GitHub Repository](https://github.com/ika-rwth-aachen/mqtt_client)
- [HiveMQ MQTT Broker](https://broker.hivemq.com/)
- [ETSI ITS Standards](https://www.etsi.org/committee/its)
- [ROS Official Website](https://www.ros.org)
- [ikaObjectList Definition](https://github.com/ika-rwth-aachen/acdc/wiki/Section-2-Object-Detection#ika-ros-object-lists-definition)
- [MQTT Protocol Overview](https://mqtt.org/)
- [Mosquitto MQTT Broker](https://mosquitto.org/)
- [RVIZ Visualization Tool](http://wiki.ros.org/rviz)