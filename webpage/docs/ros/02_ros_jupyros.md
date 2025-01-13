# ROS JupyROS

## Introduction

JupyROS is an integration of **ROS1 (Robot Operating System)** and **Jupyter Notebooks**, enabling developers to interact with ROS topics, services, and parameters interactively within a notebook environment. It provides a powerful way to prototype, visualize, and document ROS-based robotics projects.

### Key Features of JupyROS:
- Interactive exploration of ROS topics, services, and parameters.
- Seamless integration of Python-based ROS1 APIs (`rospy`) with Jupyter cells.
- Visualizations of ROS data streams directly in the notebook.
- Ideal for education, debugging, and prototyping.

---

## Prerequisites

To use JupyROS effectively, ensure you have the following installed:

1. **ROS1**: Installed and configured on your system.
   - Verify using: `roscore` and `rostopic list`.
2. **Python**: Compatible with your ROS1 distribution.
   - ROS1 typically supports Python 2 or Python 3 (depending on your ROS distribution and configuration).
3. **Jupyter Notebook**: Installed via `pip` or `conda`.
   - Install Jupyter: `pip install notebook`.
4. **JupyROS**: A ROS-specific package for Jupyter.
   - Install JupyROS: `pip install jupyros`.

---

## Installation

### Step 1: Install Jupyter and Required Packages
Run the following commands to install Jupyter and Python dependencies:
```bash
pip install notebook jupyros ipywidgets matplotlib
```

### Step 2: Verify ROS Installation
Ensure ROS environment variables are sourced correctly:
```bash
source /opt/ros/<ros-distro>/setup.bash
```
Replace `<ros-distro>` with your ROS distribution, e.g., `melodic` or `noetic`.

### Step 3: Start the ROS Master
JupyROS requires a running ROS Master. Start it using:
```bash
roscore
```

### Step 4: Install JupyROS
Install JupyROS using pip:
```bash
pip install jupyros
```

### Step 5: Test Installation
Launch a Jupyter Notebook and create a new Python notebook. Test importing `jupyros`:
```python
from jupyros import ros
print("JupyROS successfully installed!")
```

---

## Getting Started with JupyROS

### Step 1: Launch Jupyter Notebook
Run the following command to start Jupyter:
```bash
jupyter notebook
```

### Step 2: Create a New Notebook
- Navigate to your browser where the Jupyter Notebook server is running.
- Create a new notebook with the Python kernel.

### Step 3: Import Required Libraries
In your notebook, import JupyROS and `rospy`:
```python
from jupyros import ros
import rospy
```

---

## Core Functionalities

### 1. Listing Available Topics
Use the following code snippet to list all active ROS topics:
```python
ros.topics()
```

### 2. Subscribing to a Topic
Subscribe to a ROS topic and visualize the data:
```python
from jupyros import ros
msgs = ros.subscribe('/topic_name', msg_type='std_msgs/String', throttle_rate=1.0)

# Print the latest message
msgs.tail()
```
Replace `/topic_name` with your desired topic and specify the correct `msg_type`.

### 3. Publishing to a Topic
Publish messages to a ROS topic:
```python
from std_msgs.msg import String
publisher = ros.Publisher('/topic_name', String)

# Publish a message
publisher.publish(String(data="Hello, ROS from Jupyter!"))
```

### 4. Calling a ROS Service
Call a ROS service using the `ros.call_service` function:
```python
ros.call_service('/service_name', {'arg1': value1, 'arg2': value2})
```
Replace `/service_name` and the arguments with the appropriate service details.

### 5. Interacting with ROS Parameters
Get and set ROS parameters directly:
```python
# Get a parameter
param_value = ros.param_get('/parameter_name')

# Set a parameter
ros.param_set('/parameter_name', 'new_value')
```

---

## Visualizing Data

JupyROS supports data visualization using `matplotlib` and `ipywidgets`.

### Example: Plotting Topic Data
Plot real-time data from a ROS topic:
```python
import matplotlib.pyplot as plt
from IPython.display import display
import time

# Subscribe to a numeric topic (e.g., sensor readings)
msgs = ros.subscribe('/sensor_topic', msg_type='std_msgs/Float64')

# Plot the data
plt.ion()
fig, ax = plt.subplots()
line, = ax.plot([], [])

def update_plot():
    data = msgs.tail(100)  # Fetch the last 100 messages
    line.set_ydata([msg.data for msg in data])
    line.set_xdata(range(len(data)))
    ax.relim()
    ax.autoscale_view()
    plt.draw()

while True:
    update_plot()
    plt.pause(0.5)
```

---

## Use Cases

### 1. Debugging Robot Behavior
JupyROS allows you to observe and manipulate topics, services, and parameters in real-time. For instance:
- Debug sensor readings by subscribing to `/sensor_topic`.
- Modify robot control parameters dynamically using `ros.param_set`.

### 2. Teaching and Tutorials
Interactive notebooks make it easier to teach ROS concepts with live examples. Students can experiment with:
- Publishing and subscribing to topics.
- Writing and calling services.
- Exploring ROS parameters.

### 3. Prototyping and Visualization
Developers can quickly prototype ROS-based applications by:
- Visualizing sensor data.
- Testing control algorithms.
- Recording data streams.

---

## Best Practices

1. **Environment Management**:
   Use a virtual environment to isolate JupyROS and its dependencies from your main ROS installation.

2. **Notebook Organization**:
   - Divide cells into logical sections (e.g., imports, topic subscriptions, visualizations).
   - Use Markdown cells to document your workflow.

3. **Error Handling**:
   Wrap your ROS interactions in try-except blocks to handle runtime issues:
   ```python
   try:
       ros.call_service('/non_existent_service', {})
   except Exception as e:
       print(f"Error: {e}")
   ```

4. **Visualization Performance**:
   Use throttling (e.g., `throttle_rate`) for real-time data visualization to avoid performance bottlenecks.

---

## Advanced Topics

### 1. Custom Message Types
JupyROS supports custom ROS message types. Ensure the type is available in your workspace:
```python
msgs = ros.subscribe('/custom_topic', msg_type='my_package/MyCustomMessage')
```

### 2. Multi-Robot Systems
Interact with topics and namespaces for multi-robot environments:
```python
ros.topics(namespace='/robot1')
```

### 3. ROS Bags in Jupyter
Read and analyze `.bag` files directly within Jupyter:
```python
import rosbag
bag = rosbag.Bag('example.bag')

for topic, msg, t in bag.read_messages(topics=['/topic_name']):
    print(msg)
```

---

## Troubleshooting

### Issue: `rosmaster not running`
- Ensure `roscore` is running before starting your notebook.

### Issue: `Message type not recognized`
- Verify that the required message type is available in your ROS workspace.

### Issue: Jupyter Notebook Kernel Crash
- Check ROS environment variables in the terminal:
  ```bash
  echo $ROS_MASTER_URI
  echo $ROS_IP
  ```

---

## Conclusion

JupyROS bridges the gap between the ROS ecosystem and Jupyter Notebooks, providing a powerful, interactive development environment. It is particularly useful for prototyping, debugging, and teaching ROS concepts. By leveraging its capabilities, developers and researchers can enhance productivity and achieve better insights into their robotics systems.

---

## References

1. [ROS Official Documentation](http://wiki.ros.org/)
2. [Jupyter Documentation](https://jupyter.org/)
3. [JupyROS GitHub Repository](https://github.com/RoboStack/jupyros)