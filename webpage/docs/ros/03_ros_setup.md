# ROS Setup

This documentation provides a detailed guide for setting up and configuring ROS1 (Robot Operating System) on your machine. Follow these instructions to ensure a successful installation and configuration.

---

## Prerequisites

### Supported Operating Systems
ROS1 is designed to work on Ubuntu. Commonly used versions include:
- ROS Noetic: Compatible with Ubuntu 20.04 (Focal).
- ROS Melodic: Compatible with Ubuntu 18.04 (Bionic).

### System Requirements
Ensure your machine has at least 2 GB of RAM (4 GB recommended) and an active internet connection for package downloads. Administrative (sudo) privileges are also necessary.

### System Updates
Before proceeding, update your system to ensure all packages are current:
```bash
sudo apt update && sudo apt upgrade -y
```

### Required Tools
Install essential development tools for building and managing ROS projects:
```bash
sudo apt install build-essential cmake git curl -y
```

---

## Installing ROS1

### Setting Up Package Sources
Add the ROS package repository:
```bash
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
```

Add the GPG key for the ROS repository:
```bash
curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
```

### Installing ROS
Update your package list and install ROS. Choose the version corresponding to your Ubuntu distribution:
- For ROS Noetic:
  ```bash
  sudo apt install ros-noetic-desktop-full
  ```
- For ROS Melodic:
  ```bash
  sudo apt install ros-melodic-desktop-full
  ```

### Configuring Dependencies
Initialize `rosdep` to manage dependencies:
```bash
sudo rosdep init
rosdep update
```

### Environment Configuration
Add ROS environment variables to your shell configuration:
```bash
echo "source /opt/ros/<distro>/setup.bash" >> ~/.bashrc
source ~/.bashrc
```
Replace `<distro>` with the version you installed, such as `noetic` or `melodic`.

---

## Setting Up a Workspace

### Creating a Catkin Workspace
Set up a workspace for your ROS projects:
```bash
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/
catkin_make
```

Source the workspace:
```bash
echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### Verifying the Workspace
Check if your workspace has been sourced correctly:
```bash
echo $ROS_PACKAGE_PATH
```
You should see the path to your `catkin_ws/src` directory in the output.

---

## Installing Additional ROS Packages

### Installing Packages
Install additional ROS packages using:
```bash
sudo apt install ros-<distro>-<package_name>
```
For example:
```bash
sudo apt install ros-noetic-turtlebot3
```

### Listing Available Packages
To see all ROS packages available for your distribution:
```bash
apt-cache search ros-noetic
```

---

## Testing Your Installation

### Running ROS Core
Start the ROS master node:
```bash
roscore
```

### Running a Sample Package
Run a sample ROS node to ensure everything is working:
```bash
rosrun turtlesim turtlesim_node
```

Control the node using another terminal:
```bash
rosrun turtlesim turtle_teleop_key
```

### Checking Active Topics and Nodes
View the active topics:
```bash
rostopic list
```

List the running nodes:
```bash
rosnode list
```

---

## Additional Configuration

### Networking Configuration
For multi-machine setups, configure the `ROS_MASTER_URI` and `ROS_HOSTNAME` environment variables:
```bash
export ROS_MASTER_URI=http://<master_ip>:11311
export ROS_HOSTNAME=<your_ip>
```
Add these to your `~/.bashrc` file for persistence.

### Creating Aliases
Simplify ROS commands with aliases:
```bash
echo "alias cw='cd ~/catkin_ws'" >> ~/.bashrc
echo "alias cs='cd ~/catkin_ws/src'" >> ~/.bashrc
echo "alias cm='cd ~/catkin_ws && catkin_make'" >> ~/.bashrc
```

---

## Troubleshooting

### Common Issues
- **`roscore` fails to start:** Ensure it is not already running.
- **Missing dependencies:** Use `rosdep` to install them:
  ```bash
  rosdep install --from-paths src --ignore-src -r -y
  ```
- **Workspace build errors:** Confirm that all required packages are installed and sourced.

### Debugging Tools
- Use `rqt_console` for viewing ROS logs:
  ```bash
  rqt_console
  ```
- Visualize node interactions using:
  ```bash
  rqt_graph
  ```

---

## References

- [ROS Official Website](https://www.ros.org/)
- [ROS Wiki](http://wiki.ros.org/)
- [ROS Tutorials](http://wiki.ros.org/ROS/Tutorials)

---

By following this guide, you will have a fully operational ROS1 setup. For advanced topics, such as creating custom messages, integrating with hardware, or building simulation environments, refer to the official ROS documentation.