# ROS 2 Installation Guide

This comprehensive guide provides step-by-step instructions to install ROS 2 on your machine, tailored to common platforms like Ubuntu. This guide caters to both beginners and advanced users by ensuring clarity and technical depth.

---

## 1. Prerequisites

Before installing ROS 2, ensure your system meets the following requirements:

### 1.1 Supported Platforms
- Ubuntu: ROS 2 is primarily supported on Ubuntu Linux. The recommended versions for ROS 2 Humble are Ubuntu 22.04 (Jammy Jellyfish) or Ubuntu 20.04 (Focal Fossa).

### 1.2 System Preparation
- Ensure your system is up-to-date. Run:
  ```bash
  sudo apt update && sudo apt upgrade -y
  ```
- Install essential tools:
  ```bash
  sudo apt install -y curl gnupg lsb-release
  ```

---

## 2. Installing ROS 2

### 2.1 Add ROS 2 Package Repository
Add the ROS 2 package repository to your system:
1. Import the GPG key:
   ```bash
   sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
   ```
2. Add the repository to your sources:
   ```bash
   echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
   ```

---

### 2.2 Update Package List
Refresh the package list to include ROS 2 packages:
```bash
sudo apt update
```

---

### 2.3 Install ROS 2 Packages
#### Desktop Installation (Recommended)
This setup includes ROS 2 tools, libraries, and GUI tools like Rviz2:
```bash
sudo apt install ros-humble-desktop
```

#### Base Installation
For headless systems or minimal installations:
```bash
sudo apt install ros-humble-ros-base
```

#### Custom Installation
To install specific packages:
```bash
sudo apt install ros-humble-<package_name>
```

---

## 3. Environment Setup

### 3.1 Source ROS 2 Setup
To use ROS 2 commands, source the setup script:
```bash
source /opt/ros/humble/setup.bash
```

Add this to your shell's configuration file for automatic sourcing:
- For Bash:
  ```bash
  echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
  source ~/.bashrc
  ```
- For Zsh:
  ```bash
  echo "source /opt/ros/humble/setup.zsh" >> ~/.zshrc
  source ~/.zshrc
  ```

---

## 4. Installing Additional Tools

### 4.1 Install `colcon`
`colcon` is a build tool for ROS 2:
```bash
sudo apt install python3-colcon-common-extensions
```

### 4.2 Install ROS 2 CLI Tools
For managing workspaces and packages:
```bash
sudo apt install python3-rosdep python3-argcomplete
```

---

## 5. Initialize ROS 2 Dependencies
Initialize `rosdep` to manage dependencies:
1. Install and initialize:
   ```bash
   sudo rosdep init
   rosdep update
   ```
2. Verify by running:
   ```bash
   rosdep check <package_name>
   ```

---

## 6. Verify Installation
Run the following commands to verify the installation:
1. Check ROS 2 version:
   ```bash
   ros2 --version
   ```
2. Launch the demo node:
   ```bash
   ros2 run demo_nodes_cpp talker
   ```
   Open another terminal and launch the listener node:
   ```bash
   ros2 run demo_nodes_cpp listener
   ```
   You should see the nodes communicating.

---

## 7. Troubleshooting

### 7.1 Common Issues
- GPG Key Error: Re-run the GPG key installation command and ensure proper internet access.
- Missing Dependencies: Use `rosdep install` to resolve dependencies:
  ```bash
  rosdep install --from-paths src --ignore-src -r -y
  ```
- Source Issues: Ensure the correct setup file is sourced (`setup.bash` or `setup.zsh`).