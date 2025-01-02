# Setup for Operating System

This part provides a structured and user-friendly guide to setting up the required environment for the ACDC programming exercises, focusing on ROS and Jupyter Notebook tasks. It caters to users on different operating systems, offering detailed explanations, installation steps, and troubleshooting tips.

---

## **1. Overview of Operating System Requirements**

The recommended setup for the ACDC exercises involves Ubuntu, either as a native installation or via dual boot. Users can also opt for alternative setups based on their system capabilities and preferences:

### **1.1 Recommended Setup**
- **Native Ubuntu Installation**: Provides the best performance and compatibility.
- **Dual Boot with Ubuntu**: Ensures full utilization of resources for Ubuntu-specific tasks.

### **1.2 Alternative Setups**
- **Windows Subsystem for Linux (WSL2)**: Integrates a Linux environment into Windows.
- **Virtual Machines (VM)**: Suitable for users who prefer isolated environments.

### **1.3 MacOS Considerations**
- **Intel Chip**: Supports both native dual boot and virtual machines.
- **Silicon Chip**: Limited support due to ARM64 architecture restrictions.

---

## **2. Setting Up Ubuntu**

### **2.1 Installation**
For users with existing Ubuntu installations:
1. Update the package repository:
   ```bash
   sudo apt update
   ```
2. Upgrade installed packages:
   ```bash
   sudo apt upgrade
   ```

### **2.2 Required Packages**
To execute the programming tasks, install the following:
- **Git**:
  ```bash
  sudo apt install git-all
  ```
  Git is essential for source code management, collaboration, and version control.
- **Docker**:
  Docker provides containerized environments for consistent execution:
  ```bash
  sudo apt install docker.io
  ```

### **2.3 Docker-Run-CLI**
Install the `docker-run-cli` for executing Docker containers:
```bash
pip install docker-run-cli --break-system-packages
```
- If you encounter `command not found`, add the executable to your shell's PATH:
  ```bash
  echo "export PATH=$HOME/.local/bin:$PATH" >> ~/.bashrc
  source ~/.bashrc
  ```

---

## **3. Windows Setup**

### **3.1 Using WSL2**
Follow the instructions to enable WSL2:
- Install a compatible Linux distribution (e.g., Ubuntu) via the Microsoft Store.
- Configure the environment for ROS and Docker tasks.

### **3.2 Dual Boot or Virtual Machine**
If WSL2 is not an option:
1. **Dual Boot**:
   - Use tutorials for safe partitioning and installation of Ubuntu.
   - Ensure at least 80 GB of free space.
2. **Virtual Machines**:
   - Recommended: VMware Workstation Pro (free for personal use).
   - Avoid VirtualBox due to known compatibility issues.

---

## **4. MacOS Setup**

### **4.1 Intel Chip**
- Supports both dual boot and virtual machine setups.

### **4.2 Silicon Chip**
- Use Docker for limited exercises.
- Limitations include lack of RVIZ support and issues with case-sensitive packages like `trajectory_planner`.

---

## **5. Environment-Specific Considerations**

### **5.1 Performance Tips**
- Dual boot setups provide better performance than virtual machines.
- Allocate sufficient resources to virtual machines if used.

### **5.2 Known Issues**
- Avoid VirtualBox due to instability.
- MacOS Silicon users may face software restrictions; ensure Docker configurations are updated.

---

## **6. Additional Tools**

### **6.1 Jupyter Notebooks**
Jupyter provides a platform for interactive Python programming with live code, equations, and visualizations:
- Access prepared notebooks with coding tasks and explanations.
- Install Jupyter Notebook:
  ```bash
  sudo apt install python3-notebook jupyter jupyter-core
  ```

### **6.2 GitHub and Issue Tracking**
Leverage GitHub for managing repositories and tracking issues:
- Report bugs or issues via the course’s GitHub repository.
- Participate in discussions and contribute to solutions.

---

## **7. Conclusion**

Setting up the environment for ACDC tasks involves selecting the appropriate operating system configuration and installing necessary tools like ROS, Docker, and Jupyter. Following these steps ensures compatibility and smooth execution of the exercises. This guide aims to simplify the setup process, catering to both beginners and advanced users. 

For further assistance, consult:
- ROS [documentation](http://wiki.ros.org).
- Docker [documentation](https://docs.docker.com).
- ACDC support channels