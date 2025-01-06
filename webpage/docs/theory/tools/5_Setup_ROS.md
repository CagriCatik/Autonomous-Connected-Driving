# Setup ROS Coding Environment

## Introduction
This guide provides a detailed walkthrough for setting up the ROS coding environment required for the ACDC course. It includes instructions for installing and configuring essential tools like Docker, Visual Studio Code (VS Code), and related scripts, ensuring a seamless development experience with ROS and ROS 2.

---

To begin, it is recommended to use **Visual Studio Code (VS Code)** as the Integrated Development Environment (IDE) for editing and managing ROS code. Install VS Code by following the official [installation guide for Ubuntu](https://code.visualstudio.com/docs/setup/linux). Alternatively, you can use any IDE you are familiar with.

To set up the ACDC repository, open a terminal in Ubuntu and navigate to your preferred directory (e.g., `~/Documents`). Clone the repository using the following command:

```bash
git clone --recurse-submodules https://github.com/ika-rwth-aachen/acdc.git
```

Ensure the `--recurse-submodules` flag is included to download all required submodules. Once cloned, your local directory structure should resemble:

```plaintext
acdc/
├── assets/
├── bag/
├── catkin_workspace/
├── colcon_workspace/
├── docker/
├── .gitignore
├── LICENSE
└── README.md
```

**Installing Docker** is essential for running pre-configured containers tailored to the ACDC course. Follow the official installation guide:
- [Ubuntu](https://docs.docker.com/engine/install/ubuntu/)
- [MacOS](https://docs.docker.com/desktop/install/mac-install/)

After installation, test Docker by running:
```bash
sudo docker run hello-world
```

For rootless execution, add your user to the Docker group:
```bash
sudo groupadd docker
sudo usermod -aG docker $USER
newgrp docker
docker run hello-world
```

To simplify Docker container management, install the `docker-run-cli` tool:
```bash
pip install docker-run-cli --break-system-packages
```

If `docker-run` is not recognized, ensure it is added to your PATH:
```bash
echo "export PATH=\$HOME/.local/bin:\$PATH" >> ~/.bashrc
source ~/.bashrc
```

Download the necessary Docker image from DockerHub:
```bash
docker pull rwthika/acdc:ros1
```

Alternatively, use the provided Makefile commands:
- Pull the image: `make pull`
- Clean local images: `make clean`
- Build locally: `make build` (time-intensive).

Start the Docker container by navigating to the `docker` directory and executing:
```bash
./ros1_run.sh
```

Run the script again to open an additional shell inside the running container. Changes made in the shared `catkin_workspace` and `bag` directories will reflect both in the host and container.

To test GUI functionality, execute:
```bash
rqt
```
An empty GUI window should appear if everything is correctly set up.

RVIZ is a crucial visualization tool for ROS. To test it, open three terminals:
1. Start the container in Terminal 1:
   ```bash
   ./ros1_run.sh
   ```
2. Launch the ROS Master in Terminal 2:
   ```bash
   roscore
   ```
3. Start RVIZ in Terminal 3:
   ```bash
   rviz
   ```

If RVIZ does not display content or crashes with `libGL` errors, disable hardware acceleration temporarily:
```bash
export LIBGL_ALWAYS_SOFTWARE=1
rosrun rviz rviz
```

For a permanent fix, modify the `ros1_run.sh` script to include:
```bash
docker-run --env LIBGL_ALWAYS_SOFTWARE=1 --no-gpu --volume $(dirname "$DIR"):/home/rosuser/ws --image rwthika/acdc:ros1 --workdir="/home/rosuser/ws/catkin_workspace" --name acdc_ros1
```

By following this guide, your ROS coding environment will be ready to handle the tasks and projects in the ACDC course effectively. 