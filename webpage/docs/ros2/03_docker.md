# Using Docker for ROS2 Development

Docker simplifies the setup and management of development environments by providing isolated, portable, and consistent containers. Using Docker for ROS2 development ensures that your environment is easy to replicate and maintain, especially when collaborating across different systems.

---

## Overview of Docker for ROS2

Docker allows you to:

1. Isolate Development Environments: Avoid conflicts with local installations by running ROS2 in a container.
2. Share and Reproduce: Easily share configurations and ensure consistency across team members.
3. Simplify Setup: Quickly get started with pre-built ROS2 images from Docker Hub or custom images tailored to your needs.

---

## Setting Up Docker for ROS2 Development

### Prerequisites
Ensure you have the following installed:

- Docker: Follow the [official installation guide](https://docs.docker.com/get-docker/).
- Docker Compose (Optional): For managing multi-container applications.

Verify Docker installation:
```bash
$ docker --version
Docker version 20.10.x, build xxxxxxx
```

Verify Docker Compose installation:
```bash
$ docker-compose --version
Docker Compose version 2.x.x
```

### Pulling a ROS2 Image

Official ROS2 Docker images are available on Docker Hub:

```bash
$ docker pull osrf/ros:<ros2-distro>
```
Example for ROS2 Humble:
```bash
$ docker pull osrf/ros:humble-desktop
```

---

## Basic Docker Commands for ROS2

### Running a ROS2 Container

```bash
$ docker run -it --rm osrf/ros:humble-desktop
```
This starts an interactive terminal session inside the ROS2 container.

### Mounting Local Directories
To access local files in your container:

```bash
$ docker run -it --rm -v $(pwd):/workspace osrf/ros:humble-desktop
```
This mounts the current directory to `/workspace` inside the container.

### Networking with ROS2
To allow ROS2 nodes to communicate across containers or with the host machine, use the `--net=host` option:

```bash
$ docker run -it --rm --net=host osrf/ros:humble-desktop
```
> Note: The `--net=host` option works only on Linux. On Windows and macOS, consider setting up bridge networking or Docker Compose.

---

## Using Docker Compose for ROS2 Projects

Docker Compose simplifies multi-container applications. Here's an example configuration for a ROS2 setup:

### docker-compose.yml
```yaml
version: '3.8'
services:
  ros2:
    image: osrf/ros:humble-desktop
    container_name: ros2_container
    volumes:
      - ./workspace:/workspace
    networks:
      - ros2_network
    command: /bin/bash

networks:
  ros2_network:
    driver: bridge
```

### Running the Setup
1. Start the container:
   ```bash
   $ docker-compose up -d
   ```

2. Access the container:
   ```bash
   $ docker exec -it ros2_container /bin/bash
   ```

---

## Building Custom ROS2 Docker Images

If you need a tailored ROS2 environment, create a custom Dockerfile.

### Dockerfile Example
```dockerfile
FROM osrf/ros:humble-desktop

# Install additional dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    ros-humble-rmw-cyclonedds-cpp

# Set up workspace
WORKDIR /workspace
```

### Building and Running the Image
1. Build the image:
   ```bash
   $ docker build -t custom_ros2:humble .
   ```

2. Run the container:
   ```bash
   $ docker run -it --rm custom_ros2:humble
   ```

---

## Advanced Usage: Persistent Volumes and GUI Support

### Persistent Volumes
To retain data between container runs, use Docker volumes:

```bash
$ docker run -it --rm -v ros2_data:/workspace osrf/ros:humble-desktop
```

### Running GUI Applications
To enable GUI applications inside a ROS2 container (e.g., RViz):

1. Allow X11 forwarding:
   ```bash
   $ xhost +local:
   ```

2. Run the container with display access:
   ```bash
   $ docker run -it --rm \
      -e DISPLAY=$DISPLAY \
      -v /tmp/.X11-unix:/tmp/.X11-unix \
      osrf/ros:humble-desktop
   ```

---

## Best Practices

1. Version Pinning: Always use specific versions of ROS2 images to avoid unexpected changes.
2. Leverage Docker Compose: Simplify workflows for complex setups.
3. Optimize Images: Use minimal base images and remove unnecessary packages.
4. Document Your Setup: Share `Dockerfile` and `docker-compose.yml` with your team for reproducibility.

---

Using Docker for ROS2 development streamlines the process of environment setup and sharing. By leveraging Docker's features, you can focus more on developing ROS2 applications and less on troubleshooting dependencies.

