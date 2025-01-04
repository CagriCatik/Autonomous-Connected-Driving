# Automated and Connected Driving

This repository complements the [ACDC MOOC](https://learning.edx.org/course/course-v1:RWTHx+ACDC+3T2023/home) offered by RWTH Aachen University. The course provides a comprehensive guide to the development and testing of automated and connected driving functions.

## Course Overview

This **MOOC** introduces participants to key concepts in automated and connected driving, combining theoretical knowledge with practical implementation. Topics include:

- **Sensor Data Processing:** Techniques like segmentation, object detection, and grid mapping.
- **Environment Modeling and Prediction:** Methods for creating a digital twin of the driving environment.
- **Vehicle Guidance and Control:** Navigation, trajectory planning, and stabilization.
- **Connected Driving:** Understanding Vehicle-to-Everything (V2X) communication.

The course employs the **Robot Operating System (ROS)** as the backbone for building software modules.

## Repository Contents

This repository is structured to align with the ACDC course modules. It includes:

1. **Jupyter Notebooks:** Hands-on exercises for Python and ROS programming.
2. **ROS Implementation Examples:** Demonstrations of foundational ROS concepts like nodes, topics, and services.
3. **Coding Assignments:** Tasks that build practical skills in topics like object tracking, navigation, and cloud integration.

### Modules:

- **Section 1: Introduction & Tools**
- **Section 2: Sensor Data Processing**
- **Section 3: Object Fusion & Tracking**
- **Section 4: Vehicle Guidance**
- **Section 5: Connected Driving**

Each module includes coding assignments, quizzes, and additional resources.

## Getting Started

### Prerequisites

1. **Python** (3.8 or later) and **Jupyter Notebook**.
2. **ROS** (Noetic or ROS 2 Humble).
3. **Docker** for isolated development environments.
4. **Node.js** for serving Docusaurus documentation (optional).

### Steps

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/ika-rwth-aachen/acdc-notebooks.git
   ```

2. **Navigate to the Directory:**
   ```bash
   cd acdc-notebooks
   ```

3. **Set Up the Environment:**
   ```bash
   python3 -m venv acdc-env
   source acdc-env/bin/activate
   pip install -r requirements.txt
   ```

4. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```

5. **Access Notebooks:** Open the desired section and start learning!

## Documentation

I utilize **Docusaurus** for structured documentation. Follow these steps to serve the docs locally:

1. **Install Dependencies:**
   ```bash
   npm install
   ```

2. **Serve Locally:**
   ```bash
   npm start
   ```
   Access at `http://localhost:3000`.

3. **Build Static Files (Optional):**
   ```bash
   npm run build
   ```
   The static files will be generated in the `build` directory.

## Additional Resources

- **[ACDC Wiki](https://github.com/ika-rwth-aachen/acdc/wiki):** Comprehensive details about course content and tasks.
- **[edX Course Page](https://learning.edx.org/course/course-v1:RWTHx+ACDC+3T2023/home):** Enroll to access all materials.
- **[ROS Resources](https://www.ros.org/):** Documentation and community support.

## Acknowledgments

This project is developed by the **Institute for Automotive Engineering (ika)** at RWTH Aachen University. Special thanks to all contributors and participants for advancing the field of automated and connected driving.
