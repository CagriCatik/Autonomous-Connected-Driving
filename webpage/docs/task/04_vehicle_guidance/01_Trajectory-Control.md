# Trajectory Control

![ROS1](https://img.shields.io/badge/ROS1-blue)

## Overview

In autonomous vehicle systems, maintaining precise control over the vehicle's trajectory is paramount for safety and performance. In previous workshops, it was observed that the vehicle occasionally deviates from its planned path, leading to corrective maneuvers that realign it with the desired trajectory. This behavior stems from the implementation of a Bi-Level-Stabilization guidance system. However, the absence of a low-level stabilization controller in earlier sessions meant that the vehicle couldn't accurately follow the high-level trajectory planner's directives. When deviations exceeded a certain threshold, the Model Predictive Control (MPC) system was reinitialized using the vehicle's current state, causing abrupt realignments.

This workshop aims to bridge that gap by implementing a comprehensive low-level stabilization controller, enhancing the vehicle's ability to adhere to planned trajectories seamlessly.

## Learning Objectives

By the end of this workshop, participants will be able to:

- **Implement Odometry Equations:** Calculate the lateral and heading deviations of the vehicle relative to the planned trajectory.
- **Compute Control Deviations:** Utilize target trajectory values and the current vehicle state to determine control deviations.
- **Derive Discrete PID Controller Output:** Formulate the output equation for a discrete PID controller.
- **Develop Velocity and Lateral Controllers:** Implement a longitudinal velocity controller and a cascaded lateral controller that determines the desired yaw rate.
- **Apply Inverse Single-Track Model:** Convert the desired yaw rate into an appropriate steering angle using an inverse single-track model.

## Introduction

The core objective of this workshop is to implement a compensatory feedback controller serving as the low-level stabilization mechanism within the vehicle's guidance system. This controller works in tandem with the high-level trajectory planner to ensure smooth and accurate trajectory following.

### Controller Architecture

The controller's architecture is visually represented below:

![Controller Architecture](../images/section_4/ctr_arc.PNG)

The architecture comprises various components responsible for different aspects of trajectory control. Participants will be required to fill in specific sections within the controller's codebase to complete the implementation.

### Repository and Codebase

The foundational code for this workshop is hosted on GitHub. Participants should focus on the following files, which contain designated gaps to be filled:

- **Trajectory Controller:** [TrajectoryCtrl.cpp](https://github.com/ika-rwth-aachen/acdc/blob/main/catkin_workspace/src/workshops/section_4/trajectory_ctrl/src/TrajectoryCtrl.cpp)
- **PID Controller:** [PID.cpp](https://github.com/ika-rwth-aachen/acdc/blob/main/catkin_workspace/src/workshops/section_4/trajectory_ctrl/src/PID.cpp)

## Tasks

### Task 1: Implementation of Odometry Equations

**Objective:** Implement the odometry equations to determine the vehicle's lateral (`dy_`) and heading (`dpsi_`) deviations from the planned trajectory.

**Steps:**

1. **Navigate to the Code:**
   Open the `TrajectoryCtrl.cpp` file and locate line 242:
   [TrajectoryCtrl.cpp at Line 242](https://github.com/ika-rwth-aachen/acdc/blob/main/catkin_workspace/src/workshops/section_4/trajectory_ctrl/src/TrajectoryCtrl.cpp#L242)

2. **Fill in the Gaps:**
   ```c++
   // START TASK 1 CODE HERE
   // use helping comments from Wiki
   double yawRate = cur_vehicle_state_.yaw_rate;
   double velocity = cur_vehicle_state_.velocity;
   odom_dy_ += sin(dpsi_ + 0.5 * yawRate * dt) * (velocity * dt);
   odom_dpsi_ += yawRate * dt;
   // END TASK 1 CODE HERE
   ```

**Hints:**

- **Variables:**
  - `yawRate`: Current yaw rate of the vehicle.
  - `velocity`: Current velocity of the vehicle.
  - `dt`: Time step since the last odometry update.
  
- **Formulas:**
  - **Lateral Movement:** 
  $$
  dy = \sin\left(d\psi + 0.5 \cdot \dot{\psi} \cdot dt\right) \cdot ds
  $$  
    Where \( ds = velocity \cdot dt \) is the distance traveled since the last update.
  
  - **Heading Change:**
  $$
  d\psi = \dot{\psi} \cdot dt
  $$

### Task 2: Calculate the Control Deviations

**Objective:** Utilize the odometry outputs to compute the vehicle's deviations from the current trajectory.

**Steps:**

1. **Navigate to the Code:**
   Open the `TrajectoryCtrl.cpp` file and locate line 196:
   [TrajectoryCtrl.cpp at Line 196](https://github.com/ika-rwth-aachen/acdc/blob/main/catkin_workspace/src/workshops/section_4/trajectory_ctrl/src/TrajectoryCtrl.cpp#L196)

2. **Fill in the Gaps:**
   ```c++
   // START TASK 2 CODE HERE
   // calculate vehicle deviations from the trajectory
   // use helping comments from Wiki
   dy_ = odom_dy_ - target_dy;
   dpsi_ = odom_dpsi_ - target_dpsi;
   // END TASK 2 CODE HERE
   ```

**Hints:**

- **Variables:**
  - `odom_dy_`: Calculated lateral deviation from odometry.
  - `odom_dpsi_`: Calculated heading deviation from odometry.
  - `target_dy`: Desired lateral position from the trajectory.
  - `target_dpsi`: Desired heading from the trajectory.

### Task 3: Implement the Output Equation of a Discrete PID Controller

**Objective:** Develop the output equation for a discrete PID controller to establish a feedback control loop based on the calculated deviations.

**Steps:**

1. **Navigate to the Code:**
   Open the `PID.cpp` file and locate line 44:
   [PID.cpp at Line 44](https://github.com/ika-rwth-aachen/acdc/blob/main/catkin_workspace/src/workshops/section_4/trajectory_ctrl/src/PID.cpp#L44)

2. **Fill in the Gaps:**
   ```c++
   // START TASK 3 CODE HERE
   // use helping comments from Wiki
   i_val_ += e * dt;
   double d_val = (e - prev_e_) / dt;
   prev_e_ = e;
   return Kp_ * e + Ki_ * i_val_ + Kd_ * d_val;
   // END TASK 3 CODE HERE
   ```

**Hints:**

- **Variables:**
  - `e`: Controller error.
  - `i_val_`: Integral of the controller error.
  - `d_val`: Derivative of the controller error.
  - `Kp_`, `Ki_`, `Kd_`: Proportional, Integral, and Derivative gains respectively.

- **Equation:**
$$
  \text{Output} = K_p \cdot e + K_i \cdot \int e \, dt + K_d \cdot \frac{de}{dt}
$$


### Task 4: Longitudinal Velocity Controller

**Objective:** Implement a longitudinal velocity controller using the PID controller to manage the vehicle's speed.

**Steps:**

1. **Navigate to the Code:**
   Open the `TrajectoryCtrl.cpp` file and locate line 345:
   [TrajectoryCtrl.cpp at Line 345](https://github.com/ika-rwth-aachen/acdc/blob/main/catkin_workspace/src/workshops/section_4/trajectory_ctrl/src/TrajectoryCtrl.cpp#L345)

2. **Fill in the Gaps:**
   ```c++
   // START TASK 4 CODE HERE
   // use helping comments from Wiki
   double dt = ros::Time::now().toSec() - vhcl_ctrl_output_.header.stamp.toSec();
   double velocity = cur_vehicle_state_.velocity;
   double w_v = v_tgt_;
   double e_v = w_v - velocity;
   double a_fb_v = dv_pid.Calc(e_v, dt);
   // END TASK 4 CODE HERE
   ```

**Hints:**

- **Variables:**
  - `w_v`: Desired (target) velocity.
  - `velocity`: Current vehicle velocity.
  - `e_v`: Velocity error.
  - `a_fb_v`: Feedback acceleration command.
  
- **PID Controller Instance:**
  - `dv_pid`: Instance of the velocity PID controller.
  - Method: `Calc(double e, double dt)` computes the PID output based on the error and time step.

### Task 5: Cascaded Lateral Controller

**Objective:** Develop a cascaded lateral feedback controller that computes the desired yaw rate based on lateral and heading deviations.

**Steps:**

1. **Navigate to the Code:**
   Open the `TrajectoryCtrl.cpp` file and locate line 262:
   [TrajectoryCtrl.cpp at Line 262](https://github.com/ika-rwth-aachen/acdc/blob/main/catkin_workspace/src/workshops/section_4/trajectory_ctrl/src/TrajectoryCtrl.cpp#L262)

2. **Fill in the Gaps:**
   ```c++
   // START TASK 5 CODE HERE
   // use helping comments from Wiki
   double dt = (ros::Time::now() - vhcl_ctrl_output_.header.stamp).toSec();
   double w_y = 0.0; // Desired lateral position can be set or computed as needed
   double e_y = dy_ - w_y;
   double w_psi = 0.0; // Desired heading can be set or computed as needed
   double e_psi = dpsi_ - w_psi;
   double psi_dot_des = dy_pid.Calc(e_y, dt) + dpsi_pid.Calc(e_psi, dt);
   // END TASK 5 CODE HERE
   ```

**Hints:**

- **Variables:**
  - `dy_`: Lateral deviation from the trajectory.
  - `dpsi_`: Heading deviation from the trajectory.
  - `w_y`: Desired lateral position (often set to zero if following the trajectory precisely).
  - `w_psi`: Desired heading (often aligned with the trajectory's heading).
  - `psi_dot_des`: Desired yaw rate computed from the PID controllers.
  
- **PID Controller Instances:**
  - `dy_pid`: PID controller for lateral deviation.
  - `dpsi_pid`: PID controller for heading deviation.
  
- **Method:**
  - `Calc(double e, double dt)` computes the PID output based on the error and time step.

### Task 6: Inverse Single-Track Model

**Objective:** Convert the desired yaw rate into an appropriate steering angle using the inverse single-track (bicycle) model.

**Steps:**

1. **Navigate to the Code:**
   Open the `TrajectoryCtrl.cpp` file and locate line 287:
   [TrajectoryCtrl.cpp at Line 287](https://github.com/ika-rwth-aachen/acdc/blob/main/catkin_workspace/src/workshops/section_4/trajectory_ctrl/src/TrajectoryCtrl.cpp#L287)

2. **Fill in the Gaps:**
   ```c++
   // START TASK 6 CODE HERE
   // use helping comments from Wiki
   double st_ang_pid = (psi_dot_des * (wheelbase_ + self_st_gradient_ * velocity * velocity)) / velocity;
   // END TASK 6 CODE HERE
   ```

**Hints:**

- **Variables:**
  - `self_st_gradient_` $EG$: Vehicle's self-steering gradient.
  - `wheelbase_` $l $: Distance between the front and rear axles.
  - `velocity` $v$: Current velocity of the vehicle.
  - `psi_dot_des` $\dot{\psi}_{des}$: Desired yaw rate.
  
- **Formula:**
$$
  \delta = \dot{\psi} \cdot \frac{(l + EG \cdot v^2)}{v}
$$

  Where \( \delta \) is the steering angle.


## Execution and Simulation

### Building the Workspace

Once all tasks are completed, it's time to build the workspace and run the simulation to observe the controller in action.

1. **Start the Docker Container:**
   ```bash
   cd acdc/docker
   ./ros1_run.sh
   ```

2. **Build the Workspace:**
   ```bash
   catkin build
   source devel/setup.bash
   ```
   *Note: Ensure you source the setup file in every new terminal session.*

### Running the Simulation

1. **Launch the Vehicle Guidance System:**
   ```bash
   roslaunch trajectory_planner vehicle_guidance.launch
   ```

2. **Introduce Dynamic Objects:**
   In a separate terminal within the Docker container, play a pre-recorded bag file to simulate dynamic obstacles:
   ```bash
   rosbag play ~/ws/bag/acdc_fusion_guidance_noise.bag
   ```

### Expected Outcome

Upon successful implementation and execution, the vehicle should adhere closely to the planned trajectory, demonstrating improved stability and control compared to previous configurations. The simulation should resemble the following:

![Solution](../images/section_4/sol_control.gif)

## Bonus Task

Enhance your understanding of the controller's dynamics by experimenting with real-time parameter adjustments.

### Steps:

1. **Open Dynamic Reconfigure Interface:**
   In a new terminal, execute:
   ```bash
   source devel/setup.bash
   rosrun rqt_reconfigure rqt_reconfigure
   ```

2. **Adjust Parameters:**
   A graphical interface will appear, allowing you to tweak various parameters of the trajectory planner and controller. Key adjustments include:

   - **Controller Gains:**
     - Set `dv_P`, `dv_I`, `dv_D`, etc., to zero to disable feedback control, reverting to behavior observed in earlier workshops.
   
   - **Deviation Parameters:**
     - Set `deviationMax...` parameters to zero, making the MPC reinitialize every optimization step and disable low-level stabilization.
   
   - **Cost Function Weights:**
     - Modify the weights and reference factors to observe their impact on trajectory adherence and control smoothness.

3. **Observe Effects:**
   Monitor how changes influence the vehicle's behavior in the simulation. This hands-on experience solidifies the understanding of controller dynamics and the significance of each parameter.

## Wrap-up

Through this workshop, participants have:

- **Implemented Odometry Equations:** Calculated lateral and heading deviations relative to the planned trajectory.
- **Computed Control Deviations:** Utilized trajectory targets and vehicle state to determine necessary control adjustments.
- **Developed a Discrete PID Controller:** Formulated and implemented the output equation for effective feedback control.
- **Established Velocity and Lateral Controllers:** Created controllers for managing longitudinal speed and lateral positioning, culminating in the computation of desired yaw rates.
- **Applied the Inverse Single-Track Model:** Translated desired yaw rates into steering angles, completing the control loop for trajectory adherence.

These skills are foundational for developing robust autonomous vehicle systems capable of precise and reliable trajectory following.

## References

- **Control Toolbox:**
  
  ```
  @article{adrlCT,
    title={The control toolbox — An open-source C++ library for robotics, optimal and model predictive control},
    author={Markus Giftthaler and Michael Neunert and Markus St{\"a}uble and Jonas Buchli},
    journal={2018 IEEE International Conference on Simulation, Modeling, and Programming for Autonomous Robots (SIMPAR)},
    year={2018},
    pages={123-129}
  }
  ```

- **Flatland:** [https://github.com/avidbots/flatland](https://github.com/avidbots/flatland)
- **ROS (Robot Operating System):** [https://www.ros.org](https://www.ros.org)