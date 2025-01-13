# Trajectory Planning

![ROS1](https://img.shields.io/badge/ROS1-blue)

## Overview

Trajectory planning is a critical component in autonomous vehicle systems, enabling the vehicle to navigate from its current position to a desired destination while avoiding obstacles and adhering to dynamic constraints. In this workshop, we will implement a trajectory-planning approach using the [Control Toolbox](https://github.com/ethz-adrl/control-toolbox), a versatile C++ library designed for robotics, optimal, and model predictive control. Participants will engage with a pre-configured **ROS** node, focusing on implementing key aspects of the optimal control problem to achieve efficient and safe trajectory generation.

## Learning Objectives

By the end of this workshop, participants will be able to:

- **Implement System Dynamics Model:** Develop and integrate a system dynamics model for a car-like vehicle.
- **Integrate Cost Terms:** Incorporate various cost terms into the trajectory planner to penalize:
  - High lateral and longitudinal jerk values.
  - High steering rates.
  - Deviations from a specified reference velocity.
- **Handle Collision Avoidance:** Implement mechanisms to avoid collisions with dynamic objects using cost functions.
- **Utilize Control Toolbox Features:** Leverage automatic differentiation and other features of the Control Toolbox for efficient trajectory optimization.

## Setup Instructions

1. **Download the Required Bag File:**
   
   If you haven't already, download the `acdc_fusion_guidance_noise.bag` file from [here](https://rwth-aachen.sciebo.de/s/1weBryyNoDRIhFS) and save it to the local directory `acdc/bag` on your host machine.

2. **Start the Docker Container:**
   
   Open a terminal and navigate to the Docker directory:
   ```bash
   cd acdc/docker
   ./ros1_run.sh
   ```

3. **Build the Workspace:**
   
   Inside the Docker container, execute:
   ```bash
   catkin build
   source devel/setup.bash
   ```
   
   *Note:* If you encounter a compilation error similar to `g++: internal compiler error: Killed (program cc1plus)`, it indicates that the compilation process is consuming too many resources. To mitigate this, rerun the compilation with:
   ```bash
   catkin build -j 1
   ```
   This command disables parallel building, reducing resource consumption.

4. **Launch the Simulation:**
   
   After a successful build, start the simulation:
   ```bash
   roslaunch trajectory_planner vehicle_guidance.launch
   ```
   
   The Control Toolbox employs automatic differentiation, which may cause a slight delay during the initialization of the trajectory planner. Once initialized, the terminal will display messages similar to:
   ```bash
   Initialization of Trajectory Planner done!
   Trajectory optimization SUCCESSFUL after [...]s.
   Trajectory optimization SUCCESSFUL after [...]s.
   ...
   ```

5. **Visualize in RViz:**
   
   The RViz visualization should resemble the image below:
   
   ![](../images/section_4/simulation_start.png)
   
   - **Blue Line:** Reference path for trajectory optimization.
   - **Red Lines:** Lane markings.
   
   *Note:* Initially, the vehicle remains stationary as the output trajectory (green line) is not yet implemented.

## Tasks

### Task 1: Implementation of the System Dynamics

**Objective:** Develop and integrate the system dynamics model for a car-like vehicle using a kinematic single-track model.

**System Model:**

The system dynamics are defined by the differential equation:
$$
\dot{\mathbf{x}}(t) = f(\mathbf{x}(t), \mathbf{u}(t), t)
$$

where:

$$
\mathbf{x} = \begin{pmatrix} x \\ y \\ s \\ v \\ a \\ \psi \\ \delta \end{pmatrix}, \quad
\mathbf{u} = \begin{pmatrix} j \\ \alpha \end{pmatrix}
$$

- **State Vector (\(\mathbf{x}\)):**
  - \(x, y\): Vehicle position coordinates.
  - \(s\): Traveled distance.
  - \(v\): Vehicle velocity.
  - \(a\): Longitudinal acceleration.
  - \(\psi\): Vehicle heading angle.
  - \(\delta\): Steering angle.

- **Control Vector (\(\mathbf{u}\)):**
  - \(j\): Longitudinal jerk.
  - \(\alpha\): Steering rate.

**Kinematic Single-Track Model Equations:**

Using the kinematic single-track (bicycle) model, the system dynamics can be derived as follows:

$$
\begin{aligned}
\dot{x} &= v \cdot \cos(\psi) \\
\dot{y} &= v \cdot \sin(\psi) \\
\dot{s} &= v \\
\dot{v} &= a \\
\dot{a} &= j \\
\dot{\psi} &= \frac{v}{l} \cdot \tan(\delta) \\
\dot{\delta} &= \alpha \\
\end{aligned}
$$

Where \(l\) is the vehicle's wheelbase.

**Implementation Steps:**

1. **Navigate to the Code:**
   
   Open the `trajectory_planner.cpp` file and locate line 179:
   [trajectory_planner.cpp at Line 179](https://github.com/ika-rwth-aachen/acdc/blob/main/catkin_workspace/src/workshops/section_4/trajectory_planner/src/trajectory_planner.cpp#L179)

2. **Fill in the Gaps:**
   
   ```cpp
   // START TASK 1 CODE HERE
   // System Dynamics
   // use helping comments from Wiki

   // System State Vector:
   // state(0): x -> Position X
   // state(1): y -> Position Y
   // state(2): s -> Distance
   // state(3): v -> Vehicle Velocity
   // state(4): a -> Vehicle Acceleration
   // state(5): psi -> Vehicle Heading
   // state(6): delta -> Steering Angle

   // Control Vector:
   // control(0): j_lon -> longitudinal jerk
   // control(1): alpha -> Steering Rate

   // The vehicles wheel-base is defined by the class variable wheelBase

   derivative(0) = x[3] * CppAD::cos(x[5]); // derivative of x
   derivative(1) = x[3] * CppAD::sin(x[5]); // derivative of y
   derivative(2) = x[3];                   // derivative of s
   derivative(3) = x[4];                   // derivative of v
   derivative(4) = control(0);            // derivative of a
   derivative(5) = (x[3] / wheelBase) * CppAD::tan(x[6]); // derivative of psi
   derivative(6) = control(1);            // derivative of delta
   // END TASK 1 CODE HERE
   ```

**Hints:**

- **Mathematical Functions:** Utilize `CppAD::cos`, `CppAD::sin`, and `CppAD::tan` for trigonometric computations.
- **Wheelbase Access:** The wheelbase (\(l\)) is accessible via the class variable `wheelBase`.
- **State and Control Vectors:** Access elements using `x[...]` for states and `control[...]` for controls.

3. **Build and Launch:**
   
   After implementing the system dynamics, save the changes and execute the following commands inside the `catkin_workspace`:
   ```bash
   catkin build trajectory_planner
   source devel/setup.bash
   roslaunch trajectory_planner vehicle_guidance.launch
   ```
   
   **Expected Outcome:**
   
   The RViz visualization should now display a green trajectory line representing the optimized path, and the vehicle should begin to move. However, initial runs may produce warnings related to high longitudinal jerk and steering rates:
   ```bash
   [ WARN] [...]: TrajectoryControl: Longitudinal jerk limited!
   [ WARN] [...]: TrajectoryControl: Steering-rate limited!
   ...
   ```

### Task 2: Penalize High Control and Lateral Jerk Values

**Objective:** Enhance the trajectory planner by adding cost terms that penalize high longitudinal and lateral jerk values, as well as high steering rates. This refinement ensures smoother and more realistic vehicle movements.

**Cost Function Terms to Implement:**

1. **Longitudinal Jerk Term (`jerkLonTerm`):** Penalizes abrupt changes in acceleration.
2. **Lateral Jerk Term (`jerkLatTerm`):** Penalizes abrupt changes in lateral acceleration.
3. **Steering Rate Term (`alphaTerm`):** Penalizes rapid steering adjustments.

**Implementation Steps:**

1. **Navigate to the Code:**
   
   Open the `trajectory_planner.cpp` file and locate line 75:
   [trajectory_planner.cpp at Line 75](https://github.com/ika-rwth-aachen/acdc/blob/main/catkin_workspace/src/workshops/section_4/trajectory_planner/src/trajectory_planner.cpp#L75)

2. **Fill in the Gaps:**
   
   ```cpp
   // START TASK 2 CODE HERE
   // use helping comments from Wiki and README.md
   
   // System State Vector:
   // x[0]: x -> Position X
   // x[1]: y -> Position Y
   // x[2]: s -> Distance
   // x[3]: v -> Vehicle Velocity
   // x[4]: a -> Vehicle Acceleration
   // x[5]: psi -> Vehicle Heading
   // x[6]: delta -> Steering Angle
   
   // Control Vector:
   // u[0]: j_lon -> longitudinal jerk
   // u[1]: alpha -> Steering Rate
   
   // if necessary use CppAD::sin(...), CppAD::cos(...), CppAD::tan(...), CppAD::pow(...), CppAD::sqrt(...)
   // Longitudinal jerk term
   SC jerkRef = x[MPC_NODE::WEIGHTS::JERK_REF];
   SC jerkLonCost = CppAD::pow(control(0) / jerkRef, 2);
   SC jerkLonWeight = x[MPC_NODE::WEIGHTS::JERK];
   SC jerkLonTerm = jerkLonCost * jerkLonWeight;
   
   // Alpha term
   SC alphaRef = x[MPC_NODE::WEIGHTS::ALPHA_REF];
   SC alphaCost = CppAD::pow(control(1) / alphaRef, 2);
   SC alphaWeight = x[MPC_NODE::WEIGHTS::ALPHA];
   SC alphaTerm = alphaCost * alphaWeight;
   
   // Lateral jerk term
   // The vehicles wheel-base is defined by the variable wheelBase
   double wheelBase = MPC_NODE::systemDynamics::wheelBase;
   SC a_lat = (x[3] * x[3] / wheelBase) * CppAD::tan(x[6]); // a_y = v^2 / l * tan(delta)
   SC jLat = (CppAD::pow(x[4], 2) / wheelBase) * CppAD::pow(1 + CppAD::pow(CppAD::tan(x[6]), 2), 2) + (CppAD::pow(x[3], 2) / wheelBase) * (1 + CppAD::pow(CppAD::tan(x[6]), 2)) * control(1);
   SC jerkLatCost = CppAD::pow(jLat / jerkRef, 2);
   SC jerkLatWeight = x[MPC_NODE::WEIGHTS::JERK];
   SC jerkLatTerm = jerkLatCost * jerkLatWeight;
   // END TASK 2 CODE HERE
   ```

**Detailed Explanation:**

- **Longitudinal Jerk (`jerkLonTerm`):**
  
$$
  \text{jerkLonCost} = \left(\frac{j_{\text{lon}}}{\text{jerkRef}}\right)^2
$$
  
$$
  \text{jerkLonTerm} = \text{jerkLonCost} \times \text{jerkLonWeight}
$$

- **Steering Rate (`alphaTerm`):**

$$
  \text{alphaCost} = \left(\frac{\alpha}{\alphaRef}\right)^2
$$
  
$$
  \text{alphaTerm} = \text{alphaCost} \times \text{alphaWeight}
$$

- **Lateral Jerk (`jerkLatTerm`):**
  
  The lateral acceleration $ a_y $ and its derivative (lateral jerk $ j_{\text{lat}} $ are calculated as:
  
$$
  a_y = \frac{v^2}{l} \cdot \tan(\delta)
$$
  
$$
  j_{\text{lat}} = \frac{\partial a_y}{\partial t} = \frac{2v \cdot a}{l} \cdot \tan(\delta) + \frac{v^2}{l} \cdot \frac{1}{\cos^2(\delta)} \cdot \alpha
$$
  
$$
  \text{jerkLatCost} = \left(\frac{j_{\text{lat}}}{\text{jerkRef}}\right)^2
$$
  
$$
  \text{jerkLatTerm} = \text{jerkLatCost} \times \text{jerkLatWeight}
$$

3. **Build and Launch:**
   
   After implementing the cost terms, save the changes and execute:
   ```bash
   catkin build trajectory_planner
   source devel/setup.bash
   roslaunch trajectory_planner vehicle_guidance.launch
   ```
   
   **Expected Outcome:**
   
   The vehicle should begin to move; however, it may operate at a reduced speed due to the absence of a velocity deviation cost term. Additionally, the terminal may display warnings regarding limited longitudinal jerk and steering rates, indicating the effectiveness of the newly added cost terms.

### Task 3: Implement Propulsive Velocity Term

**Objective:** Incorporate a cost term that penalizes deviations from a target velocity, ensuring the vehicle maintains a desired speed during trajectory execution.

**Implementation Steps:**

1. **Navigate to the Code:**
   
   Open the `trajectory_planner.cpp` file and locate line 113:
   [trajectory_planner.cpp at Line 113](https://github.com/ika-rwth-aachen/acdc/blob/main/catkin_workspace/src/workshops/section_4/trajectory_planner/src/trajectory_planner.cpp#L113)

2. **Fill in the Gaps:**
   
   ```cpp
   // START TASK 3 CODE HERE
   // Velocity Term
   // if necessary use CppAD::sin(...), CppAD::cos(...), CppAD::tan(...), CppAD::pow(...), CppAD::sqrt(...)
   SC vScale = CppAD::CondExpGt(x[3], SC(10.0 / 3.6), x[3], SC(10.0 / 3.6));
   SC vCost = (x[3] - velocity) / vScale;
   SC vWeight = x[MPC_NODE::WEIGHTS::VEL];
   SC velTerm = CppAD::pow(vCost * vWeight, 2);
   // END TASK 3 CODE HERE
   ```

**Detailed Explanation:**

- **Velocity Scaling (`vScale`):**
  
  Ensures that the scaling factor adapts based on the current velocity to maintain numerical stability:
  
$$
  vScale = \begin{cases}
  v & \text{if } v > \frac{10.0}{3.6} \\
  \frac{10.0}{3.6} & \text{otherwise}
  \end{cases}
$$
  
  Where \( v = x[3] \) is the vehicle's current velocity.

- **Velocity Cost (`vCost`):**
  
$$
  vCost = \frac{v - \text{velocity}}{vScale}
$$
  
  Penalizes the deviation from the target velocity.

- **Velocity Term (`velTerm`):**
  
$$
  velTerm = \left(vCost \times \text{vWeight}\right)^2
$$
  
  Integrates the scaled velocity deviation into the cost function.

3. **Build and Launch:**
   
   After implementing the velocity cost term, save the changes and execute:
   ```bash
   catkin build trajectory_planner
   source devel/setup.bash
   roslaunch trajectory_planner vehicle_guidance.launch
   ```
   
   **Expected Outcome:**
   
   The vehicle should now maintain the target velocity more effectively, moving smoothly along the planned trajectory.

4. **Introduce Dynamic Objects:**
   
   To test collision avoidance, play the pre-recorded bag file in a separate terminal within the Docker container:
   ```bash
   rosbag play acdc_fusion_guidance_noise.bag
   ```
   
   **Note:** Initially, the vehicle may not respond to dynamic objects, as the collision avoidance cost term is yet to be implemented.

### Task 4: Collision Avoidance with Dynamic Objects

**Objective:** Implement a cost term that enables the trajectory planner to avoid collisions with dynamic objects detected in the environment.

**Implementation Steps:**

1. **Navigate to the Code:**
   
   Open the `trajectory_planner.cpp` file and locate line 122:
   [trajectory_planner.cpp at Line 122](https://github.com/ika-rwth-aachen/acdc/blob/main/catkin_workspace/src/workshops/section_4/trajectory_planner/src/trajectory_planner.cpp#L122)

2. **Fill in the Gaps:**
   
   ```cpp
   // START TASK 4 CODE HERE
   // Dyn obj
   // if necessary use CppAD::sin(...), CppAD::cos(...), CppAD::tan(...), CppAD::pow(...), CppAD::sqrt(...)
   SC dynObjX = x[MPC_NODE::DYNOBJCOORDS::X];
   SC dynObjY = x[MPC_NODE::DYNOBJCOORDS::Y];
   SC dynObjRef = x[MPC_NODE::WEIGHTS::DYNOBJ_REF];
   SC dynObjDist = CppAD::sqrt(CppAD::pow(x[0] - dynObjX, 2) + CppAD::pow(x[1] - dynObjY, 2));
   SC dynObjCost = CppAD::CondExpLt(dynObjDist, dynObjRef, CppAD::cos(CppAD::Pi() * CppAD::pow(dynObjDist, 2) / CppAD::pow(dynObjRef, 2)) + 1, SC(0.0));
   SC dynObjWeight = x[MPC_NODE::WEIGHTS::DYNOBJ];
   SC dynObjTerm = dynObjCost * dynObjWeight;
   // END TASK 4 CODE HERE
   ```

**Detailed Explanation:**

- **Dynamic Object Distance (`dynObjDist`):**
  
  Calculates the Euclidean distance between the vehicle's current position \((x[0], x[1])\) and the dynamic object's position \((dynObjX, dynObjY)\):
  
$$
  dynObjDist = \sqrt{(x - dynObjX)^2 + (y - dynObjY)^2}
$$

- **Dynamic Object Cost (`dynObjCost`):**
  
  Applies a cosine-based penalty when the vehicle is within a reference distance `dynObjRef` of the dynamic object:
  
$$
  dynObjCost = \begin{cases}
  \cos\left(\pi \cdot \frac{dynObjDist^2}{dynObjRef^2}\right) + 1 & \text{if } dynObjDist < dynObjRef \\
  0 & \text{otherwise}
  \end{cases}
$$
  
  This formulation ensures a smooth penalty that intensifies as the vehicle approaches the dynamic object.

- **Dynamic Object Term (`dynObjTerm`):**
  
$$
  dynObjTerm = dynObjCost \times dynObjWeight
$$
  
  Integrates the dynamic object cost into the overall cost function with appropriate weighting.

3. **Build and Launch:**
   
   After implementing the collision avoidance cost term, save the changes and execute:
   ```bash
   catkin build trajectory_planner
   source devel/setup.bash
   roslaunch trajectory_planner vehicle_guidance.launch
   ```
   
   **Testing Collision Avoidance:**
   
   Play the bag file to simulate dynamic objects:
   ```bash
   rosbag play acdc_fusion_guidance_noise.bag
   ```
   
   **Expected Outcome:**
   
   The vehicle should now adjust its trajectory to avoid collisions with dynamic objects, demonstrating responsive and safe navigation.

## Result

Upon successful completion of all tasks, the trajectory planner will exhibit robust behavior, enabling the vehicle to:

- **Follow the Reference Path:** The vehicle adheres closely to the planned trajectory, maintaining the desired speed.
- **Smooth Movements:** High jerk and steering rate penalties ensure smooth acceleration, deceleration, and steering adjustments.
- **Collision Avoidance:** The vehicle proactively alters its path to avoid dynamic obstacles, enhancing safety and reliability.

**Visualization in RViz:**

![Solution](../images/section_4/sol_planning.gif)

*Observation:* The vehicle follows the green trajectory line accurately, responds to dynamic objects by altering its path, and maintains smooth motion throughout the simulation.

*Note:* Despite these improvements, occasional deviations may still occur if the Bi-Level-Stabilization guidance system lacks a compensatory feedback controller, which will be addressed in subsequent workshops.

## Bonus Task: Enhancing Trajectory Planner Parameters

**Objective:** Gain deeper insights into the trajectory planner's behavior by experimenting with real-time parameter adjustments using ROS's dynamic reconfigure capabilities.

**Implementation Steps:**

1. **Open Dynamic Reconfigure Interface:**
   
   In a new terminal within the Docker container, execute:
   ```bash
   source devel/setup.bash
   rosrun rqt_reconfigure rqt_reconfigure
   ```
   
   This command launches a graphical interface allowing real-time tuning of various trajectory planner parameters.

2. **Adjust Parameters:**
   
   In the `rqt_reconfigure` window, you can modify parameters such as:
   
   - **Controller Gains:**
     - Adjust `dv_P`, `dv_I`, `dv_D`, etc., to observe their impact on vehicle acceleration and speed control.
     - Setting all controller gains to zero disables feedback control, mimicking behavior from earlier workshops.
   
   - **Deviation Parameters:**
     - Modify `deviationMax...` parameters to alter the maximum allowable deviations before the MPC system reinitializes.
     - Setting these to zero transforms the planner into a pure MPC mode, reinitializing optimization at every step and disabling low-level stabilization.
   
   - **Cost Function Weights:**
     - Tweak weights and reference factors to balance different aspects of the cost function, such as path adherence versus smoothness.
   
3. **Observe Effects:**
   
   As you adjust parameters, monitor the vehicle's behavior in the RViz simulation:
   
   - **High Gains:** May lead to aggressive control actions, reducing tracking errors but potentially causing oscillations.
   - **Low Gains:** Results in more gradual adjustments, increasing tracking errors but enhancing stability.
   - **Deviation Limits:** Tight limits enforce stricter adherence to the trajectory, while relaxed limits allow for more flexibility.
   - **Cost Weights:** Balancing different cost terms affects the trade-off between speed, smoothness, and safety.

4. **Experiment and Learn:**
   
   Engage in iterative experimentation to understand how each parameter influences the trajectory planner's performance. Document your observations to build an intuitive understanding of optimal control tuning.

## Wrap-up

Through this workshop, participants have achieved the following:

- **Implemented System Dynamics:** Developed a kinematic single-track model to represent the vehicle's motion dynamics accurately.
- **Integrated Cost Terms:** Added and fine-tuned cost functions to penalize high jerk values, steering rates, and velocity deviations, ensuring smooth and efficient trajectory generation.
- **Enabled Collision Avoidance:** Implemented a dynamic object avoidance mechanism, enhancing the vehicle's ability to navigate safely in dynamic environments.
- **Leveraged Control Toolbox Features:** Utilized automatic differentiation and other advanced features of the Control Toolbox to optimize trajectory planning.
- **Enhanced Understanding Through Parameter Tuning:** Explored the effects of various planner parameters using dynamic reconfigure, deepening comprehension of optimal control dynamics.

These foundational skills are essential for developing sophisticated autonomous vehicle systems capable of navigating complex environments with precision and reliability.

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
  
  Flatland is a 2D robot simulation environment tailored for testing navigation and control algorithms, providing a flexible platform for validating trajectory planning and obstacle avoidance strategies.

- **ROS (Robot Operating System):** [https://www.ros.org](https://www.ros.org)
  
  ROS is a flexible framework for writing robot software, offering a collection of tools, libraries, and conventions that simplify the task of creating complex and robust robot behavior across a wide variety of robotic platforms.