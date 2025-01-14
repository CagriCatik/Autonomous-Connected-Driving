# Trajectory Control Using Feedback PID Controllers

Trajectory control at the stabilization level is a cornerstone of autonomous driving systems, ensuring that vehicles accurately follow pre-planned paths with minimal deviation. This is achieved through the integration of feedback **PID controllers** with feedforward controls. The primary objective is to calculate and correct deviations in both longitudinal and lateral movements, maintaining precise adherence to the desired trajectory. This is implemented using a **closed-loop simulation** within the ROS (Robot Operating System) framework, leveraging odometry calculations, trajectory data interpolation, and sophisticated control system design.

---
    
## Key Components of the Trajectory Control System

Understanding the architecture of the trajectory control system is essential for implementing effective vehicle stabilization. The system is composed of three main modules:

### 1. Trajectory Data Processing Module

- **Odometry Calculations**: Computes the vehicle's current position, orientation, velocity, and acceleration based on sensor data such as wheel encoders and inertial measurement units (IMUs).
- **Trajectory Interpolation**: Processes the pre-planned trajectory to generate smooth and continuous target values for control inputs.
- **Control Input Provision**: Supplies the longitudinal and lateral controllers with the necessary target values derived from the trajectory data.

### 2. Longitudinal Controller

- **Feedforward Control**: Utilizes the target acceleration provided by the trajectory to predictively adjust the vehicle's speed.
- **Feedback Control**: Employs a PID controller to correct any deviations between the desired velocity and the actual velocity, ensuring accurate speed maintenance.

### 3. Lateral Controller

- **Feedforward Control**: Implements the Ackermann steering model and trajectory curvature to determine the desired steering angles based on the planned path.
- **Cascaded Feedback Control**: Utilizes a two-level PID controller system to manage yaw rate and steering angle, correcting any lateral deviations from the trajectory.

---
    
## Odometry Calculations

Odometry is fundamental for trajectory control, providing real-time estimates of the vehicle's state. Accurate odometry ensures that the control system can effectively minimize deviations from the planned trajectory.

### Current Position and Orientation

- **Position Calculation**:
  
  The vehicle's position in the 2D plane is updated using the following equations:
  
$$
x(t+1) = x(t) + v(t) \cdot \cos\left(\theta(t)\right) \cdot \Delta t
$$

$$
y(t+1) = y(t) + v(t) \cdot \sin\left(\theta(t)\right) \cdot \Delta t
$$

  
- **Orientation (Heading) Calculation**:
  
  The vehicle's orientation is updated based on the yaw rate:
  
  $$
  \theta(t+1) = \theta(t) + \omega(t) \cdot \Delta t
  $$
  
  Where:
  
  - $ x(t), y(t) $: Position coordinates at time $ t $.
  - $ v(t) $: Linear velocity.
  - $ \theta(t) $: Orientation angle.
  - $ \omega(t) $: Angular velocity (yaw rate).
  - $ \Delta t $: Time step duration.

### Velocity and Acceleration

- **Velocity Calculation**:
  
  The actual velocity is derived from the rate of change of position:
  
   $$
   v(t) = \sqrt{\left( \frac{dx(t)}{dt} \right)^2 + \left( \frac{dy(t)}{dt} \right)^2}
   $$

- **Acceleration Calculation**:
  
  Acceleration is determined by the change in velocity over time:
  
  $$
  a(t) = \frac{v(t) - v(t-1)}{\Delta t}
  $$
  
  Accurate calculation of velocity and acceleration is crucial for responsive and stable control actions.

---
    
## Longitudinal Controller

The longitudinal controller is responsible for maintaining the vehicle's speed as per the planned trajectory. It integrates both feedforward and feedback control mechanisms to achieve precise velocity tracking.

### Feedforward Control

- **Target Acceleration Utilization**:
  
  The feedforward component uses the target acceleration from the trajectory to anticipate and apply the necessary throttle or brake inputs. This proactive approach helps in minimizing the lag between desired and actual velocity changes.

### Feedback Control: PID Velocity Controller

- **PID Control Equation**:
  
  The PID controller calculates the control signal based on the error between desired and actual velocity:
  
  $$
  u(t) = K_p \cdot e(t) + K_i \cdot \int e(t) \, dt + K_d \cdot \frac{d e(t)}{dt}
  $$
  
  Where:
  
  - $u(t)$: Control output (e.g., throttle or brake signal).
  - $e(t)$ = $v_{desired}$ - $v_{actual}$: Velocity error.
  - $K_p$, $K_i$, $K_d$: Proportional, integral, and derivative gains, respectively.

- **Implementation Example**:
  
  ```python
  class PIDController:
      def __init__(self, kp, ki, kd):
          self.kp = kp
          self.ki = ki
          self.kd = kd
          self.previous_error = 0
          self.integral = 0

      def compute_control(self, desired, actual, dt):
          error = desired - actual
          self.integral += error * dt
          derivative = (error - self.previous_error) / dt
          self.previous_error = error
          return self.kp * error + self.ki * self.integral + self.kd * derivative

  # Example Usage
  pid_velocity = PIDController(kp=0.5, ki=0.1, kd=0.05)
  control_signal = pid_velocity.compute_control(v_desired, v_actual, dt)
  ```
  
  *Explanation:* The PID controller adjusts the throttle or brake inputs to minimize the velocity error, ensuring the vehicle follows the desired speed profile accurately.

---
    
## Lateral Controller

The lateral controller ensures that the vehicle maintains its position within the lane and follows the planned path by adjusting the steering angle. It combines feedforward and cascaded feedback control strategies for effective trajectory adherence.

### Feedforward Control: Ackermann Steering Model

- **Desired Yaw Rate Calculation**:
  
  The desired yaw rate $ \omega_d $ is determined by the vehicle's velocity $ v $ and the trajectory curvature $ \kappa $:
  
  $$
  \omega_d = v \cdot \kappa
  $$
  
- **Steering Angle Derivation**:
  
  The steering angle $\delta $ is computed using the inverse single-track model:
  
   $$
   \delta = \arctan\left( \frac{L \cdot \omega_d}{v} \right)
   $$
   
  Where $ L $ is the wheelbase of the vehicle.

### Cascaded Feedback Control: Two-Level PID Controller

1. **Outer Loop (Yaw Rate Control)**:
   - **Objective**: Minimize the lateral error $ e_y $) by computing the desired yaw rate.
   - **PID Controller**: Adjusts the yaw rate based on the lateral deviation from the trajectory.

2. **Inner Loop (Steering Control)**:
   - **Objective**: Convert the desired yaw rate into an actual steering angle.
   - **PID Controller**: Controls the steering actuators to achieve the desired yaw rate, ensuring smooth lateral movements.

### Implementation Example

```python
class CascadedLateralController:
    def __init__(self, kp_outer, ki_outer, kd_outer, kp_inner, ki_inner, kd_inner, wheelbase):
        self.pid_yaw_rate = PIDController(kp_outer, ki_outer, kd_outer)
        self.pid_steering = PIDController(kp_inner, ki_inner, kd_inner)
        self.wheelbase = wheelbase

    def compute_steering(self, lateral_error, yaw_rate_error, velocity, dt):
        # Outer loop: Compute desired yaw rate
        desired_yaw_rate = self.pid_yaw_rate.compute_control(0, lateral_error, dt)
        # Inner loop: Compute steering angle from yaw rate
        steering_angle = self.pid_steering.compute_control(
            desired_yaw_rate, yaw_rate_error, dt
        )
        return steering_angle

# Example Usage
controller = CascadedLateralController(
    kp_outer=0.4, ki_outer=0.1, kd_outer=0.05,
    kp_inner=0.3, ki_inner=0.1, kd_inner=0.05,
    wheelbase=2.5
)
steering_signal = controller.compute_steering(e_y, e_yaw, v_actual, dt)
```

*Explanation:* This cascaded PID controller first adjusts the yaw rate to correct the lateral error and then translates the desired yaw rate into a steering angle, ensuring the vehicle remains aligned with the planned trajectory.

---
    
## Practical Task Integration in ROS Framework

Implementing trajectory control within the ROS framework involves integrating various modules to create a cohesive and responsive control system. The following outlines the practical steps for achieving this integration.

### Task Breakdown

1. **Odometry Equations**:
   - Implement odometry calculations to determine the vehicle's current state $x$, $y$, $\theta$, $v, $a$.
   
2. **Feedback Controller Implementation**:
   - Develop PID controllers for both longitudinal velocity and lateral yaw rate.
   
3. **Integration of Feedforward and Feedback Controls**:
   - Combine feedforward controls (from trajectory data) with feedback controls (PID controllers) to manage longitudinal and lateral movements.
   
4. **Closed-Loop Simulation**:
   - Integrate the controllers into a ROS-based simulation environment to test and validate trajectory-following performance.

### ROS Workflow

- **Trajectory Data Processing Module**:
  - **Function**: Processes incoming trajectory data, performs odometry calculations, and interpolates target values.
  - **Output**: Publishes interpolated trajectory points and control inputs to the respective controllers.
  
- **Controller Nodes**:
  - **Function**: Subscribe to trajectory data and vehicle state topics.
  - **Output**: Compute and publish control commands to the vehicle actuators based on feedback and feedforward inputs.
  
- **Actuator Simulation**:
  - **Function**: Simulates the vehicle's response to control commands, updating the vehicle's state accordingly.

### Implementation Steps

1. **Setup ROS Environment**:
   - Install necessary ROS packages and dependencies.
   - Configure ROS nodes for trajectory processing, controllers, and actuator simulation.
   
2. **Develop Odometry Node**:
   - Implement the odometry equations to calculate the vehicle's current state.
   - Publish the state information to relevant ROS topics.
   
3. **Implement Controller Nodes**:
   - **Longitudinal Controller Node**:
     - Subscribe to desired velocity and actual velocity topics.
     - Compute throttle or brake signals using the PID velocity controller.
     - Publish control commands to the actuator simulation.
   
   - **Lateral Controller Node**:
     - Subscribe to desired steering angle and actual yaw rate topics.
     - Compute steering signals using the cascaded PID controller.
     - Publish control commands to the actuator simulation.
   
4. **Integrate Feedforward Controls**:
   - Use trajectory-provided acceleration for longitudinal control.
   - Apply trajectory curvature for lateral control using the Ackermann steering model.
   
5. **Conduct Closed-Loop Simulation**:
   - Run the ROS simulation, allowing the controllers to receive trajectory data and vehicle state updates.
   - Observe and refine the control signals to ensure accurate trajectory following.

---
    
## Summary

Trajectory control at the stabilization level is vital for the precise navigation of autonomous vehicles. By integrating **feedback PID controllers** with **feedforward controls**, the system ensures that the vehicle accurately follows a pre-planned trajectory, effectively correcting deviations in both longitudinal and lateral movements. The implementation within a **closed-loop simulation** using the ROS framework leverages odometry calculations, trajectory data interpolation, and robust control system design to achieve reliable and smooth vehicle control.

### Key Takeaways:

1. **Integrated Control Strategies**:
   - Combining feedforward and feedback controls enhances the system's ability to anticipate and correct deviations proactively and reactively.
   
2. **PID Controllers**:
   - **Longitudinal PID Controller**: Maintains desired velocity by minimizing velocity errors through throttle and brake adjustments.
   - **Cascaded Lateral PID Controller**: Ensures accurate path following by controlling yaw rate and steering angles based on lateral deviations.
   
3. **Odometry and State Estimation**:
   - Accurate odometry calculations are fundamental for determining the vehicle's current state, enabling precise control actions.
   
4. **ROS Framework Integration**:
   - Utilizing ROS allows for modular and scalable implementation of trajectory control systems, facilitating real-time simulations and validations.
   
5. **Closed-Loop Simulation**:
   - Enables comprehensive testing of control strategies in a simulated environment, ensuring robustness and reliability before real-world deployment.

By mastering these components and their integration within the ROS framework, engineers can design sophisticated trajectory control systems that enhance the performance, safety, and comfort of autonomous vehicles in dynamic driving environments.

---