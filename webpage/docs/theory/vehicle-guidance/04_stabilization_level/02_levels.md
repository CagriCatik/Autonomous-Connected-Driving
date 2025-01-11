# Low-, High-, and Bi-Level Stabilization in Vehicle Control

Vehicle stabilization approaches are essential for ensuring that autonomous vehicles adhere closely to their planned trajectories, minimizing deviations caused by various disturbances. This document delves into three primary stabilization methodologies—**low-level stabilization**, **high-level stabilization**, and **bi-level stabilization**—each tailored to address specific types of disturbances and operational scenarios within automated driving systems.

---

## Low-Level Stabilization

### Principles

In the **low-level stabilization** approach, the **trajectory planner** operates independently of the vehicle's current state. This method assumes that the **planned trajectory** is accurately followed by the low-level controllers responsible for vehicle actuation. Feedback mechanisms are applied directly at the **controller level** to minimize any deviations from the planned path.

### Characteristics

1. **Feedback Location**:
   - Feedback is confined to the low-level controller and does not influence the trajectory planner.
   
2. **Response to Permanent Disturbances**:
   - Effectively handles steady-state or **permanent disturbances** such as lateral road slopes or consistent crosswinds.
   
3. **Trajectory Reference**:
   - New trajectories are planned using the **previously planned trajectory** as a reference, rather than the actual vehicle state.

### Behavior Under Disturbances

- **Permanent Disturbances**:
  - The low-level controller ensures minimal deviation by making slight adjustments to the control output.
  
- **Sudden Disturbances**:
  - Reacts aggressively with **high control inputs** (e.g., sharp steering corrections) to realign the vehicle with the original trajectory.
  - May result in **uncomfortable maneuvers** due to large corrective actions.

---

## High-Level Stabilization

### Principles

The **high-level stabilization** approach integrates the **actual vehicle state** back into the **trajectory planner**, effectively transforming it into a **Model Predictive Controller (MPC)**. This integration allows the trajectory planner to generate new trajectories based on real-time vehicle states, facilitating continuous realignment with the target trajectory.

### Characteristics

1. **Feedback Location**:
   - Feedback is applied at the trajectory planning level.
   
2. **Response to Sudden Disturbances**:
   - Handles **sudden disturbances** more effectively, ensuring smooth realignment with the target trajectory.
   
3. **Trajectory Reference**:
   - Plans new trajectories using the **current vehicle state** as the reference.

### Behavior Under Disturbances

- **Permanent Disturbances**:
  - Generates **“dog curves”**, where the trajectory undergoes repetitive realignments due to cyclic corrections to the target position.
  
- **Sudden Disturbances**:
  - Smoothly adjusts the trajectory with **relatively small steering inputs**, enhancing passenger comfort.

---

## Bi-Level Stabilization

### Introduction

**Bi-level stabilization**, introduced by Werling, synergizes the strengths of both low- and high-level stabilization to overcome their individual limitations. This hybrid approach ensures robust and efficient handling of various disturbance types in autonomous driving.

### Principles

1. **Normal Operation**:
   - Operates primarily in a **low-level stabilized manner**, with the trajectory planner and controller functioning independently.
   - The controller minimizes deviations from the planned trajectory under normal conditions.
   
2. **High Deviation Handling**:
   - When deviations exceed a predefined threshold (e.g., lateral displacement \(d_y\) or heading error \(d_\theta\)), the system reinitializes the trajectory planner.
   - The **actual vehicle state** is fed back to the trajectory planner, switching the system to a **high-level stabilized mode** to generate a new trajectory.

### Characteristics

1. **Adaptive Feedback**:
   - Most of the time, the system operates in low-level stabilization.
   - High-level stabilization is activated only when significant deviations are detected.
   
2. **Resetting Deviations**:
   - Reinitialization ensures that deviations \(d_y\) (lateral offset) and \(d_\theta\) (heading error) are reset to zero, maintaining trajectory adherence.

---

## Behavior Comparison Under Disturbances

### Permanent Disturbances

- **Low-Level Stabilization**:
  - Maintains a shorter trajectory to the target with minimal deviation.
  - Implements slight control adjustments to compensate for steady-state disturbances.
  
- **High-Level Stabilization**:
  - Engages in repetitive trajectory adjustments (“dog curves”) due to continuous realignments with the target.
  
### Sudden Disturbances

- **Low-Level Stabilization**:
  - Executes aggressive corrections with high steering inputs.
  - Can lead to passenger discomfort due to abrupt maneuvers.
  
- **High-Level Stabilization**:
  - Adjusts the trajectory smoothly with relatively mild steering inputs.
  - Enhances passenger comfort by avoiding sharp corrective actions.

---

## Advantages and Limitations

### Low-Level Stabilization

- **Advantages**:
  - **Efficiency**: Effective for steady-state disturbances with minimal computational overhead.
  - **Simplicity**: Simplified trajectory planning process.
  
- **Limitations**:
  - **Handling Sudden Disturbances**: Poor performance in managing abrupt disturbances.
  - **Passenger Comfort**: Can result in uncomfortable corrective actions due to aggressive control inputs.

### High-Level Stabilization

- **Advantages**:
  - **Sudden Disturbance Management**: Excels in handling unexpected disturbances smoothly.
  - **Passenger Comfort**: Produces smoother and more comfortable trajectory adjustments.
  
- **Limitations**:
  - **Permanent Disturbances**: Inefficient in managing steady-state disturbances, leading to repetitive corrections.
  - **Computational Demand**: Higher computational resources required due to frequent trajectory replanning.

### Bi-Level Stabilization

- **Advantages**:
  - **Comprehensive Disturbance Handling**: Combines strengths of both low- and high-level stabilization.
  - **Efficiency and Comfort**: Efficiently manages both steady-state and sudden disturbances while ensuring passenger comfort.
  - **Adaptive Control**: Dynamically switches between stabilization modes based on deviation thresholds.
  
- **Limitations**:
  - **Complex Implementation**: Requires careful tuning and management of thresholds for reinitialization.
  - **System Complexity**: Increased complexity in the control system design and integration.

---

## Practical Example: Adaptive Bi-Level Stabilization

Below is a Python-based pseudocode implementation of a bi-level stabilization system, illustrating how the system adapts between low- and high-level stabilization based on deviation thresholds.

```python
class BiLevelStabilization:
    def __init__(self, planner, controller, deviation_threshold):
        """
        Initializes the bi-level stabilization system.
        
        Parameters:
        - planner: High-level trajectory planner instance.
        - controller: Low-level controller instance.
        - deviation_threshold: Threshold value for triggering high-level stabilization.
        """
        self.planner = planner  # High-level trajectory planner
        self.controller = controller  # Low-level controller
        self.deviation_threshold = deviation_threshold  # Threshold for reinitialization
        self.current_mode = "low-level"  # Default mode

    def compute_control(self, current_state, planned_trajectory):
        """
        Computes the control output based on the current state and planned trajectory.
        
        Parameters:
        - current_state: The current state of the vehicle (e.g., position, heading).
        - planned_trajectory: The trajectory planned by the guidance level.
        
        Returns:
        - control_output: The control signal to be applied to the vehicle actuators.
        """
        # Compute deviations
        dy, dtheta = self.compute_deviations(current_state, planned_trajectory)
        
        # Check if reinitialization is required
        if abs(dy) > self.deviation_threshold or abs(dtheta) > self.deviation_threshold:
            self.current_mode = "high-level"
            new_trajectory = self.planner.plan_trajectory(current_state)  # Replan trajectory
            dy, dtheta = 0, 0  # Reset deviations
        else:
            self.current_mode = "low-level"
            new_trajectory = planned_trajectory

        # Compute control signals
        control_output = self.controller.compute_control(current_state, new_trajectory)
        return control_output

    def compute_deviations(self, current_state, planned_trajectory):
        """
        Computes the lateral and heading deviations from the planned trajectory.
        
        Parameters:
        - current_state: The current state of the vehicle.
        - planned_trajectory: The planned trajectory.
        
        Returns:
        - dy: Lateral deviation.
        - dtheta: Heading deviation.
        """
        # Example deviation computation
        dy = current_state.y - planned_trajectory.y
        dtheta = current_state.theta - planned_trajectory.theta
        return dy, dtheta

# Example usage
if __name__ == "__main__":
    # Initialize planner and controller (pseudo-implementations)
    planner = TrajectoryPlanner()
    controller = LowLevelController()
    deviation_threshold = 0.5  # Example threshold values
    
    # Initialize bi-level stabilization system
    stabilization_system = BiLevelStabilization(planner, controller, deviation_threshold)
    
    # Current vehicle state and planned trajectory
    current_state = VehicleState(y=2.0, theta=5.0)
    planned_trajectory = Trajectory(y=0.0, theta=0.0)
    
    # Compute control output
    control_signal = stabilization_system.compute_control(current_state, planned_trajectory)
    print(f"Control Signal: {control_signal}")
```

*Note: The `TrajectoryPlanner`, `LowLevelController`, `VehicleState`, and `Trajectory` classes are placeholders and should be implemented based on specific system requirements.*

---

## Summary

Vehicle stabilization is crucial for maintaining adherence to planned trajectories in autonomous driving systems. The three stabilization methodologies—**low-level**, **high-level**, and **bi-level**—offer distinct approaches to managing different types of disturbances:

1. **Low-Level Stabilization**:
   - Best suited for handling steady-state disturbances with minimal computational overhead.
   - May struggle with sudden disturbances, leading to aggressive corrective actions that can compromise passenger comfort.

2. **High-Level Stabilization**:
   - Excels in managing sudden disturbances, ensuring smooth and comfortable trajectory realignments.
   - Inefficient for steady-state disturbances due to repetitive trajectory adjustments, resulting in higher computational demands.

3. **Bi-Level Stabilization**:
   - Combines the strengths of both low- and high-level approaches.
   - Provides adaptive and robust control, efficiently handling both steady-state and sudden disturbances while maintaining passenger comfort.
   - Requires careful system design and tuning to manage the transition between stabilization modes effectively.

By leveraging these stabilization strategies, engineers can design autonomous vehicle control systems that ensure precise trajectory adherence, enhancing both safety and user experience in dynamic driving environments.

---