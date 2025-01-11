# Mathematical Notation

Understanding the mathematical foundation is crucial for grasping the mechanics of object prediction using Kalman filters. This section elucidates the symbols, indices, and coordinate systems employed throughout the documentation, providing a clear and consistent framework for interpreting the subsequent equations and implementations.

## Symbols and Indices

Accurate representation of mathematical concepts relies on a well-defined set of symbols and indices. Below are the key symbols and indices used in this documentation:

### 1. Hat Symbol ($\hat{\cdot}$)

- **Definition**: Represents an estimated value.
- **Usage**: Denotes values derived through computation rather than direct measurement.
  
  **Example**:
  - $\hat{x}$: Estimated position along the x-axis.
  - $\hat{\mathbf{x}}$: Estimated state vector.

### 2. Transpose ($^T$)

- **Definition**: Denotes the transposed version of a vector or matrix.
- **Usage**: Used to switch the rows and columns of a matrix or to convert a column vector to a row vector and vice versa.
  
  **Example**:
  - $\mathbf{A}^T$: Transpose of matrix $\mathbf{A}$.
  - $\mathbf{x}^T$: Transpose of state vector $\mathbf{x}$.

### 3. Indices

Indices are used to differentiate between various levels of data and to specify the source or context of the data.

- **$G$ (Global Environment Model)**
  - **Definition**: Pertains to global-level data.
  - **Usage**: Represents data integrated into a unified global frame of reference.
  
- **$S$ (Sensor-Level Data)**
  - **Definition**: Refers to raw sensor measurements.
  - **Usage**: Represents data originating directly from individual sensors before any processing or fusion.

**Summary Table of Symbols and Indices**

| Symbol | Description                                 |
|--------|---------------------------------------------|
| $\hat{\cdot}$ | Estimated value                        |
| $^T$   | Transpose of a vector or matrix             |
| $G$    | Global Environment Model (global-level data)|
| $S$    | Sensor-Level Data (raw sensor measurements) |

## Reference Coordinate System

A consistent and well-defined coordinate system is essential for accurate measurement, prediction, and integration of sensor data within the autonomous driving framework. This section details the alignment, origin, and axes of the reference coordinate system used throughout the documentation.

### Alignment

- **Description**: The reference coordinate system is synchronized with the **ego vehicle** to ensure consistency in measurements and predictions.
- **Purpose**: Aligning the coordinate system with the ego vehicle simplifies the interpretation of sensor data relative to the vehicle's position and movement.

### Origin

- **Position**: Located at the **rear axle** of the vehicle.
- **Function**: Serves as the central reference point for all positional measurements and predictions.
  
  **Visualization**:

  ```
  Front of Vehicle
         |
         v
    ----------------
    |              |
    |              |
    |     Ego      |
    |    Vehicle   |
    |              |
    |              |
    ----------------
         ^
         |
    Rear Axle (Origin)
  ```

### Axes

The coordinate system is defined with two primary axes:

1. **$x$-axis (Longitudinal Axis)**
   - **Direction**: Extends **forward** relative to the vehicle.
   - **Positive Direction**: Forward direction of the vehicle.
   - **Usage**: Measures longitudinal positions and velocities.
   
2. **$y$-axis (Lateral Axis)**
   - **Direction**: Extends **to the left** of the vehicle.
   - **Positive Direction**: Left side of the vehicle.
   - **Usage**: Measures lateral positions and velocities.

**Coordinate System Diagram**

```plaintext
          y
          ^
          |
          |
          |
          |_____________> x
         /
        /
       / Vehicle
```

- **$x$-axis**: Forward direction.
- **$y$-axis**: Leftward direction.
- **Origin**: Rear axle of the vehicle.

### Importance of Alignment and Origin

- **Consistency**: Ensures that all sensor data and predictions are referenced to a common frame, reducing discrepancies and errors.
- **Simplified Calculations**: Facilitates straightforward mathematical operations and transformations between different data sources.
- **Integration**: Enhances the seamless integration of diverse sensor inputs into the global environment model.

## Summary

A robust understanding of the **mathematical notation** and the **reference coordinate system** is foundational for implementing effective object prediction using Kalman filters. By standardizing symbols, indices, and coordinate definitions, this documentation ensures clarity and consistency, enabling accurate interpretation and application of the prediction mechanisms within autonomous driving systems.

**Key Points:**

- **Symbols and Indices**:
  - Utilize the hat symbol ($\hat{\cdot}$) for estimated values.
  - Employ the transpose operator ($^T$) for vector and matrix transformations.
  - Distinguish between Global Environment Model ($G$) and Sensor-Level Data ($S$) using indices.

- **Reference Coordinate System**:
  - Align with the ego vehicle for consistency.
  - Set the origin at the rear axle for centralized reference.
  - Define $x$-axis as longitudinal and $y$-axis as lateral for clear positional measurements.

By adhering to these mathematical conventions and coordinate definitions, the subsequent sections on **State Description**, **Prediction Equations**, and **Integration with ROS** build upon a solid and unambiguous foundation, ensuring effective and reliable object prediction in automated driving scenarios.
