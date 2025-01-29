# Object Description

This section outlines the fundamental components involved in object prediction, namely the **State Vector** and the **Error Covariance Matrix**. Understanding these components is essential for implementing the Kalman filter effectively, as they form the basis for estimating and quantifying the state of tracked objects within the global environment model.

## 1. State Vector ($\mathbf{x}_{\hat{G}}$)

The **state vector** encapsulates all the properties of an object that are estimated by the Kalman filter. It provides a comprehensive representation of the object's current state within the global environment model, enabling accurate predictions of its future positions and movements.

### Components of the State Vector

The state vector is typically defined as follows:



$$
\mathbf{x}_{\hat{G}} = 
\begin{bmatrix}
x \\
y \\
v_x \\
v_y \\
a_x \\
a_y
\end{bmatrix}
$$


Where:
- **Position**:
  -  $x$ : Longitudinal position relative to the ego vehicle.
  -  $y$ : Lateral position relative to the ego vehicle.
- **Velocity**:
  -  $v_x$: Longitudinal velocity.
  -  $v_y$ : Lateral velocity.
- **Acceleration**:
  -  $a_x$ : Longitudinal acceleration.
  -  $a_y$ : Lateral acceleration.
- **Dimensions**:
  - **Width**: The lateral size of the object.
  - **Height**: The vertical size of the object.

**Note**: The **heading angle** of the object is calculated externally and is not incorporated within this state vector. This separation allows for more flexible integration with other system components that may handle orientation separately.

### Explanation of Components

- **Position ( $x$, $y$ )**:
  - Represents the current location of the object in the global environment model.
  - Critical for determining the object's proximity to the ego vehicle and potential collision paths.
  
- **Velocity ( $v_x$, $v_y$ )**:
  - Indicates the speed and direction of the object's movement.
  - Essential for predicting future positions based on current motion.
  
- **Acceleration ( $a_x$, $a_y$ )**:
  - Captures changes in the object's velocity over time.
  - Allows the Kalman filter to adapt to dynamic movements, such as speeding up, slowing down, or changing direction.

- **Dimensions (Width and Height)**:
  - Provide information about the size of the object.
  - Useful for object classification and determining the extent of the area the object occupies.

### Initialization of the State Vector

Proper initialization of the state vector is crucial for the Kalman filter's performance. Initially, the state vector may be set based on the first available sensor measurements, with assumptions made for unmeasured states (e.g., setting velocities and accelerations to zero if not directly observable).

```python
import numpy as np

# Initialize state vector (6x1 matrix)
x_hat = np.zeros((6, 1))  # [x, y, v_x, v_y, a_x, a_y]^T

# Example initialization based on first measurement
x_hat[0, 0] = initial_x  # x position
x_hat[1, 0] = initial_y  # y position
# v_x, v_y, a_x, a_y remain initialized to zero
```

## 2. Error Covariance Matrix ($\mathbf{P}$)

The **error covariance matrix** quantifies the uncertainty associated with each element of the state vector. It provides a measure of confidence in the estimated state, facilitating informed decision-making during data fusion and object association. Proper management of the covariance matrix is essential for maintaining the Kalman filter's effectiveness in various scenarios.

### Characteristics of the Error Covariance Matrix

$$
\mathbf{P} = 
\begin{bmatrix}
P_{xx} & P_{xy} & P_{x v_x} & P_{x v_y} & P_{x a_x} & P_{x a_y} \\
P_{yx} & P_{yy} & P_{y v_x} & P_{y v_y} & P_{y a_x} & P_{y a_y} \\
P_{v_x x} & P_{v_x y} & P_{v_x v_x} & P_{v_x v_y} & P_{v_x a_x} & P_{v_x a_y} \\
P_{v_y x} & P_{v_y y} & P_{v_y v_x} & P_{v_y v_y} & P_{v_y a_x} & P_{v_y a_y} \\
P_{a_x x} & P_{a_x y} & P_{a_x v_x} & P_{a_x v_y} & P_{a_x a_x} & P_{a_x a_y} \\
P_{a_y x} & P_{a_y y} & P_{a_y v_x} & P_{a_y v_y} & P_{a_y a_x} & P_{a_y a_y}
\end{bmatrix}
$$



Where each $ P_{ij} $ represents the covariance between state variables $ i $ and $ j $.

### Interpretation

- **Diagonal Elements $( P_{ii} )$**:
  - Represent the variance of each corresponding state variable.
  - Indicate the degree of uncertainty in each estimate.
  - **Smaller Values**: Higher confidence and lower uncertainty.
  - **Larger Values**: Greater uncertainty and lower confidence.
  
- **Off-Diagonal Elements $( P_{ij} $, $ i \neq j )$**:
  - Capture the covariance between different state variables.
  - Reflect the interdependencies and correlations between estimates.
  - **Positive Values**: Indicate that an increase in one variable is associated with an increase in another.
  - **Negative Values**: Indicate that an increase in one variable is associated with a decrease in another.

### Example of an Initial Error Covariance Matrix

An initial covariance matrix is often set to represent high uncertainty in the state estimates before any measurements are incorporated.

$$
\mathbf{P}_0 = 
\begin{bmatrix}
1000 & 0 & 0 & 0 & 0 & 0 \\
0 & 1000 & 0 & 0 & 0 & 0 \\
0 & 0 & 1000 & 0 & 0 & 0 \\
0 & 0 & 0 & 1000 & 0 & 0 \\
0 & 0 & 0 & 0 & 1000 & 0 \\
0 & 0 & 0 & 0 & 0 & 1000 \\
\end{bmatrix}
$$

```python
import numpy as np

# Initialize error covariance matrix (6x6)
P = np.eye(6) * 1000  # High initial uncertainty
```

### Updating the Error Covariance Matrix

During the prediction and update steps of the Kalman filter, the error covariance matrix is updated to reflect the propagated uncertainty and the influence of new measurements.

#### Prediction Step

$$
\mathbf{P}[k] = \mathbf{F} \cdot \mathbf{P}[k-1] \cdot \mathbf{F}^T + \mathbf{Q}
$$

- **$ \mathbf{F} $**: Motion model matrix.
- **$ \mathbf{Q} $**: Process noise covariance matrix.

#### Update Step

$$
\mathbf{P}[k|k] = (\mathbf{I} - \mathbf{K} \cdot \mathbf{H}) \cdot \mathbf{P}[k|k-1]
$$

- **$ \mathbf{K} $**: Kalman Gain.
- **$ \mathbf{H} $**: Measurement matrix.
- **$ \mathbf{I} $**: Identity matrix.

### Importance in the Kalman Filter

The error covariance matrix plays a critical role in the Kalman filter by:

- **Determining the Kalman Gain ($ \mathbf{K} $)**:
  - The Kalman Gain balances the trust between the prediction and the new measurements.
  - High uncertainty in the prediction $( \mathbf{P} )$ leads to higher Kalman Gain, giving more weight to the measurements.
  
- **Assessing Filter Confidence**:
  - The magnitude of the covariance elements indicates the filter's confidence in each state estimate.
  - Allows for adaptive tuning and reliability assessment of the tracking system.

### Example: Updating the Error Covariance Matrix in Python

```python
import numpy as np

# Previous error covariance matrix (6x6)
P_prev = np.eye(6) * 1000  # Example initialization

# Motion model matrix (F)
F = np.array([
    [1, 0, delta_t, 0, 0, 0],
    [0, 1, 0, delta_t, 0, 0],
    [0, 0, 1, 0, delta_t, 0],
    [0, 0, 0, 1, 0, delta_t],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1]
])

# Process noise matrix (Q)
Q = q * np.array([
    [delta_t**4 / 4, 0, delta_t**3 / 2, 0, delta_t**2 / 2, 0],
    [0, delta_t**4 / 4, 0, delta_t**3 / 2, 0, delta_t**2 / 2],
    [delta_t**3 / 2, 0, delta_t**2, 0, delta_t, 0],
    [0, delta_t**3 / 2, 0, delta_t**2, 0, delta_t],
    [delta_t**2 / 2, 0, delta_t, 0, 1, 0],
    [0, delta_t**2 / 2, 0, delta_t, 0, 1]
])

# Covariance prediction
P_pred = np.dot(F, np.dot(P_prev, F.T)) + Q

print("Predicted Error Covariance Matrix:")
print(P_pred)
```

**Output:**
```
Predicted Error Covariance Matrix:
[[1000.01    0.     ...]
 [    0. 1000.01    ...]
 ...
]
```

## Summary

The **State Vector** and the **Error Covariance Matrix** are foundational elements in the Kalman filter-based object prediction framework. The state vector provides a detailed representation of an object's current state, while the error covariance matrix quantifies the uncertainty associated with each state estimate. Together, they enable the Kalman filter to perform accurate and reliable predictions, essential for effective object tracking and data fusion in automated driving systems.

**Key Points:**

- **State Vector**:
  - Encapsulates position, velocity, acceleration, and dimensions of the object.
  - Essential for representing the object's current and future states.

- **Error Covariance Matrix**:
  - Quantifies uncertainty in the state estimates.
  - Crucial for determining the Kalman Gain and assessing filter confidence.

Understanding and accurately implementing these components are critical for the success of object prediction mechanisms, ensuring that autonomous systems can navigate their environments safely and efficiently.

