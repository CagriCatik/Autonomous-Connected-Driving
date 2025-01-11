# Object Fusion

In the realm of automated driving systems, accurate perception of the environment is paramount for safe and efficient navigation. **Object fusion**, also referred to as the **measurement update** in Kalman filters, plays a critical role in synthesizing data from multiple sensors to form a coherent understanding of surrounding objects. This process involves integrating **sensor-level objects** ($\hat{x}_S$)—the raw measurements from individual sensors—with their corresponding **global objects** ($\hat{x}_G$)—the predicted states maintained by the system.

The primary objective of object fusion is to optimally weigh measurements and predictions to minimize overall error covariances, thereby enhancing the precision of state estimates. This documentation provides a comprehensive exploration of the theory and implementation of object fusion within Kalman filters, tailored for both beginners and advanced practitioners in the field.

---

## Key Concepts

Understanding the foundational concepts is essential for effectively implementing object fusion in Kalman filters. This section elucidates the fundamental elements that underpin the fusion process.

### Sensor-Level and Global Objects

- **Sensor-Level Objects**: $\hat{x}_S$
  - **Definition**: Objects detected and measured directly by individual sensors (e.g., radar, lidar, camera).
  - **Characteristics**:
    - May provide a subset of the full state vector (e.g., position but not velocity).
    - Subject to sensor-specific noise and inaccuracies.
- **Global Objects ($\hat{x}_G$)**:
  - **Definition**: Predicted states maintained by the system based on prior estimates and system dynamics.
  - **Characteristics**:
    - Represents a holistic view of the object's state within the global environment model.
    - Continuously updated through prediction and measurement update steps of the Kalman filter.

### Error Covariances

- **Definition**: Matrices that quantify the uncertainty associated with state estimates.
- **Properties**:
  - **Inverse Relationship**: Higher error covariance implies lower confidence in the estimate.
  - **Matrix Representation**: Covariance matrices are symmetric and positive semi-definite.

### Measurement Matrix

- **Definition**: A matrix that maps the sensor-level state vector ($\hat{x}_S$) to the measurement space.
- **Function**:
  - Selects the relevant state variables that are observable by the sensor.
  - Facilitates the comparison between predicted global states and actual sensor measurements.
- **Example**:  $[z = C \hat{x}_S\]$

  Where:
  - $ z $ is the measurement vector.
  - $ C $ is the measurement matrix.

### Innovation

- **Definition**: The residual between the actual measurement and the predicted measurement. $\tilde{y}$
- **Calculation**:
  $$
      \[
      \tilde{y} = z - C \hat{x}_G
      \]
  $$
  Where:
  - $ z $ is the actual measurement.
  - $ \hat{x}\_G $ is the predicted global state.
- **Interpretation**:
  - Represents the new information provided by the sensor measurement.
  - Guides the update of the global state estimate.

### Innovation Covariance

- **Definition**: The covariance of the innovation, combining uncertainties from both prediction and measurement.
- **Calculation**:
  $$
      \[
      S = C P_G C^T + R
      \]
  $$
  Where:
  - $ P_G $ is the predicted global state covariance.
  - $ R $ is the measurement noise covariance.

### Kalman Gain

- **Definition**: A matrix that determines the weight given to the innovation in updating the state estimate.
- **Calculation**:
  $$
      \[
      K = P_G C^T S^{-1}
      \]
  $$
- **Role**:
  - Balances the trust between the predicted state and the new measurement.
  - Ensures optimal reduction of uncertainty in the updated state.

### State and Covariance Update

- **State Update**:

  $$
  \[
  \hat{x}_G' = \hat{x}_G + K \tilde{y}
  \]
  $$

  - **Explanation**: Adjusts the predicted global state by incorporating the innovation weighted by the Kalman Gain.

- **Covariance Update**:
  $$
  \[
  P_G' = (I - K C) P_G
  \]
  $$
  - **Explanation**: Updates the global state covariance to reflect the reduced uncertainty after incorporating the measurement.
  - **Property**: The matrix $ (I - K C) $ ensures that the updated covariance $ P_G' $ decreases, as its eigenvalues are always less than 1.

---

## Implementation Steps

Implementing object fusion within a Kalman filter involves a series of methodical steps. Below is a detailed guide accompanied by Python code snippets to facilitate understanding and application.

### 1. Mapping Measurement Variables

**Objective**: Define how sensor measurements relate to the global state vector through the measurement matrix $ C $.

```python
import numpy as np

# Define measurement matrix C
# Assuming the sensor measures position only (x and y)
C = np.array([
    [1, 0, 0, 0, 0, 0],  # Maps to x position
    [0, 1, 0, 0, 0, 0]   # Maps to y position
])

# Example sensor-level state vector x_S (6x1)
# [x, y, v_x, v_y, a_x, a_y]^T
x_S = np.array([[x1], [x2], [x3], [x4], [x5], [x6]])

# Map sensor-level state to measurement space
z = np.dot(C, x_S)  # Measurement vector z (2x1)
```

**Explanation**:

- The measurement matrix $ C $ selects the relevant components from the global state vector that the sensor can observe.
- In this example, the sensor measures only the $ x $ and $ y $ positions of an object.

### 2. Calculating Innovation

**Objective**: Determine the discrepancy between the actual measurement and the predicted measurement.

```python
# Predicted global state vector x_G (6x1)
x_G = np.array([[gx1], [gy1], [gvx1], [gvy1], [gax1], [gay1]])

# Predicted measurement based on global state
z_pred = np.dot(C, x_G)  # Predicted measurement (2x1)

# Innovation (residual)
y_tilde = z - z_pred  # Innovation vector (2x1)
```

**Explanation**:

- $ z\_{\text{pred}} $ is the predicted measurement derived from the global state estimate.
- $ \tilde{y} $ captures the new information introduced by the actual measurement.

### 3. Computing Innovation Covariance

**Objective**: Quantify the uncertainty associated with the innovation.

```python
# Predicted global state covariance matrix P_G (6x6)
P_G = np.array([
    [p11, p12, p13, p14, p15, p16],
    [p21, p22, p23, p24, p25, p26],
    [p31, p32, p33, p34, p35, p36],
    [p41, p42, p43, p44, p45, p46],
    [p51, p52, p53, p54, p55, p56],
    [p61, p62, p63, p64, p65, p66]
])

# Measurement noise covariance matrix R (2x2)
R = np.array([
    [r11, r12],
    [r21, r22]
])

# Compute innovation covariance S
S = np.dot(C, np.dot(P_G, C.T)) + R  # S is (2x2)
```

**Explanation**:

- $ S $ combines the uncertainties from the predicted global state and the measurement noise.
- A larger $ S $ indicates higher uncertainty in the innovation.

### 4. Calculating Kalman Gain

**Objective**: Determine the optimal weighting between the predicted state and the new measurement.

```python
# Compute Kalman Gain K (6x2)
K = np.dot(P_G, np.dot(C.T, np.linalg.inv(S)))
```

**Explanation**:

- $ K $ determines how much the innovation should influence the updated state estimate.
- It balances the trust between the prediction and the new measurement based on their respective uncertainties.

### 5. Updating State and Covariance

**Objective**: Refine the global state estimate and reduce uncertainty based on the new measurement.

```python
# Update global state estimate x_G (6x1)
x_G_updated = x_G + np.dot(K, y_tilde)  # Updated state vector (6x1)

# Identity matrix I (6x6)
I = np.eye(len(P_G))

# Update global state covariance P_G (6x6)
P_G_updated = np.dot((I - np.dot(K, C)), P_G)
```

**Explanation**:

- The global state vector $ \hat{x}\_G $ is updated by adding the weighted innovation.
- The covariance matrix $ P_G $ is adjusted to reflect the reduced uncertainty after the update.

---

## Practical Considerations

Implementing object fusion effectively requires attention to various practical aspects to ensure robustness and accuracy.

### Sensor Characteristics

- **Measurement Matrix ($ C $) and Noise Covariance ($ R $)**:
  - Vary based on sensor types (e.g., radar, lidar, camera).
  - Each sensor may measure different subsets of the state vector and have distinct noise profiles.
- **Example**:
  - **Radar**: May provide measurements for position and velocity.
  - **Lidar**: Typically offers precise position measurements but limited velocity information.

### Trustworthiness of Measurements

- **Kalman Gain Adjustment**:
  - The Kalman Gain $ K $ inherently adjusts the influence of measurements based on their uncertainty.
  - Higher measurement noise ($ R $) results in lower trust in the measurement, giving more weight to the prediction.
- **Dynamic Environments**:
  - In rapidly changing environments, sensor reliability may vary, necessitating adaptive tuning of $ R $.

### Edge Cases

- **Perfect Prediction ($ P_G \to 0 $)**:
  - No update is necessary as the prediction is already accurate.
- **Perfect Measurement ($ R \to 0 $)**:
  - The measurement is trusted entirely, overriding the prediction.
- **Numerical Stability**:
  - Ensure that matrices, especially $ S $, are invertible to avoid computational errors.

### Parameter Tuning

- **Relative Magnitudes of $ P_G $ and $ R $**:
  - Critical for balancing prediction and measurement influence.
  - Requires empirical testing or simulation-based optimization.
- **Process Noise ($ Q $)**:
  - Represents model uncertainties and influences the prediction step.
  - Tuning $ Q $ affects how responsive the filter is to changes.

---

## Advanced Topics

For those seeking to delve deeper into Kalman filters and object fusion, exploring advanced variations can offer enhanced performance in complex scenarios.

### Extended Kalman Filter (EKF)

- **Purpose**: Handles nonlinear relationships between state variables and measurements.
- **Approach**:
  - Linearizes nonlinear functions around the current estimate using Jacobian matrices.
- **Application**:
  - Useful in scenarios where sensor measurements or system dynamics are inherently nonlinear.

### Unscented Kalman Filter (UKF)

- **Purpose**: Provides better accuracy in handling nonlinearities compared to EKF without the need for explicit linearization.
- **Approach**:
  - Utilizes the Unscented Transform to propagate a set of sigma points through nonlinear functions.
- **Advantages**:
  - More accurate in capturing the mean and covariance of the transformed distribution.
  - Avoids the potential inaccuracies introduced by linearization in EKF.

---

## Conclusion

Object fusion within Kalman filters is a cornerstone of sensor data processing in automated driving systems. By adeptly combining sensor-level measurements with predicted global states, object fusion enhances the accuracy and reliability of object tracking and environment modeling. This fusion process not only mitigates individual sensor uncertainties but also synthesizes disparate data sources into a coherent and actionable state estimate.

**Key Takeaways**:

- **Mathematical Foundations**: Mastery of state vectors, covariance matrices, and the Kalman filter equations is essential for effective object fusion.
- **Implementation Mechanics**: Methodical execution of mapping, innovation calculation, covariance computation, Kalman Gain determination, and state updating ensures robust fusion outcomes.
- **Integration with Systems**: Seamless incorporation of object fusion into broader automated driving frameworks demands attention to sensor characteristics, parameter tuning, and error handling.

By leveraging the principles and methodologies outlined in this documentation, practitioners can implement sophisticated object fusion mechanisms, paving the way for safer and more efficient autonomous navigation systems.
