# Introduction to Localization

Localization is a critical component in the realm of automated vehicles, enabling them to understand and navigate their environment effectively. While perception systems allow these vehicles to interpret their surroundings, localization ensures that the vehicle accurately determines its own position and orientation within that environment. This documentation delves into the fundamentals of localization, its significance in vehicle guidance, various localization methods, and the sensors that facilitate this essential task.

---

## Understanding Pose

At the heart of localization lies the concept of **pose**, which encapsulates both the position and orientation of a vehicle within a reference system.

### 2D Pose

A **2D pose** comprises three primary components:

1. **Translation in the X Direction**: Represents the vehicle's position along the longitudinal axis.
2. **Translation in the Y Direction**: Represents the vehicle's position along the lateral axis.
3. **Rotation around the Z-Axis (Yaw Angle)**: Denoted as ψ (psi), it indicates the vehicle's orientation relative to the reference system.

For many applications, a 2D pose suffices since vehicles predominantly operate along the X, Y, and ψ dimensions.

### 3D Pose

A **3D pose** extends the 2D pose by adding three more dimensions:

1. **Translation in the Z Direction**: Accounts for vertical movements.
2. **Rotation around the Longitudinal Axis (Roll Angle)**: Represents tilting to the left or right.
3. **Rotation around the Lateral Axis (Pitch Angle)**: Represents tilting forward or backward.

In total, a 3D pose includes:

- **Translations**: X, Y, Z
- **Rotations**: Roll, Pitch, Yaw

While 2D poses are adequate for many scenarios, 3D poses provide a more comprehensive understanding, especially in complex environments where vertical movements and tilts are significant.

---

## Importance of Localization in Automated Vehicles

Localization is indispensable for the effective guidance and navigation of automated vehicles. It provides the foundational data required for planning and executing driving maneuvers. Understanding its role across different levels of vehicle guidance underscores its multifaceted importance.

### Levels of Vehicle Guidance

Vehicle guidance can be segmented into three distinct levels, each imposing unique localization requirements:

#### Navigation Level

- **Objective**: Determine a route within the road network from the current position to a desired destination.
- **Localization Requirement**: Accurate knowledge of the vehicle's position within the road network is paramount.

#### Guidance Level

- **Objective**: Compute optimal driving maneuvers considering environmental constraints.
- **Localization Requirement**: Detailed understanding of the vehicle's pose, including lane position, to facilitate maneuvers like overtaking.

#### Stabilization Level

- **Objective**: Ensure the vehicle adheres to the planned trajectory, such as maintaining the center of a lane.
- **Localization Requirement**: Highly precise localization relative to previous positions and lane markings is essential. Global localization becomes less critical at this level.

### Role of Digital Maps

Digital maps play a pivotal role across all guidance levels by providing:

- **Lane Topology and Geometry**: Information on lane connections, widths, and curvatures.
- **Road Attributes**: Distinctions between highway and urban roads.
- **Lane Attributes**: Specifications like dedicated bus lanes.
- **Regulatory Elements**: Speed limits, warning signs, traffic lights, and more.
- **Fixed Objects**: Details about walls, curbs, trees, etc., which aid in planning and decision-making.

*Note*: For digital maps to be effective, the vehicle must accurately localize itself within the map, highlighting the intrinsic link between localization and mapping.

---

## Localization Methods

Localization can be categorized based on the reference system and the approach used to determine the vehicle's pose. The primary categories include:

1. **Global Localization**
2. **Relative Localization**
3. **Simultaneous Localization and Mapping (SLAM)**

### Global Localization

**Global Localization** involves determining the vehicle's pose within a known global coordinate system, typically fixed relative to the Earth.

#### GNSS-Based Localization

- **Global Navigation Satellite System (GNSS)**: Utilizes satellite signals (e.g., GPS, GLONASS) to ascertain the vehicle's position on a global scale.
- **Advantages**:
    - Provides absolute positioning.
    - Widely available and standardized.
- **Limitations**:
    - Susceptible to signal obstructions (e.g., tunnels, urban canyons).
    - Limited precision in certain environments.

**Implementation Example: Basic GNSS Positioning**

```python
import gps

def get_gnss_position():
    session = gps.gps(mode=gps.WATCH_ENABLE)
    try:
        report = session.next()
        if report['class'] == 'TPV':
            latitude = getattr(report, 'lat', None)
            longitude = getattr(report, 'lon', None)
            altitude = getattr(report, 'alt', None)
            return latitude, longitude, altitude
    except StopIteration:
        session = None
    return None
```

*Figure 2: Simple Python implementation to retrieve GNSS coordinates.*

#### Landmark-Based Localization

- **Approach**: Uses identifiable landmarks within the environment to determine the vehicle's position.
- **Techniques**:
    - Visual landmarks detected via cameras.
    - Reflective markers detected via lidar or radar.
- **Advantages**:
    - Can enhance accuracy in GNSS-degraded environments.
- **Limitations**:
    - Requires a pre-mapped set of landmarks.
    - May struggle in dynamic or cluttered environments.

**Implementation Example: Landmark Matching Using OpenCV**

```python
import cv2
import numpy as np

# Load pre-mapped landmarks
mapped_landmarks = load_landmarks('map_landmarks.yml')

def detect_landmarks(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = feature_detector.detectAndCompute(gray, None)
    matches = feature_matching(descriptors, mapped_landmarks['descriptors'])
    if len(matches) > MIN_MATCH_COUNT:
        # Estimate pose based on matched landmarks
        pose = estimate_pose(matches, mapped_landmarks['positions'])
        return pose
    return None
```

*Figure 3: Simplified landmark detection and matching using OpenCV.*

### Relative Localization

**Relative Localization** determines the vehicle's pose relative to a previously established position.

#### Inertial Navigation

- **Components**: Combines data from gyroscopes and accelerometers.
- **Functionality**: Calculates changes in position and orientation based on inertial measurements.
- **Advantages**:
    - Independent of external signals.
    - Provides high-frequency updates.
- **Limitations**:
    - Prone to drift over time without external corrections.

**Implementation Example: Basic Inertial Navigation Using an IMU**

```python
import imu_library

def compute_position(accelerations, gyroscope_data, dt):
    velocity += accelerations * dt
    position += velocity * dt
    orientation += gyroscope_data * dt
    return position, orientation
```

*Figure 4: Simple inertial navigation computation loop.*

#### Odometry

- **Definition**: Estimates the vehicle's movement by tracking wheel rotations or other motion-related metrics.
- **Advantages**:
    - Simple and cost-effective.
    - Provides continuous pose updates.
- **Limitations**:
    - Accumulates errors over time.
    - Sensitive to wheel slippage and uneven terrains.

**Implementation Example: Wheel Odometry Calculation**

```python
def wheel_odometry(left_wheel_rotations, right_wheel_rotations, wheel_radius, axle_length):
    left_distance = left_wheel_rotations * 2 * np.pi * wheel_radius
    right_distance = right_wheel_rotations * 2 * np.pi * wheel_radius
    delta_distance = (left_distance + right_distance) / 2
    delta_theta = (right_distance - left_distance) / axle_length
    return delta_distance, delta_theta
```

*Figure 5: Calculation of distance and rotation from wheel rotations.*

#### Visual Odometry

- **Approach**: Utilizes cameras or lidars to track environmental features and estimate motion.
- **Advantages**:
    - Can provide rich environmental context.
    - More robust to certain types of errors compared to traditional odometry.
- **Limitations**:
    - Computationally intensive.
    - Performance can degrade in low-texture or dynamic environments.

**Implementation Example: Feature Tracking for Visual Odometry**

```python
import cv2

def visual_odometry(prev_frame, current_frame):
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(prev_frame, None)
    kp2, des2 = orb.detectAndCompute(current_frame, None)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
    E, mask = cv2.findEssentialMat(src_pts, dst_pts, focal=1.0, pp=(0.,0.))
    _, R, t, mask = cv2.recoverPose(E, src_pts, dst_pts)
    return R, t
```

*Figure 6: Basic visual odometry using feature matching and pose recovery.*

### Simultaneous Localization and Mapping (SLAM)

**SLAM** is a sophisticated approach where the vehicle simultaneously builds a map of its environment and localizes itself within that map.

- **Process**:
    1. **Mapping**: Detect and record environmental features.
    2. **Localization**: Determine the vehicle's pose relative to the newly created map.
- **Challenges**:
    - Both the map and the pose are initially unknown.
    - Requires robust feature detection and data association.
- **Relevance to Automated Driving**:
    - Primarily useful during the initial deployment or in unmapped areas.
    - Less critical for live operations where pre-mapped environments are standard.

**Implementation Example: Basic SLAM Workflow Using ORB-SLAM**

```python
from orbslam2 import System

def initialize_slam(vocabulary_path, settings_path):
    slam = System(vocabulary_path, settings_path, System.Sensor.STEREO)
    slam.initialize()
    return slam

def process_frame(slam, left_image, right_image):
    slam.process_image_stereo(left_image, right_image, timestamp)
    pose = slam.get_current_pose()
    return pose

def shutdown_slam(slam):
    slam.shutdown()
```

*Figure 7: Initialization and processing loop for ORB-SLAM.*

*Note*: Given the focus on live operation in automated driving systems, SLAM is less emphasized compared to global and relative localization methods.

---

## Sensors for Localization

Effective localization relies on a combination of sensors that can perceive both external and internal states of the vehicle.

### Exteroceptive Sensors

**Exteroceptive Sensors** gather information from outside the vehicle, providing data about the surrounding environment.

- **Types**:
    - **GNSS Receivers**: Capture satellite signals for global positioning.
    - **Lidar (Light Detection and Ranging)**: Measures distances to objects using laser pulses.
    - **Radar (Radio Detection and Ranging)**: Detects objects and their velocities using radio waves.
    - **Cameras**: Capture visual information for feature detection and recognition.
- **Functionality**: These sensors primarily receive electromagnetic signals from external sources to interpret the environment.

**Example: Lidar Data Processing for Localization**

```python
import numpy as np
import open3d as o3d

def process_lidar_point_cloud(point_cloud_data):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud_data)
    pcd = pcd.voxel_down_sample(voxel_size=0.05)
    # Further processing like feature extraction can be added here
    return pcd
```

*Figure 8: Simple Lidar point cloud processing using Open3D.*

### Proprioceptive Sensors

**Proprioceptive Sensors** monitor the internal state of the vehicle, providing data about its own dynamics and movements.

- **Types**:
    - **Encoders**: Measure wheel speeds and steering angles.
    - **Gyroscopes**: Detect rotational movements.
    - **Accelerometers**: Measure linear accelerations.
    - **Inertial Measurement Units (IMUs)**: Combine multiple inertial sensors to calculate comprehensive translations and rotations across all six degrees of freedom.
- **Functionality**: These sensors assess internal vehicle metrics to aid in relative localization.

**Example: Reading Data from an IMU**

```python
import smbus
import time

def read_imu_data(bus_number, address):
    bus = smbus.SMBus(bus_number)
    accel_x = bus.read_word_data(address, 0x3B)
    accel_y = bus.read_word_data(address, 0x3D)
    accel_z = bus.read_word_data(address, 0x3F)
    gyro_x = bus.read_word_data(address, 0x43)
    gyro_y = bus.read_word_data(address, 0x45)
    gyro_z = bus.read_word_data(address, 0x47)
    return {
        'accel_x': accel_x,
        'accel_y': accel_y,
        'accel_z': accel_z,
        'gyro_x': gyro_x,
        'gyro_y': gyro_y,
        'gyro_z': gyro_z
    }
```

*Figure 9: Basic IMU data retrieval using SMBus.*

### Sensor Integration

For robust and accurate localization, a synergistic integration of both exteroceptive and proprioceptive sensors is essential. This fusion compensates for the limitations of individual sensors, ensuring reliable pose estimation across diverse driving scenarios.

**Example: Sensor Fusion Using an Extended Kalman Filter (EKF)**

```python
import numpy as np

class EKF:
    def __init__(self, state_dim, meas_dim):
        self.state = np.zeros(state_dim)
        self.P = np.eye(state_dim)
        self.F = np.eye(state_dim)  # State transition model
        self.H = np.zeros((meas_dim, state_dim))  # Measurement model
        self.Q = np.eye(state_dim) * 0.01  # Process noise covariance
        self.R = np.eye(meas_dim) * 0.1    # Measurement noise covariance

    def predict(self):
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        y = z - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state += K @ y
        self.P = (np.eye(len(self.state)) - K @ self.H) @ self.P

# Usage Example
ekf = EKF(state_dim=6, meas_dim=3)
ekf.H[0, 0] = 1  # Assuming measurement affects state x
ekf.H[1, 1] = 1  # measurement affects state y
ekf.H[2, 5] = 1  # measurement affects yaw

# In the main loop
ekf.predict()
measurement = get_gnss_position()  # Example measurement
if measurement:
    ekf.update(measurement)
current_state = ekf.state
```

*Figure 10: Simplified Extended Kalman Filter for sensor fusion.*

---

## Integration of Localization Methods

To achieve high-frequency and resilient pose estimation, automated vehicles often employ a combination of localization methods:

- **Global Localization** (e.g., GNSS) provides absolute positioning but may suffer from signal interruptions.
- **Relative Localization** (e.g., inertial navigation, odometry) offers continuous updates but is prone to drift.
- **Combined Approach**: Integrating both global and relative methods mitigates individual weaknesses, enhancing overall localization accuracy and reliability.

**Example Integration Strategy**:

1. **Primary Localization**: Utilize GNSS for global positioning.
2. **Secondary Localization**: Employ inertial navigation to track short-term movements.
3. **Correction Mechanism**: Use global data to correct and calibrate the relative localization system, preventing drift.

**Implementation Example: Sensor Fusion Combining GNSS and IMU Using EKF**

```python
import numpy as np

class LocalizationSystem:
    def __init__(self):
        self.ekf = EKF(state_dim=6, meas_dim=3)
        # Initialize state transition and measurement models
        self.ekf.F = np.array([[1, 0, 0, 1, 0, 0],
                               [0, 1, 0, 0, 1, 0],
                               [0, 0, 1, 0, 0, 1],
                               [0, 0, 0, 1, 0, 0],
                               [0, 0, 0, 0, 1, 0],
                               [0, 0, 0, 0, 0, 1]])
        self.ekf.H = np.array([[1, 0, 0, 0, 0, 0],
                               [0, 1, 0, 0, 0, 0],
                               [0, 0, 1, 0, 0, 0]])

    def update_with_gnss(self, gnss_position):
        self.ekf.update(gnss_position)

    def update_with_imu(self, imu_data):
        # Update state based on IMU data (simplified)
        self.ekf.state += np.array([imu_data['accel_x'], imu_data['accel_y'], imu_data['gyro_z'], 0, 0, 0])
        self.ekf.predict()

    def get_current_pose(self):
        return self.ekf.state

# Usage Example
localization = LocalizationSystem()

while True:
    imu_data = read_imu_data(bus_number=1, address=0x68)
    localization.update_with_imu(imu_data)
    
    gnss_position = get_gnss_position()
    if gnss_position:
        localization.update_with_gnss(gnss_position)
    
    current_pose = localization.get_current_pose()
    print(f"Current Pose: {current_pose}")
```

*Figure 11: Integrated localization system combining GNSS and IMU data using an Extended Kalman Filter.*

---

## Conclusion

Localization stands as a cornerstone in the architecture of automated vehicles, enabling precise navigation and maneuvering within complex environments. By understanding the nuances of pose determination, the interplay of various localization methods, and the critical role of diverse sensors, stakeholders can enhance the reliability and efficiency of automated driving systems. As technology evolves, the integration and optimization of these components will continue to drive advancements in autonomous mobility.
