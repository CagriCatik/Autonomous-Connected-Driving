import numpy as np
import time

# ___Vectors___
print("___Vectors___")

# A vector can represent various data in autonomous driving, such as sensor readings.
# Creating a simple vector using a list
sensor_readings = [0.5, 0.7, 0.6, 0.8]  # Distances from front, rear, left, right sensors in meters
print("Sensor Readings (List):", sensor_readings)

# Converting list to a NumPy vector
sensor_vector = np.array(sensor_readings)
print("Sensor Readings (NumPy Vector):", sensor_vector)
print("================")

# ___Vector Creation___
print("___Vector Creation___")

# Create a vector using NumPy's array
velocity_vector = np.array([60, 65, 70, 75, 80, 85, 90, 95, 100, 105])  # Speeds from different time steps in mph
print("Velocity Vector:", velocity_vector)

# Create sequences using NumPy
time_steps = np.arange(0, 10, 1)  # Time steps from 0 to 9 seconds
print("Time Steps:", time_steps)

# Create vectors with custom values
custom_vector = np.linspace(0, 1, 10)  # 10 values between 0 and 1
print("Custom Vector (Linspace):", custom_vector)
print("================")

# ___Vector Indexing___
print("___Vector Indexing___")

# Accessing elements by index
print("First Velocity:", velocity_vector[0], "mph")    # Output: 60 mph
print("Third Velocity:", velocity_vector[2], "mph")    # Output: 70 mph

# Negative indexing
print("Last Velocity:", velocity_vector[-1], "mph")    # Output: 105 mph
print("Second Velocity from End:", velocity_vector[-2], "mph")  # Output: 100 mph
print("================")

# ___Vector Slicing___
print("___Vector Slicing___")

# Slicing vectors
print("First two velocities:", velocity_vector[0:2])  # Output: [60 65]
print("Last two velocities:", velocity_vector[-2:])   # Output: [100 105]
print("================")

# ___Mathematical Vector Operations___
print("___Mathematical Vector Operations___")

# Adding two vectors
acceleration_vector = np.array([2]*10)  # Acceleration in m/s²
new_velocity = velocity_vector + acceleration_vector
print("New Velocity Vector after Acceleration:", new_velocity)

# Subtracting vectors
deceleration_vector = np.array([1]*10)  # Deceleration in m/s²
adjusted_velocity = velocity_vector - deceleration_vector
print("Adjusted Velocity Vector after Deceleration:", adjusted_velocity)
print("================")

# ___Single Vector Operations___
print("___Single Vector Operations___")

# Calculating the mean speed
mean_speed = np.mean(velocity_vector)
print("Mean Speed:", mean_speed, "mph")

# Calculating the maximum speed
max_speed = np.max(velocity_vector)
print("Maximum Speed:", max_speed, "mph")

# Calculating the minimum speed
min_speed = np.min(velocity_vector)
print("Minimum Speed:", min_speed, "mph")
print("================")

# ___Vector Element-wise Operations___
print("___Vector Element-wise Operations___")

# Multiplying vectors element-wise
# Now both vectors have the same size (10,)
distance_traveled = velocity_vector * time_steps  # Assuming time_steps in hours for simplicity
print("Distance Traveled per Time Step:", distance_traveled)

# Dividing vectors element-wise
speed_ratio = velocity_vector / (acceleration_vector + 1)  # Adding 1 to avoid division by zero
print("Speed to Acceleration Ratio:", speed_ratio)
print("================")

# ___Scalar Vector Operations___
print("___Scalar Vector Operations___")

# Multiplying a vector by a scalar
scaled_velocity = velocity_vector * 1.1  # Increase speed by 10%
print("Scaled Velocity Vector (10% Increase):", scaled_velocity)

# Dividing a vector by a scalar
reduced_velocity = velocity_vector / 2
print("Reduced Velocity Vector (Half Speed):", reduced_velocity)
print("================")

# ___Vector Dot Product___
print("___Vector Dot Product___")

# Dot product of two vectors
# For example, power calculation: Power = Force * Velocity
force_vector = np.array([1000]*10)  # Force in Newtons
power = np.dot(force_vector, velocity_vector)
print("Total Power:", power, "Watts")
print("================")

# ___Speed Comparison: Vector vs Loop vs Built-in___
print("___Speed Comparison: Vector vs Loop vs Built-in___")

# Creating large vectors for comparison
large_vector_size = 1000000
vec1 = np.random.rand(large_vector_size)
vec2 = np.random.rand(large_vector_size)

# Vectorized operation
start_time = time.time()
vector_result = vec1 + vec2
vector_time = time.time() - start_time
print(f"Vectorized operation time: {vector_time:.5f} seconds")

# Loop operation
start_time = time.time()
loop_result = np.zeros(large_vector_size)
for i in range(large_vector_size):
    loop_result[i] = vec1[i] + vec2[i]
loop_time = time.time() - start_time
print(f"Loop operation time: {loop_time:.5f} seconds")

# Built-in addition
start_time = time.time()
builtin_result = np.add(vec1, vec2)
builtin_time = time.time() - start_time
print(f"Built-in NumPy add operation time: {builtin_time:.5f} seconds")
print("================")

# ___Matrices___
print("___Matrices___")

# Creating a matrix representing a grid of waypoints
waypoint_grid = np.array([
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8]
])
print("Waypoint Grid Matrix:\n", waypoint_grid)
print("================")

# ___Matrix Creation___
print("___Matrix Creation___")

# Create a 3x3 matrix with zeros
zero_matrix = np.zeros((3, 3))
print("Zero Matrix:\n", zero_matrix)

# Create a 3x3 matrix with ones
ones_matrix = np.ones((3, 3))
print("Ones Matrix:\n", ones_matrix)

# Create an identity matrix
identity_matrix = np.eye(3)
print("Identity Matrix:\n", identity_matrix)
print("================")

# ___Matrix Indexing___
print("___Matrix Indexing___")

# Accessing elements
print("Element at (0,0):", waypoint_grid[0, 0])  # Output: 0
print("Element at (2,2):", waypoint_grid[2, 2])  # Output: 8

# Accessing entire row
print("Second Row:", waypoint_grid[1, :])  # Output: [3 4 5]

# Accessing entire column
print("Third Column:", waypoint_grid[:, 2])  # Output: [2 5 8]
print("================")

# ___Matrix Reshape___
print("___Matrix Reshape___")

# Reshape the waypoint grid to a 1x9 matrix
reshaped_matrix = waypoint_grid.reshape(1, 9)
print("Reshaped Matrix (1x9):\n", reshaped_matrix)

# Reshape to a 9x1 matrix
reshaped_matrix = waypoint_grid.reshape(9, 1)
print("Reshaped Matrix (9x1):\n", reshaped_matrix)
print("================")

# ___Matrix Slicing___
print("___Matrix Slicing___")

# Slice to get top-left 2x2 submatrix
top_left = waypoint_grid[:2, :2]
print("Top-left 2x2 Submatrix:\n", top_left)

# Slice to get bottom-right 2x2 submatrix
bottom_right = waypoint_grid[1:, 1:]
print("Bottom-right 2x2 Submatrix:\n", bottom_right)
print("================")

# ___Broadcasting___
print("___Broadcasting___")

# Broadcasting allows operations on arrays of different shapes
# Example: Adding a scalar to a vector
offset = 5
adjusted_velocity = velocity_vector + offset
print("Adjusted Velocity with Offset (Broadcasting):", adjusted_velocity)

# Broadcasting with matrices
bias = np.array([10, 20, 30])
adjusted_grid = waypoint_grid + bias[:, np.newaxis]  # Adding bias to each row
print("Adjusted Waypoint Grid with Bias:\n", adjusted_grid)
print("================")

# ___Advanced Indexing___
print("___Advanced Indexing___")

# Boolean indexing: Select velocities greater than 80 mph
high_speeds = velocity_vector[velocity_vector > 80]
print("Velocities > 80 mph:", high_speeds)

# Fancy indexing: Select specific indices
selected_indices = [0, 2, 4]
selected_velocities = velocity_vector[selected_indices]
print("Selected Velocities (indices 0,2,4):", selected_velocities)
print("================")

# ___Masked Arrays___
print("___Masked Arrays___")

# Masked arrays can handle missing or invalid data
# Example: Mask sensor readings that exceed a threshold
sensor_readings = np.array([0.5, 0.7, 0.6, 0.8, 1.2, 0.4])
mask = sensor_readings > 1.0
masked_sensors = np.ma.masked_where(mask, sensor_readings)
print("Masked Sensor Readings (values > 1.0 masked):", masked_sensors)
print("================")

# ___Stacking and Splitting Arrays___
print("___Stacking and Splitting Arrays___")

# Stacking arrays vertically
stacked_vertically = np.vstack((velocity_vector, acceleration_vector))
print("Vertically Stacked Array:\n", stacked_vertically)

# Stacking arrays horizontally
stacked_horizontally = np.hstack((velocity_vector.reshape(-1,1), acceleration_vector.reshape(-1,1)))
print("Horizontally Stacked Array:\n", stacked_horizontally)

# Splitting arrays
split_arrays = np.split(velocity_vector, 2)  # Split into two equal parts
print("Split Arrays:")
for arr in split_arrays:
    print(arr)
print("================")

# ___Transposing Matrices___
print("___Transposing Matrices___")

# Transpose the waypoint grid
transposed_grid = waypoint_grid.T
print("Transposed Waypoint Grid:\n", transposed_grid)
print("================")

# ___Determinant and Inverse___
print("___Determinant and Inverse___")

# Calculate determinant of a matrix
det_identity = np.linalg.det(identity_matrix)
print("Determinant of Identity Matrix:", det_identity)

# Calculate inverse of a matrix
# For demonstration, use a 2x2 matrix
matrix_2x2 = np.array([[4, 7],
                       [2, 6]])
inverse_matrix = np.linalg.inv(matrix_2x2)
print("Inverse of 2x2 Matrix:\n", inverse_matrix)

# Verify by multiplying the matrix with its inverse
identity_verify = np.dot(matrix_2x2, inverse_matrix)
print("Verification (Matrix * Inverse):\n", identity_verify)
print("================")

# ___Principal Component Analysis (PCA)___
print("___Principal Component Analysis (PCA)___")

# PCA is used for dimensionality reduction
# Example: Reducing sensor data dimensions
from numpy.linalg import svd

# Simulated sensor data matrix (10 samples, 4 sensors)
sensor_data = np.random.rand(10, 4)
print("Sensor Data:\n", sensor_data)

# Perform PCA using Singular Value Decomposition (SVD)
U, S, Vt = svd(sensor_data, full_matrices=False)
print("Singular Values:", S)

# Project data onto first two principal components
principal_components = np.dot(sensor_data, Vt[:2].T)
print("Principal Components (First 2):\n", principal_components)
print("================")

# ___Statistical Functions___
print("___Statistical Functions___")

# Calculate standard deviation
std_dev = np.std(velocity_vector)
print("Standard Deviation of Velocity:", std_dev)

# Calculate variance
variance = np.var(velocity_vector)
print("Variance of Velocity:", variance)

# Calculate correlation matrix
corr_matrix = np.corrcoef(velocity_vector, acceleration_vector)
print("Correlation Matrix between Velocity and Acceleration:\n", corr_matrix)
print("================")

# ___Random Number Generation___
print("___Random Number Generation___")

# Generate random sensor data
random_sensors = np.random.normal(loc=0.7, scale=0.05, size=10)  # Mean=0.7, Std=0.05
print("Random Sensor Readings (Normal Distribution):", random_sensors)

# Generate random integers for simulation
random_integers = np.random.randint(50, 100, size=10)  # Speeds between 50 and 100 mph
print("Random Speeds:", random_integers)
print("================")

# ___Task: Mean Squared Error___
print("___Task: Mean Squared Error___")

# Suppose we have predicted sensor distances and actual sensor distances
actual_distances = np.array([0.5, 0.7, 0.6, 0.8, 0.9, 1.0, 0.95, 0.85, 0.75, 0.65])
predicted_distances = np.array([0.55, 0.65, 0.58, 0.82, 0.88, 1.02, 0.90, 0.80, 0.70, 0.60])

# Calculate Mean Squared Error
mse = np.mean((actual_distances - predicted_distances) ** 2)
print("Mean Squared Error between actual and predicted distances:", mse)
print("================")

# ___Time Comparison___
print("___Time Comparison___")

# Comparing time taken to compute Mean Squared Error using vectorized operations vs loops

# Vectorized computation
start_time = time.time()
mse_vectorized = np.mean((actual_distances - predicted_distances) ** 2)
vectorized_time = time.time() - start_time
print(f"Vectorized MSE computation time: {vectorized_time:.8f} seconds")

# Loop-based computation
start_time = time.time()
squared_errors = []
for a, p in zip(actual_distances, predicted_distances):
    squared_errors.append((a - p) ** 2)
mse_loop = np.mean(squared_errors)
loop_time = time.time() - start_time
print(f"Loop-based MSE computation time: {loop_time:.8f} seconds")
print("================")

# ___Advanced Mathematical Functions___
print("___Advanced Mathematical Functions___")

# Trigonometric functions example: Calculating steering angles
angles_degrees = np.array([0, 30, 45, 60, 90])  # Steering angles in degrees
angles_radians = np.deg2rad(angles_degrees)     # Convert to radians
sin_angles = np.sin(angles_radians)
cos_angles = np.cos(angles_radians)
print("Sine of Angles:", sin_angles)
print("Cosine of Angles:", cos_angles)

# Exponential function example: Signal attenuation
signal_strength = np.array([100, 80, 60, 40, 20])
attenuated_signal = np.exp(-0.1 * signal_strength)
print("Attenuated Signal Strength:", attenuated_signal)
print("================")

# ___Boolean Operations and Masking___
print("___Boolean Operations and Masking___")

# Create a boolean mask for velocities greater than 80 mph
mask_high_speed = velocity_vector > 80
print("Mask for Velocities > 80 mph:", mask_high_speed)

# Apply the mask to get high speeds
high_speeds = velocity_vector[mask_high_speed]
print("High Speeds:", high_speeds)

# Modify elements based on mask
velocity_vector[mask_high_speed] = 80  # Cap speeds at 80 mph
print("Velocity Vector after Capping High Speeds:", velocity_vector)
print("================")

# ___Saving and Loading NumPy Arrays___
print("___Saving and Loading NumPy Arrays___")

# Save the velocity vector to a file
np.save('./assignments/section_1_introduction_and_tools/velocity_vector.npy', velocity_vector)
print("Velocity vector saved to 'velocity_vector.npy'.")

# Load the velocity vector from the file
loaded_velocity = np.load('./assignments/section_1_introduction_and_tools/velocity_vector.npy')
print("Loaded Velocity Vector:", loaded_velocity)
print("================")