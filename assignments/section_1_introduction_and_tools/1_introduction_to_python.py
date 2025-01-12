# autonomous_driving_examples.py

# ___Variables___
print("___Variables___")

vehicle_id = "AV-2025-XYZ"    # String representing the Autonomous Vehicle ID
speed = 65.5                  # Float representing speed in mph
fuel_level = 75               # Integer representing fuel level in percentage
is_autonomous = True          # Boolean indicating if the vehicle is in autonomous mode

print("Vehicle ID:", vehicle_id)
print("Speed:", speed, "mph")
print("Fuel Level:", fuel_level, "%")
print("Is Autonomous:", is_autonomous)
print("================")

# ___Multiple assignments___
print("___Multiple assignments___")

# Assigning multiple sensor readings
front_sensor, rear_sensor, left_sensor, right_sensor = 0.5, 0.7, 0.6, 0.8  # Distances in meters

print("Front Sensor Distance:", front_sensor, "meters")
print("Rear Sensor Distance:", rear_sensor, "meters")
print("Left Sensor Distance:", left_sensor, "meters")
print("Right Sensor Distance:", right_sensor, "meters")
print("================")

# ___Dynamic Typing___
print("___Dynamic Typing___")

# Initially, fuel_level is an integer
fuel_level = 75
print("Fuel Level (int):", fuel_level)  # Output: 75

# Now, fuel_level is updated to a string with units
fuel_level = "75%"
print("Fuel Level (str):", fuel_level)  # Output: 75%
print("================")

# ___Data Types___
print("___Data Types___")

# Integer
number_of_passengers = 4
print("number_of_passengers type:", type(number_of_passengers))  # Output: <class 'int'>

# Float
average_speed = 60.5
print("average_speed type:", type(average_speed))  # Output: <class 'float'>

# String
current_route = "Downtown Loop"
print("current_route type:", type(current_route))  # Output: <class 'str'>

# Boolean
engine_on = True
print("engine_on type:", type(engine_on))  # Output: <class 'bool'>
print("================")

# ___Type Conversion___
print("___Type Conversion___")

# Convert string to integer
fuel_str = "50"
fuel_int = int(fuel_str)
print("Converted Fuel:", fuel_int)                 # Output: 50
print("fuel_int type:", type(fuel_int))           # Output: <class 'int'>

# Convert integer to float
distance_int = 100
distance_float = float(distance_int)
print("Converted Distance:", distance_float)       # Output: 100.0
print("distance_float type:", type(distance_float))  # Output: <class 'float'>

# Convert number to string
speed = 88.8
speed_str = str(speed)
print("Converted Speed:", speed_str)               # Output: "88.8"
print("speed_str type:", type(speed_str))         # Output: <class 'str'>
print("================")

# ___Lists___
print("___Lists___")

# Creating a list of waypoints
waypoints = ["Intersection A", "Intersection B", "Intersection C", "Destination"]
print("Waypoints:", waypoints)

# Accessing elements by index
print("First Waypoint:", waypoints[0])    # Output: Intersection A
print("Third Waypoint:", waypoints[2])    # Output: Intersection C

# Negative indexing
print("Last Waypoint:", waypoints[-1])    # Output: Destination
print("Second Waypoint from End:", waypoints[-3])  # Output: Intersection B

# Changing the second waypoint
waypoints[1] = "Intersection B Updated"
print("Updated Waypoints:", waypoints)  # Output: ['Intersection A', 'Intersection B Updated', 'Intersection C', 'Destination']

# Append
waypoints.append("Intersection D")
print("After Append:", waypoints)  # Adds 'Intersection D' at the end

# Insert
waypoints.insert(2, "Intersection E")
print("After Insert:", waypoints)  # Inserts 'Intersection E' at index 2

# Extend
additional_waypoints = ["Intersection F", "Intersection G"]
waypoints.extend(additional_waypoints)
print("After Extend:", waypoints)  # Extends the list with additional_waypoints

# Remove by value
waypoints.remove("Intersection C")
print("After Remove:", waypoints)  # Removes 'Intersection C'

# Remove by index using pop()
removed_waypoint = waypoints.pop(3)  # Removes the waypoint at index 3
print("Removed Waypoint:", removed_waypoint)
print("Waypoints after Pop:", waypoints)

# Delete using del
del waypoints[0]  # Deletes 'Intersection A'
print("After Delete:", waypoints)

# Clear the list
waypoints.clear()
print("After Clear:", waypoints)  # Output: []
print("================")

# ___List Operations___
print("___List Operations___")

list1 = [1, 2, 3]
list2 = [4, 5, 6]

# Concatenation
combined = list1 + list2
print("Combined List:", combined)  # Output: [1, 2, 3, 4, 5, 6]

# Repetition
repeated = list1 * 3
print("Repeated List:", repeated)  # Output: [1, 2, 3, 1, 2, 3, 1, 2, 3]

# Slicing
subset = combined[2:5]
print("Sliced Subset:", subset)     # Output: [3, 4, 5]

# Create a list of squares from 1 to 10
squares = [x**2 for x in range(1, 11)]
print("List of Squares:", squares)  # Output: [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]
print("================")

# ___F-Strings___
print("___F-Strings___")

vehicle_model = "Tesla Model S"
battery_level = 85.5

# Using f-string
status = f"Vehicle {vehicle_model} has a battery level of {battery_level}%."
print(status)

# Expressions Inside F-Strings
current_speed = 55
target_speed = 65
speed_difference = target_speed - current_speed
speed_info = f"Need to increase speed by {speed_difference} mph to reach the target."
print(speed_info)

# Formatting Numbers
pi_value = 3.141592653589793
formatted_pi = f"Pi rounded to 3 decimal places is {pi_value:.3f}."
print(formatted_pi)

# Using F-Strings with Dictionaries and Lists
driver_info = {
    "name": "Elena",
    "experience_years": 5
}

sensor_readings = ["LIDAR", "RADAR", "Ultrasonic"]

# Accessing dictionary values
intro = f"Driver {driver_info['name']} has {driver_info['experience_years']} years of experience."
print(intro)

# Accessing list elements
primary_sensor = f"The primary sensor is {sensor_readings[0]}."
print(primary_sensor)

# Multiline F-Strings
location = "San Francisco"
speed_limit = 65
fuel_status = "Full"

vehicle_status = f"""
Location: {location}
Current Speed: {speed} mph
Speed Limit: {speed_limit} mph
Fuel Status: {fuel_status}
"""
print(vehicle_status)
print("================")

# ___Simple Loops___
print("___Simple Loops___")

# Iterating over a list of sensor types
sensor_types = ["LIDAR", "RADAR", "Ultrasonic", "Camera"]
for sensor in sensor_types:
    print(f"Processing data from {sensor} sensor.")

# Using range()
print("\nProcessing waypoints:")
for i in range(1, 6):
    print(f"Navigating to waypoint {i}.")

# Looping Through Characters in a String
vehicle_name = "Autonomo"
print("\nVehicle Name Characters:")
for char in vehicle_name:
    print(char)

# While Loop
print("\nCountdown to departure:")
count = 5
while count > 0:
    print(f"Departing in {count} seconds...")
    count -= 1
print("Departure!")
print("================")

# ___Nested Loops___
print("___Nested Loops___")

# Displaying a simple grid of waypoints
print("Waypoint Grid:")
for row in range(1, 4):
    for col in range(1, 4):
        print(f"Waypoint ({row},{col})", end=" | ")
    print("\n" + "---" * 10)

# ___Loop Control Statements___
print("\n___Loop Control Statements___")

# Using break
print("\nScanning sensors:")
for num in range(1, 10):
    if num == 5:
        print("Critical sensor reading detected. Stopping scan.")
        break
    print(f"Sensor {num} scanning...")

# Using continue
print("\nProcessing sensor data:")
for num in range(1, 6):
    if num == 3:
        print("Sensor 3 malfunctioning. Skipping.")
        continue
    print(f"Sensor {num} data processed.")

# Using pass
print("\nFuture sensor integrations:")
for num in range(1, 4):
    if num == 2:
        pass  # Placeholder for future code integration
    print(f"Sensor {num} status checked.")
print("================")

# ___If-Else Statements___
print("___If-Else Statements___")

# Check if speed is within limits
current_speed = 70
speed_limit = 65

if current_speed > speed_limit:
    print("Speeding detected! Initiating slowdown procedures.")
else:
    print("Speed is within the limit.")

# Check if battery level is sufficient
battery_level = 15

if battery_level > 20:
    print("Battery level sufficient for the trip.")
else:
    print("Battery low! Redirecting to nearest charging station.")

# Grade classification (e.g., system health status)
system_health_score = 85

if system_health_score >= 90:
    status = "Excellent"
elif system_health_score >= 80:
    status = "Good"
elif system_health_score >= 70:
    status = "Fair"
elif system_health_score >= 60:
    status = "Poor"
else:
    status = "Critical"

print(f"System Health Status: {status}")

# Check eligibility for autonomous mode
driver_age = 30
has_license = True

if driver_age >= 18:
    if has_license:
        print("Eligible to engage autonomous mode.")
    else:
        print("Not eligible: Valid driver's license required.")
else:
    print("Not eligible: Must be at least 18 years old.")

# Logical Operators
# Check if vehicle is within geofenced area
latitude = 37.7749
longitude = -122.4194

if (latitude > 37.0 and latitude < 38.0) and (longitude > -123.0 and longitude < -122.0):
    print("Vehicle is within the operational area.")
else:
    print("Vehicle is outside the operational area.")

# Using 'or'
current_day = "Sunday"

if current_day == "Saturday" or current_day == "Sunday":
    print("Maintenance scheduled for the weekend.")

# Using 'not'
emergency_mode = False

if not emergency_mode:
    print("Vehicle operating normally.")
else:
    print("Emergency protocols activated!")

# Ternary Operator
# Assign "Autonomous" or "Manual" based on mode
mode = "Autonomous" if is_autonomous else "Manual"
print(f"Current Driving Mode: {mode}")
print("================")

# ___Dictionaries___
print("___Dictionaries___")

# Creating a dictionary for vehicle telemetry
telemetry = {
    "vehicle_id": "AV-2025-XYZ",
    "speed": 60,
    "battery_level": 80,
    "location": "Intersection A"
}
print("Telemetry Data:", telemetry)

# Accessing a value by key
print("Vehicle ID:", telemetry["vehicle_id"])  # Output: AV-2025-XYZ

# Modifying a value
telemetry["speed"] = 65
print("Updated Speed:", telemetry["speed"])   # Output: 65

# Adding a new key-value pair
telemetry["temperature"] = 75
print("Updated Telemetry:", telemetry)

# Remove a key-value pair
del telemetry["location"]
print("After Deletion:", telemetry)  # Removes 'location'

# Using pop()
battery = telemetry.pop("battery_level")
print("Popped Battery Level:", battery)      # Output: 80
print("Telemetry after Pop:", telemetry)

# Iterate through keys
print("\nTelemetry Keys:")
for key in telemetry:
    print(key)

# Iterate through values
print("\nTelemetry Values:")
for value in telemetry.values():
    print(value)

# Iterate through key-value pairs
print("\nTelemetry Key-Value Pairs:")
for key, value in telemetry.items():
    print(f"{key}: {value}")
print("================")

# ___Functions___
print("___Functions___")

# Defining a simple function
def activate_autonomous_mode():
    print("Autonomous mode activated. Navigating to destination.")

# Calling the function
activate_autonomous_mode()

# Function with parameters
def adjust_speed(current_speed, target_speed):
    print(f"Adjusting speed from {current_speed} mph to {target_speed} mph.")

adjust_speed(55, 65)  # Output: Adjusting speed from 55 mph to 65 mph.

# Function that returns a value
def calculate_remaining_range(battery_percentage, consumption_rate):
    return battery_percentage / consumption_rate

remaining_range = calculate_remaining_range(80, 0.8)
print(f"Remaining Range: {remaining_range} miles")  # Output: 100.0 miles

# Function with default parameter
def update_fuel_level(new_level="Full"):
    print(f"Fuel level updated to {new_level}.")

update_fuel_level()            # Output: Fuel level updated to Full.
update_fuel_level("Half")      # Output: Fuel level updated to Half.

# Function with keyword arguments
def set_navigation(destination, route="Shortest", avoid_tolls=True):
    print(f"Setting navigation to {destination} via {route} route. Avoid tolls: {avoid_tolls}")

set_navigation(destination="Downtown", avoid_tolls=False, route="Scenic")
# Output: Setting navigation to Downtown via Scenic route. Avoid tolls: False
print("================")

# ___Error Handling___
print("___Error Handling___")

# Handling division by zero in speed calculation
try:
    current_speed = 60
    target_speed = 0
    speed_ratio = current_speed / target_speed
except ZeroDivisionError:
    print("Error: Target speed cannot be zero.")
finally:
    print("Speed adjustment attempt completed.")

# Handling key access in dictionary
try:
    print(telemetry["location"])
except KeyError:
    print("Error: 'location' key not found in telemetry data.")
finally:
    print("Telemetry data access attempt completed.")
print("================")

# ___Importing Modules___
print("___Importing Modules___")

import math

# Using a function from math module
distance = math.sqrt(256)
print("Calculated Distance:", distance)  # Output: 16.0

# Import specific function from a module
from math import factorial

print("Factorial of 5:", factorial(5))  # Output: 120

print("================")

# ___Classes and Objects___
print("___Classes and Objects___")

# Defining a simple class for Autonomous Vehicle
class AutonomousVehicle:
    def __init__(self, vehicle_id, model, battery_level):
        self.vehicle_id = vehicle_id
        self.model = model
        self.battery_level = battery_level

    def display_status(self):
        print(f"Vehicle {self.vehicle_id} ({self.model}) - Battery: {self.battery_level}%")

    def start_engine(self):
        print(f"Engine started for Vehicle {self.vehicle_id}.")

# Creating an object
av1 = AutonomousVehicle("AV-2025-XYZ", "Tesla Model S", 85)
av1.display_status()  # Output: Vehicle AV-2025-XYZ (Tesla Model S) - Battery: 85%
av1.start_engine()    # Output: Engine started for Vehicle AV-2025-XYZ.

# Parent class
class Sensor:
    def __init__(self, sensor_type, status="Active"):
        self.sensor_type = sensor_type
        self.status = status

    def activate(self):
        self.status = "Active"
        print(f"{self.sensor_type} sensor activated.")

    def deactivate(self):
        self.status = "Inactive"
        print(f"{self.sensor_type} sensor deactivated.")

# Child class inheriting from Sensor
class LidarSensor(Sensor):
    def __init__(self, range_distance):
        super().__init__("LIDAR")
        self.range_distance = range_distance

    def scan(self):
        print(f"LIDAR scanning up to {self.range_distance} meters.")

# Creating an object of child class
lidar = LidarSensor(100)
lidar.activate()
lidar.scan()
print("================")

# ___List Comprehensions___
print("___List Comprehensions___")

# Create a list of even numbers from 1 to 20
even_numbers = [x for x in range(1, 21) if x % 2 == 0]
print("Even Numbers:", even_numbers)  # Output: [2, 4, 6, ..., 20]

# Create a list of sensor statuses
sensor_statuses = ["Active", "Inactive", "Active", "Active", "Inactive"]
active_sensors = [status for status in sensor_statuses if status == "Active"]
print("Active Sensors:", active_sensors)  # Output: ['Active', 'Active', 'Active']

# Nested List Comprehensions
# Create a matrix representing sensor grid
sensor_grid = [[f"Sensor({row},{col})" for col in range(1, 4)] for row in range(1, 4)]
print("Sensor Grid:")
for row in sensor_grid:
    print(row)

# Conditional List Comprehensions
# Create a list of sensors needing maintenance
sensor_health = [95, 80, 60, 45, 30]
sensors_needing_maintenance = [f"Sensor {i+1}" for i, health in enumerate(sensor_health) if health < 70]
print("Sensors Needing Maintenance:", sensors_needing_maintenance)
print("================")

# ___Conclusion___
print("___Conclusion___")
print("All examples related to autonomous driving concepts have been executed successfully.")
print("This script demonstrates fundamental Python programming concepts tailored for an autonomous driving context.")
