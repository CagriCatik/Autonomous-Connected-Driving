# Connectivity in Automated Driving

Connectivity plays a pivotal role in automated driving, enabling a multitude of functions that enhance efficiency, safety, and overall functionality. These connectivity-enabled functions can be categorized into **Cooperative Functions**, **Collective Functions**, and **Supportive Functions**. This documentation delves into each category, providing comprehensive insights supported by concepts from the ACDC course material.

---

## Cooperative Functions

Cooperative functions center around the **exchange of data between individual connected entities**, enriching each entity's locally available information. Each entity independently integrates the shared data with its own, performing necessary computations in a **decentralized** manner. This approach enhances the overall system without relying on a central processing unit.

### Characteristics

- **Decentralized Data Processing**: Each entity handles computations independently, ensuring scalability and resilience.
- **Enhanced Local Awareness**: Shared data supplements locally available information, improving perception and planning capabilities.

### Examples

#### Cooperative Perception

Vehicles share their local **environment models** with nearby counterparts, enabling the detection of objects that might be **occluded** or outside their direct line of sight. For instance, if one vehicle detects a pedestrian obscured from another vehicle's view, sharing this information ensures that all vehicles in the vicinity can respond appropriately, thereby enhancing safety and situational awareness.

```python
# Example: Cooperative Perception Data Sharing

class Vehicle:
    def __init__(self, vehicle_id):
        self.vehicle_id = vehicle_id
        self.local_environment = {}

    def detect_objects(self):
        # Simulate object detection
        self.local_environment = {
            'pedestrian': {'position': (10, 15), 'velocity': (0, 0)},
            'obstacle': {'position': (20, 25), 'velocity': (0, 0)}
        }

    def share_environment(self):
        # Share detected objects with nearby vehicles
        return self.local_environment

    def integrate_shared_data(self, shared_data):
        # Integrate shared data into local environment
        for obj, details in shared_data.items():
            if obj not in self.local_environment:
                self.local_environment[obj] = details

# Usage
vehicle_a = Vehicle('A')
vehicle_b = Vehicle('B')

vehicle_a.detect_objects()
shared_data = vehicle_a.share_environment()
vehicle_b.integrate_shared_data(shared_data)

print(vehicle_b.local_environment)
```

#### Cooperative Planning

Vehicles exchange their **planned behaviors**, such as lane changes or turns, with nearby vehicles. This facilitates better prediction of other vehicles' actions, allowing for smoother coordination in traffic scenarios like approaching a roundabout. By sharing planned routes, vehicles can reduce uncertainty and prevent potential collisions.

```python
# Example: Cooperative Planning

class Vehicle:
    def __init__(self, vehicle_id):
        self.vehicle_id = vehicle_id
        self.planned_route = []

    def plan_route(self, route):
        self.planned_route = route

    def share_route(self):
        return self.planned_route

    def receive_other_route(self, other_route):
        # Adjust planning based on other vehicles' routes
        self.planned_route = self.optimize_route(self.planned_route, other_route)

    def optimize_route(self, own_route, other_route):
        # Simple optimization example
        if own_route == other_route:
            own_route.append('Wait')
        return own_route

# Usage
vehicle_a = Vehicle('A')
vehicle_b = Vehicle('B')

vehicle_a.plan_route(['Lane 1', 'Roundabout', 'Lane 2'])
vehicle_b.plan_route(['Lane 1', 'Roundabout', 'Lane 3'])

route_a = vehicle_a.share_route()
route_b = vehicle_b.share_route()

vehicle_a.receive_other_route(route_b)
vehicle_b.receive_other_route(route_a)

print(vehicle_a.planned_route)
print(vehicle_b.planned_route)
```

These cooperative functions significantly improve situational awareness and predictive capabilities, operating effectively without centralized processing systems.

---

## Collective Functions

Collective functions involve the **aggregation of data from multiple connected entities** and its centralized processing, typically on a **cloud server** or **edge cloud server**. The central server performs data fusion and computations before distributing the processed results back to the connected entities. Utilizing **edge cloud servers**, which are situated near data sources, can further reduce communication latency.

### Characteristics

- **Centralized Data Fusion**: Data from various sources is combined into a unified representation, ensuring consistency and comprehensiveness.
- **Efficient Resource Utilization**: Centralized servers possess greater computational power, enabling the handling of complex tasks more effectively than individual entities.

### Examples

#### Collective Perception

Environment models from individual vehicles are transmitted to a central server, which **fuses** the data into a comprehensive **environment model**. This unified model is then disseminated to all relevant vehicles, ensuring consistent data quality and alleviating the computational burden on individual vehicles.

```python
# Example: Collective Perception with Central Server

class CentralServer:
    def __init__(self):
        self.comprehensive_environment = {}

    def fuse_data(self, vehicle_data):
        # Simple data fusion by aggregating object positions
        for vehicle_id, data in vehicle_data.items():
            for obj, details in data.items():
                self.comprehensive_environment.setdefault(obj, []).append(details['position'])

    def distribute_environment(self):
        return self.comprehensive_environment

class Vehicle:
    def __init__(self, vehicle_id):
        self.vehicle_id = vehicle_id
        self.local_environment = {}

    def detect_objects(self):
        # Simulate object detection
        self.local_environment = {
            'pedestrian': {'position': (10, 15)},
            'obstacle': {'position': (20, 25)}
        }

    def send_data_to_server(self, server):
        server.fuse_data({self.vehicle_id: self.local_environment})

    def receive_comprehensive_environment(self, comprehensive_env):
        self.comprehensive_environment = comprehensive_env

# Usage
server = CentralServer()
vehicle_a = Vehicle('A')
vehicle_b = Vehicle('B')

vehicle_a.detect_objects()
vehicle_b.detect_objects()

vehicle_a.send_data_to_server(server)
vehicle_b.send_data_to_server(server)

comprehensive_env = server.distribute_environment()

vehicle_a.receive_comprehensive_environment(comprehensive_env)
vehicle_b.receive_comprehensive_environment(comprehensive_env)

print(vehicle_a.comprehensive_environment)
print(vehicle_b.comprehensive_environment)
```

#### Collective Planning

Vehicles share their intended routes with a central server, which computes **optimal trajectories** for all vehicles within a specific area. These optimized trajectories are then communicated back to the vehicles for execution, eliminating the need for direct inter-vehicle coordination and achieving global optimization.

```python
# Example: Collective Planning with Central Server

class CentralServer:
    def __init__(self):
        self.vehicle_routes = {}

    def collect_routes(self, vehicle_id, route):
        self.vehicle_routes[vehicle_id] = route

    def compute_optimal_trajectories(self):
        # Simple optimization: avoid route conflicts
        optimized_routes = {}
        for vehicle_id, route in self.vehicle_routes.items():
            if 'Conflict Point' in route:
                optimized_routes[vehicle_id] = [step for step in route if step != 'Conflict Point']
                optimized_routes[vehicle_id].append('Wait')
            else:
                optimized_routes[vehicle_id] = route
        return optimized_routes

    def distribute_trajectories(self, optimized_routes):
        return optimized_routes

class Vehicle:
    def __init__(self, vehicle_id):
        self.vehicle_id = vehicle_id
        self.route = []
        self.optimized_route = []

    def plan_route(self, route):
        self.route = route

    def send_route_to_server(self, server):
        server.collect_routes(self.vehicle_id, self.route)

    def receive_optimized_route(self, optimized_route):
        self.optimized_route = optimized_route

# Usage
server = CentralServer()
vehicle_a = Vehicle('A')
vehicle_b = Vehicle('B')

vehicle_a.plan_route(['Start', 'Conflict Point', 'End'])
vehicle_b.plan_route(['Start', 'Conflict Point', 'End'])

vehicle_a.send_route_to_server(server)
vehicle_b.send_route_to_server(server)

optimized_routes = server.compute_optimal_trajectories()
distributed_routes = server.distribute_trajectories(optimized_routes)

vehicle_a.receive_optimized_route(distributed_routes['A'])
vehicle_b.receive_optimized_route(distributed_routes['B'])

print(vehicle_a.optimized_route)
print(vehicle_b.optimized_route)
```

The centralized nature of collective functions enables more **powerful and consistent computation**, effectively offloading processing tasks from individual vehicles and ensuring harmonized operations across the fleet.

---

## Supportive Functions

Supportive functions depend on **external infrastructure** to provide connected entities with data, computational resources, or services. This category encompasses tasks that are computationally intensive or resource-heavy for individual vehicles to handle independently.

### Characteristics

- **External Support**: Infrastructure components such as servers or traffic management systems offer additional resources or information to vehicles.
- **Scalability**: External servers can manage complex tasks that exceed the computational capacities of individual vehicles, facilitating scalability.

### Examples

#### Traffic Control

Connected traffic systems, including **traffic lights** and **digital speed limit signs**, communicate their current and projected states to road users. Vehicles utilize this information to plan **optimized behaviors**, such as adjusting speed to minimize stops, leading to smoother and more efficient traffic flow. This functionality is already operational in certain urban areas, contributing to improved traffic management.

```python
# Example: Traffic Control Integration

class TrafficLight:
    def __init__(self, location):
        self.location = location
        self.state = 'Red'

    def update_state(self, new_state):
        self.state = new_state

    def broadcast_state(self):
        return {'location': self.location, 'state': self.state}

class Vehicle:
    def __init__(self, vehicle_id):
        self.vehicle_id = vehicle_id
        self.speed = 60  # km/h

    def receive_traffic_info(self, traffic_info):
        for info in traffic_info:
            if info['state'] == 'Red':
                self.adjust_speed(0)
            elif info['state'] == 'Yellow':
                self.adjust_speed(30)
            elif info['state'] == 'Green':
                self.adjust_speed(60)

    def adjust_speed(self, new_speed):
        self.speed = new_speed
        print(f"Vehicle {self.vehicle_id} speed adjusted to {self.speed} km/h")

# Usage
traffic_light = TrafficLight('Main St & 1st Ave')
vehicle = Vehicle('V1')

# Traffic light changes state
traffic_light.update_state('Red')
traffic_info = [traffic_light.broadcast_state()]

vehicle.receive_traffic_info(traffic_info)

traffic_light.update_state('Green')
traffic_info = [traffic_light.broadcast_state()]

vehicle.receive_traffic_info(traffic_info)
```

#### Function Offloading

Vehicles transmit their **sensor data**—such as camera feeds and LiDAR data—to external servers, where advanced **object detection** or **path planning algorithms** are executed. These external servers can leverage large neural networks and higher energy resources, which are often impractical for onboard vehicle systems. For example, a vehicle might send real-time LiDAR data to an edge cloud server for object recognition, receiving the results to inform its decision-making processes.

```python
# Example: Function Offloading to External Server

import requests
import json

class Vehicle:
    def __init__(self, vehicle_id):
        self.vehicle_id = vehicle_id
        self.sensor_data = {}
        self.detected_objects = {}

    def collect_sensor_data(self):
        # Simulate sensor data collection
        self.sensor_data = {
            'LiDAR': {'points': [(1,2), (3,4), (5,6)]},
            'Camera': {'images': ['image1.png', 'image2.png']}
        }

    def send_data_to_server(self, server_url):
        response = requests.post(server_url, json=self.sensor_data)
        if response.status_code == 200:
            self.detected_objects = response.json()
            self.process_detected_objects()

    def process_detected_objects(self):
        # Implement decision-making based on detected objects
        print(f"Vehicle {self.vehicle_id} detected objects: {self.detected_objects}")

# Simulated external server endpoint
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/process_sensor_data', methods=['POST'])
def process_sensor_data():
    data = request.get_json()
    # Simulate object detection
    detected = {'objects': ['pedestrian', 'vehicle']}
    return jsonify(detected), 200

# To run the server, uncomment the following lines:
# if __name__ == '__main__':
#     app.run(debug=True)

# Usage
# Note: This requires the Flask server to be running separately.
# vehicle = Vehicle('V1')
# vehicle.collect_sensor_data()
# vehicle.send_data_to_server('http://localhost:5000/process_sensor_data')
```

Supportive functions exemplify the extension of computational and informational capabilities of connected vehicles through external resources, enabling more sophisticated operations without overburdening onboard systems.

---

## Key Benefits Across Categories

Each category of connectivity-enabled functions uniquely contributes to the advancement of automated and connected driving. Collectively, they offer the following benefits:

- **Increased Safety**: Data sharing and centralized or supportive computation mitigate risks associated with limited perception and delayed decision-making.
- **Enhanced Traffic Efficiency**: Optimized trajectories and real-time communication alleviate congestion and improve traffic flow.
- **Improved Vehicle Coordination**: Cooperative and collective planning ensure smoother and more predictable interactions between vehicles.
- **Scalability and Flexibility**: Supportive functions allow for complex computations and resource offloading, expanding vehicle capabilities without necessitating additional onboard hardware.

---

## Conclusion

The classification of connectivity-enabled functions into **Cooperative**, **Collective**, and **Supportive** highlights the diverse methodologies through which connected entities collaborate to achieve efficient, safe, and intelligent mobility systems. Leveraging these functions addresses the challenges inherent in automated and connected driving, paving the way for significant advancements in transportation technology and infrastructure.