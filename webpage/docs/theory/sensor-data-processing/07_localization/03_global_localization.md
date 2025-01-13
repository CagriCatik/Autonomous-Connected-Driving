# Global Localization 

Global localization is a fundamental component in the realm of automated vehicles, enabling precise determination of a vehicle's position within a fixed reference system. Accurate localization is crucial for tasks such as route planning, guidance, and control within digital maps. This documentation provides a comprehensive overview of global localization, exploring various methodologies, reference systems, and their applications in automated driving. Whether you are a beginner seeking to understand the basics or an advanced user aiming to deepen your technical knowledge, this guide offers clear explanations and contextual relevance to cater to your needs.

## What is Global Localization?

Global localization refers to the process of determining a vehicle's pose—its position and orientation—in a fixed, global reference system. This knowledge allows automated vehicles to navigate effectively by planning routes, utilizing digital maps for guidance, and executing precise control actions. Unlike relative localization, which determines a vehicle's movement relative to its previous position, global localization provides an absolute position within a broader spatial framework.

### Importance in Automated Driving

- **Route Planning:** Enables the vehicle to determine the most efficient path from point A to point B.
- **Map Integration:** Utilizes digital maps to enhance navigation accuracy and reliability.
- **Guidance and Control:** Assists in executing precise maneuvers based on the vehicle's exact location.

## Reference Coordinate Systems

Global localization relies on established coordinate systems to describe positions on Earth. Understanding these systems is essential for interpreting localization data accurately.

### Spherical Coordinates and Spheroids

- **Spherical Coordinates:** Utilize angles to describe positions on a spherical model of Earth.
  - **Longitude (λ):** Angle describing rotation around the vertical axis.
  - **Latitude (φ):** Angle describing rotation around an axis in the equatorial plane.
  - **Height (h):** Elevation relative to the Earth's modeled surface.
  
- **Spheroids:** More accurate models of Earth's shape, accounting for its slight flattening at the poles.
  - **Semi-Axes (a and b):** Define the shape by rotating an ellipse around one of its semi-axes.

### World Geodetic System 1984 (WGS84)

- **Description:** A widely used global reference system based on spherical coordinates.
- **Origin:** Located at Earth's center of mass.
- **Usage:** Utilized in Global Navigation Satellite Systems (GNSS) for positioning.

### Universal Transverse Mercator (UTM) System

- **Description:** A Cartesian coordinate system projecting spheroid coordinates onto a 2D plane.
- **Structure:** Divides Earth into sixty zones and 24 longitude bands to minimize projection errors.
- **Coordinates:**
  - **Easting:** Represents the distance eastward, typically in meters.
  - **Northing:** Represents the distance northward, typically in meters.
  - **Grid Zone:** Specifies the particular zone and band.

**Advantages of UTM:**
- Simplifies distance calculations compared to latitude and longitude.
- Provides a consistent unit of measurement (meters) for localization tasks.

## Global Navigation Satellite Systems (GNSS)

GNSS plays a pivotal role in global localization by providing accurate positioning data through satellite constellations. Multiple GNSS systems operate globally, each contributing to enhanced localization accuracy.

### Key GNSS Systems

1. **Global Positioning System (GPS):** Developed by the United States.
2. **GLONASS:** Russia's GNSS system.
3. **Galileo:** European Union's GNSS initiative.
4. **BeiDou:** Operated by China.

### Principles of GNSS

- **Signal Transmission:** Satellites emit signals at two distinct frequencies (L1 and L2).
- **Distance Calculation:** Receivers measure the time it takes for signals to reach them, calculating distances based on the constant speed of light.
- **Multilateration:** A geometric method that uses distances from multiple satellites to determine the receiver's precise location.

### Strengths of GNSS-Based Localization

- **Independence from Past Measurements:** Each position calculation is independent, preventing error accumulation over time.
- **High Potential Accuracy:** Under optimal conditions, GNSS can achieve centimeter-level accuracy.

### Weaknesses of GNSS-Based Localization

- **Environmental Dependence:** Relies on external elements like satellites, which are beyond the vehicle's control.
- **Variable Accuracy:** Real-world conditions often lead to inconsistent accuracy levels.
- **Signal Obstruction:** Urban canyons, tunnels, and dense foliage can disrupt signal reception.

## Enhancing GNSS Accuracy

While GNSS provides robust localization capabilities, certain techniques can further enhance its precision and reliability, especially crucial for automated driving applications.

### Dual Frequency Transmission

- **Function:** Utilizes two distinct satellite signal frequencies (L1 and L2).
- **Benefit:** Enables correction of ionospheric delays, reducing errors.
- **Result:** Improves GPS accuracy to approximately five meters.

### GNSS Augmentation Methods

GNSS augmentation involves integrating additional data to correct and refine GNSS positioning.

1. **Ground-Based Systems:**
   - **Operation:** Use nearby base stations with known positions to calculate and transmit correction data.
   - **Coverage:** Limited to areas close to the base stations.

2. **Satellite-Based Systems:**
   - **Operation:** Employ satellites to receive correction data from base stations and broadcast it back to receivers.
   - **Coverage:** Wider area coverage compared to ground-based systems.

3. **Internet-Based Systems:**
   - **Operation:** Transmit correction data via internet connectivity.
   - **Benefit:** Leverages existing communication infrastructure for data dissemination.

### Real-Time Kinematics (RTK)

- **Description:** A specialized GNSS augmentation technique that provides real-time correction data.
- **Mechanism:** Shares correction information between a base station and receiver in real-time, considering the phase of received signals.
- **Accuracy:** Achieves centimeter-level precision.
- **Suitability:** Ideal for automated driving due to its high accuracy and reliability.

**Implementation Example:**
```python
# Example: RTK GNSS Positioning Initialization
import rtklib

# Initialize RTK settings
rtk_settings = rtklib.RTKSettings()
rtk_settings.frequency = 'dual'  # Use dual frequency
rtk_settings.correction_method = 'RTK'

# Connect to GNSS receiver
gnss_receiver = rtklib.GNSSReceiver(port='/dev/ttyUSB0', baudrate=9600)
gnss_receiver.connect()

# Start RTK processing
rtk_processor = rtklib.RTKProcessor(settings=rtk_settings)
rtk_processor.start()

# Retrieve precise position
position = rtk_processor.get_position()
print(f"Precise Position: Easting={position.easting}m, Northing={position.northing}m, Height={position.height}m")
```

*Note: The above code is a simplified example and requires appropriate GNSS libraries and hardware for actual implementation.*

## Landmark-Based Localization

In scenarios where GNSS accuracy is insufficient or unreliable, landmark-based localization serves as an alternative or complementary method. This approach leverages identifiable objects within the environment to determine the vehicle's pose.

### How It Works

1. **Landmark Detection:** The vehicle's sensors identify specific landmarks in the environment.
2. **Spatial Measurement:** The relative positions and orientations of these landmarks are measured concerning the vehicle.
3. **Map Matching:** Detected landmarks are matched with their counterparts in a digital map.
4. **Pose Calculation:** The vehicle's pose is inferred based on the matched landmarks.

### Types of Landmarks

1. **Preexisting Landmarks:**
   - **Definition:** Natural or existing objects in the environment repurposed for localization.
   - **Examples:** Traffic signs, traffic lights, lane markings.
   - **Advantages:** Cost-effective as they leverage existing infrastructure.
   - **Challenges:** May lack uniqueness, leading to potential ambiguities.

2. **Localization-Specific Landmarks (Fiducial Markers):**
   - **Definition:** Dedicated markers placed specifically to aid localization.
   - **Examples:** RFID tags, QR codes, ArUco markers.
   - **Advantages:** Provide precise and unambiguous reference points.
   - **Challenges:** Implementation costs and maintenance across extensive road networks.

### Application Example: Parking Garages

- **Scenario:** GNSS signals are often unreliable in parking garages due to signal obstruction.
- **Solution:** Deploy localization-specific landmarks to indicate garage levels.
- **Benefit:** Enhances localization accuracy by providing clear vertical position cues.

## Comparing Global and Relative Localization

While this documentation focuses on global localization, it's essential to understand its relationship with relative localization, which determines the vehicle's movement based on changes from a previous position. Future sections will delve into relative localization methods and their interplay with global localization to provide a holistic view of vehicle localization strategies.

## Conclusion

Global localization is a critical aspect of automated driving, enabling vehicles to navigate accurately within a fixed reference system. By leveraging GNSS and landmark-based methods, vehicles can achieve precise positioning necessary for safe and efficient operation. Enhancements such as dual frequency transmission and RTK further refine GNSS accuracy, while landmark-based approaches provide reliable alternatives in challenging environments. Understanding and implementing these techniques are essential for advancing the capabilities of automated vehicles.

# References

- **World Geodetic System 1984 (WGS84):** [WGS84 Documentation](https://www.ngs.noaa.gov/CORS/Documentation/WGS84/)
- **Universal Transverse Mercator (UTM):** [UTM Overview](https://en.wikipedia.org/wiki/Universal_Transverse_Mercator_coordinate_system)
- **Real-Time Kinematics (RTK):** [RTK Fundamentals](https://www.gps.gov/applications/rtk/)
- **GNSS Systems:** [Global Navigation Satellite Systems Overview](https://en.wikipedia.org/wiki/Global_Navigation_Satellite_System)

# Glossary

- **GNSS (Global Navigation Satellite System):** A system of satellites providing global positioning data.
- **RTK (Real-Time Kinematics):** A GNSS augmentation technique offering centimeter-level accuracy.
- **Easting:** The distance measured eastward in a Cartesian coordinate system.
- **Northing:** The distance measured northward in a Cartesian coordinate system.
- **Multilateration:** A method of determining position by measuring distances from multiple reference points.
- **Fiducial Markers:** Dedicated markers used to aid in localization and mapping.