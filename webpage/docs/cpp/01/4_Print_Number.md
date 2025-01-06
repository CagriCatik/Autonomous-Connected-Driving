# Output for Numbers

---

## Introduction

In C++, the `cout` object can also be used to output **numbers**, making it essential for printing numerical data such as sensor readings, calculations, and real-time statistics in **autonomous driving systems**. Unlike text, numbers are printed without surrounding them in double quotes.

This tutorial covers how to output numbers and perform basic calculations in C++, using examples relevant to autonomous driving, such as displaying vehicle speed, sensor distances, and computational results.

---

### 1. Printing Numbers with `cout`

Numbers can be directly passed to the `cout` object without double quotes.

## Example: Simple Number Output

```cpp
#include <iostream>
using namespace std;

int main() {
    cout << 42;  // Outputs a number
    return 0;
}
```

**Output:**
```
42
```

- **Explanation:** The number `42` is directly sent to the `cout` object and displayed on the console.

---

### 2. Performing Mathematical Calculations

The `cout` object can display the result of mathematical operations.

## Example: Basic Arithmetic Operations

```cpp
#include <iostream>
using namespace std;

int main() {
    cout << 5 + 3 << "\n";  // Addition
    cout << 10 - 4 << "\n"; // Subtraction
    cout << 2 * 6 << "\n";  // Multiplication
    cout << 20 / 4 << "\n"; // Division
    return 0;
}
```

**Output:**
```
8
6
12
5
```

- **Explanation:** Each `cout` statement prints the result of an arithmetic operation.

---

### 3. Practical Applications in Autonomous Driving

## Example: Displaying Vehicle Speed

```cpp
#include <iostream>
using namespace std;

int main() {
    int currentSpeed = 80;  // Speed in km/h
    cout << "Current speed: " << currentSpeed << " km/h\n";
    return 0;
}
```

**Output:**
```
Current speed: 80 km/h
```

- **Explanation:** The integer value `currentSpeed` is concatenated with strings and printed.

---

## Example: Calculating and Displaying Sensor Ranges

```cpp
#include <iostream>
using namespace std;

int main() {
    float lidarRange = 25.5;  // LiDAR range in meters
    float radarRange = 100.3; // Radar range in meters

    float totalCoverage = lidarRange + radarRange;

    cout << "LiDAR range: " << lidarRange << " meters\n";
    cout << "Radar range: " << radarRange << " meters\n";
    cout << "Total coverage: " << totalCoverage << " meters\n";

    return 0;
}
```

**Output:**
```
LiDAR range: 25.5 meters
Radar range: 100.3 meters
Total coverage: 125.8 meters
```

- **Explanation:** The sum of `lidarRange` and `radarRange` is calculated and printed.

---

### 4. Combining Numbers and Text in Output

In autonomous driving applications, it’s common to print numbers alongside descriptive text to convey context.

## Example: Obstacle Detection

```cpp
#include <iostream>
using namespace std;

int main() {
    int detectedObstacles = 5;
    cout << "Number of obstacles detected: " << detectedObstacles << "\n";
    return 0;
}
```

**Output:**
```
Number of obstacles detected: 5
```

- **Explanation:** The number `detectedObstacles` is combined with text for a meaningful output.

---

### 5. Advanced Example: Mathematical Operations in Autonomous Systems

## Example: Calculating Stopping Distance

Stopping distance is calculated as:

$$
\text{Stopping Distance} = \text{Reaction Distance} + \text{Braking Distance}
$$


```cpp
#include <iostream>
using namespace std;

int main() {
    float reactionTime = 1.5;  // Reaction time in seconds
    int speed = 80;           // Speed in km/h
    float brakingDistance = 40.0;  // Braking distance in meters

    // Reaction distance in meters
    float reactionDistance = (speed * 1000 / 3600) * reactionTime;

    // Total stopping distance
    float stoppingDistance = reactionDistance + brakingDistance;

    cout << "Reaction Distance: " << reactionDistance << " meters\n";
    cout << "Braking Distance: " << brakingDistance << " meters\n";
    cout << "Total Stopping Distance: " << stoppingDistance << " meters\n";

    return 0;
}
```

**Output:**
```
Reaction Distance: 33.3333 meters
Braking Distance: 40 meters
Total Stopping Distance: 73.3333 meters
```

- **Explanation:** The reaction distance is calculated using speed and reaction time, and the stopping distance is the sum of reaction and braking distances.

---

### 6. Common Errors with `cout` for Numbers

1. **Including Numbers in Double Quotes:**
   - **Error:** Treats the number as a string.
   - **Example:**
     ```cpp
     cout << "42";  // Outputs "42" as text, not a number
     ```
   - **Solution:** Omit double quotes for numerical values.

2. **Dividing by Zero:**
   - **Error:** Causes runtime errors.
   - **Example:**
     ```cpp
     cout << 10 / 0;
     ```
   - **Solution:** Always validate divisor values.

3. **Incorrect Data Types:**
   - **Error:** Operations on mismatched data types may lead to unintended results.
   - **Solution:** Use appropriate type casting.

---

### 7. Practical Tips for Printing Numbers in C++

1. **Use Descriptive Messages:**
   - Combine numbers with explanatory text for clarity.
   - Example: `"Distance to obstacle: 12.5 meters"`.

2. **Format Outputs:**
   - Use `\n` for new lines and `\t` for tab spaces to improve readability.

3. **Debugging:**
   - Use `cout` to print intermediate results during calculations for debugging complex algorithms.

---

### Conclusion

Using `cout` to output numbers is a fundamental skill in C++. In autonomous driving applications, it is essential for displaying critical information like vehicle speed, sensor ranges, and computational results. By combining numbers with meaningful text and leveraging mathematical operations, you can create informative and readable outputs for debugging and system monitoring. Practice with real-world examples to master this concept and apply it effectively in your projects.