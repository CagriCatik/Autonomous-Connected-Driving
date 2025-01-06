# Statements

---

## Introduction

In C++, **statements** are the fundamental building blocks of a program. They define the instructions the computer will execute, making them essential for writing robust and functional code. For **autonomous driving systems**, understanding and utilizing statements effectively is critical for tasks such as controlling vehicle behavior, processing sensor data, and managing real-time operations.

This tutorial will introduce C++ statements and their role in autonomous driving through practical examples.

---

### 1. What Are C++ Statements?

A **statement** is a single instruction written in a C++ program. Each statement must end with a **semicolon** (`;`) to indicate the end of the instruction.

## Example of a Single Statement:

```cpp
cout << "Autonomous driving system initialized.";
```

- **Instruction:** Prints the message to the screen.
- **Semicolon (`;`) Importance:** If omitted, the compiler will throw an error.

## Example of an Error Without a Semicolon:

```cpp
cout << "Autonomous driving system initialized"
```

**Error:** `error: expected ';' before 'return'`

---

### 2. Many Statements in a Program

Most C++ programs contain multiple statements, executed one by one in the order they are written.

## Example: Sequential Statements

```cpp
#include <iostream>
using namespace std;

int main() {
    cout << "Initializing sensors...\n";
    cout << "Calibrating LiDAR...\n";
    cout << "System ready for operation.\n";
    return 0;
}
```

**Explanation:**
1. The first statement prints "Initializing sensors...".
2. The second statement prints "Calibrating LiDAR...".
3. The third statement prints "System ready for operation.".
4. The final statement (`return 0;`) indicates the program executed successfully.

---

### 3. C++ Statements in Autonomous Driving Applications

## Real-World Example: Logging System Events

In autonomous driving, sequential statements can be used to log system events during initialization.

```cpp
#include <iostream>
using namespace std;

int main() {
    cout << "Starting Autonomous Driving System...\n";
    cout << "Loading navigation maps...\n";
    cout << "Activating vehicle control module...\n";
    cout << "System operational.\n";
    return 0;
}
```

**Output:**
```
Starting Autonomous Driving System...
Loading navigation maps...
Activating vehicle control module...
System operational.
```

---

### 4. Statements in Error Handling

In an autonomous driving system, handling errors is vital. C++ statements can be used to log errors when a module fails.

## Example: Error Logging Statements

```cpp
#include <iostream>
using namespace std;

int main() {
    bool sensorActive = false;  // Simulate a sensor error

    if (!sensorActive) {
        cout << "Error: Sensor initialization failed.\n";
    }

    return 0;
}
```

**Explanation:**
- If the sensor is not active, the program outputs an error message.

---

### 5. Combining Statements with Control Structures

While basic statements execute sequentially, control structures allow conditional or repetitive execution of statements. For example, in autonomous driving, statements can be executed repeatedly to process continuous sensor input.

## Example: Processing Sensor Data

```cpp
#include <iostream>
using namespace std;

int main() {
    int distances[] = {12, 15, 8, 20};  // Simulated distance data (in meters)

    for (int i = 0; i < 4; ++i) {
        cout << "Obstacle " << i + 1 << " is " << distances[i] << " meters away.\n";
    }

    return 0;
}
```

**Output:**
```
Obstacle 1 is 12 meters away.
Obstacle 2 is 15 meters away.
Obstacle 3 is 8 meters away.
Obstacle 4 is 20 meters away.
```

---

### 6. Common Errors with Statements

1. **Missing Semicolon:**
   - **Error:** Forgetting the semicolon results in a compilation error.
   - **Solution:** Ensure every statement ends with a semicolon.

   ```cpp
   cout << "Hello World" // Missing semicolon
   ```

2. **Improper Syntax:**
   - **Error:** Typos or incorrect syntax in the statement can cause errors.
   - **Solution:** Double-check syntax before compiling.

   ```cpp
   cout << "Hello World! << endl;  // Missing closing quote
   ```

3. **Order of Execution:**
   - **Error:** Statements execute in the order written. If the order is incorrect, the program may not function as intended.
   - **Solution:** Carefully plan the sequence of statements.

   ```cpp
   cout << "System ready.\n";
   cout << "Initializing...\n";  // Wrong order
   ```

---

### 7. Practical Application: Sequential Task Execution

In autonomous driving systems, statements can represent tasks executed in sequence, such as system initialization and real-time monitoring.

## Example: Vehicle Start Sequence

```cpp
#include <iostream>
using namespace std;

int main() {
    cout << "Starting vehicle...\n";
    cout << "Checking all sensors...\n";
    cout << "Sensors operational.\n";
    cout << "Vehicle in autonomous mode.\n";
    return 0;
}
```

**Output:**
```
Starting vehicle...
Checking all sensors...
Sensors operational.
Vehicle in autonomous mode.
```

---

### 8. Tips for Writing Effective Statements in C++

1. **Use Meaningful Messages:** Provide clear output messages, especially for logging in critical systems like autonomous vehicles.
2. **Minimize Errors:** Always double-check for missing semicolons or typos.
3. **Plan Execution Order:** Ensure statements are ordered logically to reflect the intended sequence of tasks.

---

### Conclusion

C++ statements form the core of every program, providing the instructions the computer executes. In autonomous driving applications, writing clear, logical, and error-free statements is essential to ensure smooth operation and effective debugging. By practicing with examples like logging system events and processing sensor data, you can master the use of statements in real-world scenarios.