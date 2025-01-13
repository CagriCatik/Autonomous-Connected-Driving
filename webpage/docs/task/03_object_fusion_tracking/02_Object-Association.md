# Object Association

![ROS1](https://img.shields.io/badge/ROS1-blue)

Object association is a critical component in robotics applications, enabling the system to track and manage multiple objects across sensor data streams. In the context of the Robot Operating System (ROS1), object association facilitates the fusion of data from various sensors to maintain coherent and accurate representations of objects in the environment.

This documentation provides a comprehensive guide to implementing object association using two primary distance measures: Intersection over Union (IoU) and Mahalanobis Distance. It is tailored to assist both beginners and advanced users in understanding and executing the implementation within a ROS1 workspace.

## Prerequisites

Before proceeding, ensure you have the following:

- **ROS1** installed and properly configured.
- **Catkin Workspace** set up.
- Basic knowledge of **C++** programming.
- Familiarity with **Git** and version control.
- Access to the [ACDC GitHub Repository](https://github.com/ika-rwth-aachen/acdc).

## Overview of Object Association

Object association involves matching detected objects across different frames or sensor inputs to maintain consistent tracking. It plays a vital role in applications like autonomous driving, where understanding the movement and behavior of surrounding objects is essential.

In this guide, we focus on enhancing the `object_fusion` module within a ROS1 workspace. Specifically, we'll implement two distance measures that quantify the similarity between objects: IoU and Mahalanobis Distance. These measures aid in accurately associating objects detected by different sensors or at different times.

## Distance Measures for Object Association

### Intersection over Union (IoU)

**Intersection over Union (IoU)** is a metric used to evaluate the overlap between two bounding boxes. It is defined as the area of overlap divided by the area of union of the two bounding boxes.

**Formula:**

$$
IoU = \frac{\text{Area of Overlap}}{\text{Area of Union}}
$$

IoU is widely used in object detection tasks to measure the accuracy of predicted bounding boxes against ground truth.

### Mahalanobis Distance

**Mahalanobis Distance** measures the distance between a point and a distribution. Unlike Euclidean distance, it accounts for the correlations of the data set, making it effective in high-dimensional spaces.

**Formula:**

$$
d_{G,S} = \sqrt{(x - \mu)^T \Sigma^{-1} (x - \mu)}
$$

Where:
- $ x $ is the point.
- $ \mu $ is the mean of the distribution.
- $ \Sigma$ is the covariance matrix.

In object association, Mahalanobis distance is used to assess the similarity between object positions, considering their positional uncertainties.

## Implementation Guide

### 1. Setting Up the Environment

Ensure your catkin workspace is properly set up and sourced. Navigate to your workspace directory:

```bash
cd ~/catkin_workspace/
```

If you haven't initialized the workspace yet:

```bash
catkin init
catkin config --extend /opt/ros/noetic
catkin build
source devel/setup.bash
```

### 2. Implementing Intersection over Union (IoU)

The IoU implementation is located in the `IntersectionOverUnion.cpp` file within the `distance_measures` module.

#### Step-by-Step Implementation:

1. **Navigate to the Distance Measures Directory:**

    ```bash
    cd ~/catkin_workspace/src/workshops/section_3/object_fusion/src/modules/matcher/distance_measures/
    ```

2. **Open `IntersectionOverUnion.cpp` for Editing:**

    ```bash
    nano IntersectionOverUnion.cpp
    ```

3. **Fill in the IoU Calculation:**

    Locate the function responsible for calculating IoU (typically around line 40 based on the provided link) and implement the following logic:

    ```cpp
    double IntersectionOverUnion::compute(const Object& obj1, const Object& obj2) {
        // Calculate the (x, y)-coordinates of the intersection rectangle
        double x_left = std::max(obj1.bounding_box.x_min, obj2.bounding_box.x_min);
        double y_top = std::max(obj1.bounding_box.y_min, obj2.bounding_box.y_min);
        double x_right = std::min(obj1.bounding_box.x_max, obj2.bounding_box.x_max);
        double y_bottom = std::min(obj1.bounding_box.y_max, obj2.bounding_box.y_max);

        // Compute the area of intersection rectangle
        double intersection_area = std::max(0.0, x_right - x_left) * std::max(0.0, y_bottom - y_top);

        // Compute the area of both bounding boxes
        double obj1_area = (obj1.bounding_box.x_max - obj1.bounding_box.x_min) * 
                            (obj1.bounding_box.y_max - obj1.bounding_box.y_min);
        double obj2_area = (obj2.bounding_box.x_max - obj2.bounding_box.x_min) * 
                            (obj2.bounding_box.y_max - obj2.bounding_box.y_min);

        // Compute the IoU
        double iou = intersection_area / (obj1_area + obj2_area - intersection_area);

        return iou;
    }
    ```

4. **Save and Exit:**

    Press `Ctrl + X`, then `Y`, and `Enter` to save the changes.

#### Code Snippet:

```cpp
double IntersectionOverUnion::compute(const Object& obj1, const Object& obj2) {
    double x_left = std::max(obj1.bounding_box.x_min, obj2.bounding_box.x_min);
    double y_top = std::max(obj1.bounding_box.y_min, obj2.bounding_box.y_min);
    double x_right = std::min(obj1.bounding_box.x_max, obj2.bounding_box.x_max);
    double y_bottom = std::min(obj1.bounding_box.y_max, obj2.bounding_box.y_max);

    double intersection_area = std::max(0.0, x_right - x_left) * std::max(0.0, y_bottom - y_top);
    double obj1_area = (obj1.bounding_box.x_max - obj1.bounding_box.x_min) * 
                        (obj1.bounding_box.y_max - obj1.bounding_box.y_min);
    double obj2_area = (obj2.bounding_box.x_max - obj2.bounding_box.x_min) * 
                        (obj2.bounding_box.y_max - obj2.bounding_box.y_min);

    double iou = intersection_area / (obj1_area + obj2_area - intersection_area);
    return iou;
}
```

### 3. Implementing Mahalanobis Distance

The Mahalanobis distance implementation is located in the `Mahalanobis.cpp` file within the `distance_measures` module.

#### Step-by-Step Implementation:

1. **Open `Mahalanobis.cpp` for Editing:**

    ```bash
    nano Mahalanobis.cpp
    ```

2. **Implement the Mahalanobis Distance Calculation:**

    Locate the function responsible for calculating Mahalanobis distance (typically around line 69 based on the provided link) and implement the following logic:

    ```cpp
    double Mahalanobis::compute(const Object& obj1, const Object& obj2) {
        // Extract positions
        Eigen::Vector2d x(obj1.position.x, obj1.position.y);
        Eigen::Vector2d y(obj2.position.x, obj2.position.y);

        // Calculate the difference vector
        Eigen::Vector2d diff = x - y;

        // Compute the covariance matrix (assuming independence, diagonal matrix)
        Eigen::Matrix2d covariance;
        covariance << obj1.covariance_xx, 0,
                      0, obj1.covariance_yy;

        // Compute the inverse of the covariance matrix
        Eigen::Matrix2d covariance_inv = covariance.inverse();

        // Calculate Mahalanobis distance
        double distance = std::sqrt(diff.transpose() * covariance_inv * diff);

        return distance;
    }
    ```

3. **Save and Exit:**

    Press `Ctrl + X`, then `Y`, and `Enter` to save the changes.

#### Code Snippet:

```cpp
double Mahalanobis::compute(const Object& obj1, const Object& obj2) {
    Eigen::Vector2d x(obj1.position.x, obj1.position.y);
    Eigen::Vector2d y(obj2.position.x, obj2.position.y);
    Eigen::Vector2d diff = x - y;

    Eigen::Matrix2d covariance;
    covariance << obj1.covariance_xx, 0,
                  0, obj1.covariance_yy;

    Eigen::Matrix2d covariance_inv = covariance.inverse();
    double distance = std::sqrt(diff.transpose() * covariance_inv * diff);

    return distance;
}
```

### 4. Configuring the Distance Measure

To select the desired distance measure for object association, you need to modify the fusion configuration file.

#### Step-by-Step Configuration:

1. **Navigate to the Configuration Directory:**

    ```bash
    cd ~/catkin_workspace/src/workshops/section_3/object_fusion_wrapper/param/
    ```

2. **Open `fusion.yaml` for Editing:**

    ```bash
    nano fusion.yaml
    ```

3. **Modify the Distance Measure Parameter:**

    Locate the parameter that specifies the distance measure (line 4 based on the provided link). Set it to either `IoU` or `Mahalanobis` based on your implementation.

    ```yaml
    distance_measure: "IoU"  # Options: "IoU", "Mahalanobis"

    # Other association parameters
    association_threshold: 0.5
    max_association_distance: 2.0
    ```
    
    **Example for Mahalanobis Distance:**

    ```yaml
    distance_measure: "Mahalanobis"
    
    # Other association parameters
    association_threshold: 2.5
    max_association_distance: 5.0
    ```

4. **Save and Exit:**

    Press `Ctrl + X`, then `Y`, and `Enter` to save the changes.

## Building the Workspace

After implementing the distance measures, it's essential to rebuild the catkin workspace to incorporate the changes.

#### Steps to Build:

1. **Navigate to the Catkin Workspace Root:**

    ```bash
    cd ~/catkin_workspace/
    ```

2. **Build the Workspace Using Catkin:**

    ```bash
    catkin build
    ```

    *Note:* Ensure that there are no compilation errors. If errors occur, revisit the implemented code sections for potential syntax or logical mistakes.

3. **Source the Setup File:**

    ```bash
    source devel/setup.bash
    ```

## Verification and Testing

Once the workspace is successfully built, it's crucial to verify that the distance measures function as intended.

#### Steps for Verification:

1. **Launch the Object Fusion Node:**

    ```bash
    roslaunch object_fusion_wrapper object_fusion.launch
    ```

2. **Monitor the Output:**

    Observe the console logs to ensure that objects are being associated correctly based on the chosen distance measure. Look for messages indicating successful association or any warnings/errors.

3. **Test with Different Distance Measures:**

    - Switch between `IoU` and `Mahalanobis` in the `fusion.yaml` configuration.
    - Rebuild the workspace after each change.
    - Verify that the association behavior aligns with the selected distance measure.

4. **Use Visualization Tools:**

    Utilize tools like `rviz` to visualize object associations and ensure that objects are tracked accurately without drifting.

    ```bash
    rosrun rviz rviz
    ```

## Troubleshooting

During implementation, you may encounter issues. Here are common problems and their solutions:

### 1. Compilation Errors

**Issue:** Errors during `catkin build` related to the implemented C++ code.

**Solution:**
- Check for syntax errors in `IntersectionOverUnion.cpp` and `Mahalanobis.cpp`.
- Ensure that all necessary headers are included.
- Verify that Eigen library is correctly referenced if used.

### 2. Runtime Errors

**Issue:** Object fusion node crashes or behaves unexpectedly during runtime.

**Solution:**
- Review console logs for error messages.
- Ensure that all object attributes (e.g., bounding boxes, positions, covariances) are correctly initialized and populated.
- Validate that the covariance matrix is invertible when calculating Mahalanobis distance.

### 3. Incorrect Associations

**Issue:** Objects are not being associated correctly based on the distance measure.

**Solution:**
- Verify that the distance measure implementation aligns with the mathematical definitions.
- Adjust `association_threshold` and `max_association_distance` parameters in `fusion.yaml` to appropriate values.
- Ensure consistency in the units and scales of positional data.

## Next Steps: Object Fusion

While implementing distance measures completes the association step, it's essential to proceed with the **Object Fusion** step to integrate associated objects into a unified representation. This prevents object drift and maintains accurate tracking over time.

Upcoming tasks include:

- Implementing fusion algorithms to merge associated objects.
- Handling uncertainties and dynamic changes in the environment.
- Optimizing performance for real-time applications.

Stay tuned for the next documentation segment on completing object fusion!

## Conclusion

This guide provided a detailed walkthrough for implementing object association in a ROS1 workspace using IoU and Mahalanobis distance measures. By following the steps outlined, you can enhance the object fusion module to accurately associate objects, laying the foundation for robust tracking and perception in your robotic applications. For further assistance, refer to the [ACDC GitHub Repository](https://github.com/ika-rwth-aachen/acdc) or consult the ROS1 documentation.

---

# Appendix

## Code References

- **Intersection over Union (IoU):**
    - [IntersectionOverUnion.cpp](https://github.com/ika-rwth-aachen/acdc/blob/main/catkin_workspace/src/workshops/section_3/object_fusion/src/modules/matcher/distance_measures/IntersectionOverUnion.cpp#L40)

- **Mahalanobis Distance:**
    - [Mahalanobis.cpp](https://github.com/ika-rwth-aachen/acdc/blob/main/catkin_workspace/src/workshops/section_3/object_fusion/src/modules/matcher/distance_measures/Mahalanobis.cpp#L69)

- **Fusion Configuration:**
    - [fusion.yaml](https://github.com/ika-rwth-aachen/acdc/blob/main/catkin_workspace/src/workshops/section_3/object_fusion_wrapper/param/fusion.yaml#L4)

---

# Further Reading

- [ROS1 Documentation](http://wiki.ros.org/)
- [Understanding IoU and Its Applications](https://towardsdatascience.com/understanding-mean-average-precision-mAP-623efb7f407e)
- [Mahalanobis Distance Explained](https://www.geeksforgeeks.org/mahalanobis-distance-python-example/)