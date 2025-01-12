Certainly! Below are the corrected sections of your documentation where the variable `K` is used. These revisions ensure that `K` is properly defined within all relevant code blocks and references to prevent build errors.

---

### Mathematical Foundation

A camera's projection model describes how 3D world coordinates are mapped to 2D image coordinates. This relationship is captured by the camera's intrinsic and extrinsic parameters.

The projection can be represented as:

$$
\mathbf{x}_{\text{image}} = \mathbf{K} \begin{bmatrix} \mathbf{R} & \mathbf{t} \end{bmatrix} \mathbf{X}_{\text{world}}
$$

Where:
- $\mathbf{x}_{\text{image}}$: Homogeneous pixel coordinates in the image.
- $\mathbf{K}$: Intrinsic matrix, encapsulating focal length and principal point.
- $\mathbf{R}$: Rotation matrix, representing the camera's orientation.
- $\mathbf{t}$: Translation vector, representing the camera's position.
- $\mathbf{X}_{\text{world}}$: Homogeneous world coordinates.

**Intrinsic Matrix** ($\mathbf{K}$) **Example:**

$$
\mathbf{K} = \begin{bmatrix}
f_x & 0 & c_x \\
0 & f_y & c_y \\
0 & 0 & 1
\end{bmatrix}
$$

Where:
- $f_x, f_y$: Focal lengths in pixels.
- $c_x, c_y$: Principal point coordinates.

---

### Implementation Example

To illustrate the practical application of IPM, consider a Python implementation using OpenCV and NumPy. This example demonstrates how to transform a semantically segmented image into a top-down semantic grid map.

#### Code Explanation

```python
import numpy as np
import cv2

def inverse_perspective_mapping(image, K, R, t, grid_size=(500, 500), cell_size=0.1):
    """
    Perform inverse perspective mapping on a semantically segmented image.

    Args:
        image (np.ndarray): Input semantically segmented image.
        K (np.ndarray): Intrinsic matrix (3x3).
        R (np.ndarray): Rotation matrix (3x3).
        t (np.ndarray): Translation vector (3x1).
        grid_size (tuple): Size of the output grid map (height, width).
        cell_size (float): Size of each grid cell in meters.

    Returns:
        np.ndarray: Top-down view of the semantic grid.
    """
    height, width = grid_size
    grid_map = np.zeros((height, width, 3), dtype=np.uint8)

    # Precompute the inverse matrices
    K_inv = np.linalg.inv(K)
    R_inv = np.linalg.inv(R)

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            pixel = np.array([x, y, 1]).reshape(3, 1)
            # Convert pixel to normalized camera coordinates
            cam_coords = K_inv @ pixel
            # Apply inverse rotation and translation
            world_coords = R_inv @ (cam_coords - t)
            grid_x = int(world_coords[0][0] / cell_size) + width // 2
            grid_y = int(world_coords[1][0] / cell_size) + height // 2

            if 0 <= grid_x < width and 0 <= grid_y < height:
                grid_map[grid_y, grid_x] = image[y, x]

    return grid_map

# Example usage
if __name__ == "__main__":
    # Load the semantically segmented image
    image = cv2.imread("semantic_image.png")

    # Define intrinsic parameters
    fx, fy = 800, 800  # Focal lengths in pixels
    cx, cy = image.shape[1] / 2, image.shape[0] / 2  # Principal point
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]])

    # Define extrinsic parameters (assuming camera is level and at origin)
    R = np.eye(3)  # No rotation
    t = np.zeros((3, 1))  # No translation

    # Perform IPM
    semantic_grid = inverse_perspective_mapping(image, K, R, t)

    # Save the resulting semantic grid map
    cv2.imwrite("semantic_grid.png", semantic_grid)
```

#### Optimized IPM Implementation

```python
import numpy as np
import cv2

def optimized_inverse_perspective_mapping(image, K, R, t, grid_size=(500, 500), cell_size=0.1):
    """
    Optimized Inverse Perspective Mapping using homography.

    Args:
        image (np.ndarray): Input semantically segmented image.
        K (np.ndarray): Intrinsic matrix (3x3).
        R (np.ndarray): Rotation matrix (3x3).
        t (np.ndarray): Translation vector (3x1).
        grid_size (tuple): Size of the output grid map (height, width).
        cell_size (float): Size of each grid cell in meters.

    Returns:
        np.ndarray: Top-down view of the semantic grid.
    """
    height, width = grid_size
    grid_map = np.zeros((height, width, 3), dtype=np.uint8)

    # Compute homography matrix
    H = K @ np.hstack((R, t))
    H_inv = np.linalg.inv(H)

    # Generate grid coordinates
    xx, yy = np.meshgrid(np.arange(width), np.arange(height))
    grid_coords = np.stack([xx * cell_size - (width / 2) * cell_size,
                            yy * cell_size - (height / 2) * cell_size,
                            np.ones_like(xx)], axis=-1).reshape(-1, 3).T

    # Map grid coordinates to image pixels
    image_coords = H_inv @ grid_coords
    image_coords /= image_coords[2, :]

    x_img = image_coords[0, :].astype(np.int32)
    y_img = image_coords[1, :].astype(np.int32)

    # Filter valid coordinates
    valid = (x_img >= 0) & (x_img < image.shape[1]) & (y_img >= 0) & (y_img < image.shape[0])

    grid_map[yy.flatten()[valid], xx.flatten()[valid]] = image[y_img[valid], x_img[valid]]

    return grid_map

# Example usage
if __name__ == "__main__":
    # Load the semantically segmented image
    image = cv2.imread("semantic_image.png")

    # Define intrinsic parameters
    fx, fy = 800, 800  # Focal lengths in pixels
    cx, cy = image.shape[1] / 2, image.shape[0] / 2  # Principal point
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]])

    # Define extrinsic parameters (assuming camera is level and at origin)
    R = np.eye(3)  # No rotation
    t = np.zeros((3, 1))  # No translation

    # Perform optimized IPM
    semantic_grid = optimized_inverse_perspective_mapping(image, K, R, t)

    # Save the resulting semantic grid map
    cv2.imwrite("semantic_grid_optimized.png", semantic_grid)
```

---

Ensure that all instances of `K` within your documentation are defined within code blocks as shown above. This will prevent the `ReferenceError: K is not defined` during the build process.

If you have any further issues or need additional corrections, feel free to ask!