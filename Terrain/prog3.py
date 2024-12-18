import cv2
import numpy as np
from matplotlib import pyplot as plt

def euclidean_distance(p1, p2):
    """Calculate the Euclidean distance between two points."""
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

# Load the image
image_path = "map_route.png"  # Update with the correct path
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Step 1: Isolate the blue path using color masking (HSV color space)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower_blue = np.array([100, 50, 50])  # Lower bound for blue color in HSV
upper_blue = np.array([140, 255, 255])  # Upper bound for blue color in HSV

# Create a mask for the blue color
mask = cv2.inRange(hsv, lower_blue, upper_blue)

# Step 2: Find contours on the masked path
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Select the largest contour (likely the path)
if contours:
    path_contour = max(contours, key=cv2.contourArea)

    # Step 3: Sample points along the contour while avoiding close or duplicate points
    path_points = []
    min_distance = 30  # Minimum distance between circles in pixels

    for point in path_contour:
        point = tuple(point[0])  # Extract coordinates as a tuple
        
        # Check distance with all previously sampled points
        if all(euclidean_distance(point, sampled) > min_distance for sampled in path_points):
            path_points.append(point)

    # Step 4: Draw non-overlapping circles at sampled points
    output_image = image_rgb.copy()
    for point in path_points:
        cv2.circle(output_image, point, 15, (255, 0, 0), 2)  # Draw red circle with radius=10 and thickness=2

    # Display the result
    plt.figure(figsize=(10, 8))
    plt.imshow(output_image)
    plt.title("Path with Non-Overlapping Circles")
    plt.axis("off")
    plt.show()

    # Optional: Print the sampled points
    print("Sampled Path Points:", path_points)
else:
    print("No path detected. Adjust the blue color range in HSV.")
