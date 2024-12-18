import cv2
import numpy as np
from matplotlib import pyplot as plt
from collections import Counter
import os

def euclidean_distance(p1, p2):
    """Calculate the Euclidean distance between two points."""
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def crop_circle_region(image, center, radius, save_path, index):
    """Crop a square region around the circle and save it."""
    x, y = center
    x1, y1 = max(x - radius, 0), max(y - radius, 0)
    x2, y2 = min(x + radius, image.shape[1]), min(y + radius, image.shape[0])
    
    cropped_image = image[y1:y2, x1:x2]
    crop_file_path = os.path.join(save_path, f"circle_{index}.png")
    cv2.imwrite(crop_file_path, cropped_image)
    return crop_file_path

def analyze_circle_image(image_path):
    """Analyze the image to find the most common color, excluding purple."""
    image = cv2.imread(image_path)  
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

    # Step 2: Create a mask for the circle
    height, width, _ = image.shape
    center = (width // 2, height // 2)
    radius = min(width, height) // 2
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, thickness=-1)

    # Step 3: Extract pixels inside the circle
    pixels_inside_circle = image[mask == 255]
    pixels_as_tuples = [tuple(pixel) for pixel in pixels_inside_circle]

    # Step 4: Exclude purple pixels
    def is_purple(pixel):
        r, g, b = pixel
        return b > 120 and r > 80 and r < 150 and abs(r - b) < 60

    filtered_pixels = [pixel for pixel in pixels_as_tuples if not is_purple(pixel)]

    # Step 5: Find the most common color
    if filtered_pixels:
        most_common_color, count = Counter(filtered_pixels).most_common(1)[0]
        print(f"Most Common Color in {image_path}: {most_common_color}, Count: {count}")
    else:
        print(f"No valid colors found in {image_path} after excluding purple.")

# Step 1: Load the main image
main_image_path = "map_route.png"  # Update path to input image
output_dir = "cropped_circles"  # Directory to save circle crops
os.makedirs(output_dir, exist_ok=True)

image = cv2.imread(main_image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Step 2: Isolate the blue path using color masking
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower_blue = np.array([100, 50, 50])
upper_blue = np.array([140, 255, 255])
mask = cv2.inRange(hsv, lower_blue, upper_blue)
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Step 3: Detect circle centers along the path
if contours:
    path_contour = max(contours, key=cv2.contourArea)

    path_points = []
    min_distance = 30  # Minimum spacing
    radius = 15  # Circle radius for cropping

    for point in path_contour:
        point = tuple(point[0])
        if all(euclidean_distance(point, p) > min_distance for p in path_points):
            path_points.append(point)

    # Step 4: Crop each circle and analyze it
    print("Analyzing cropped circle regions...")
    for idx, center in enumerate(path_points):
        cropped_path = crop_circle_region(image_rgb, center, radius, output_dir, idx)
        analyze_circle_image(cropped_path)

    # Step 5: Visualize the result
    output_image = image_rgb.copy()
    for point in path_points:
        cv2.circle(output_image, point, radius, (255, 0, 0), 2)

    plt.figure(figsize=(10, 8))
    plt.imshow(output_image)
    plt.title("Path with Circles and Cropped Analysis")
    plt.axis("off")
    plt.show()
else:
    print("No path detected. Adjust the blue color range in HSV.")
