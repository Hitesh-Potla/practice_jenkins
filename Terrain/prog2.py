import cv2
import numpy as np
from collections import Counter

# Step 1: Load the image
image = cv2.imread('6.png')  # Replace with your image path
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR (OpenCV default) to RGB

# Step 2: Create a mask for the circle
height, width, _ = image.shape
center = (width // 2, height // 2)  # Assuming the circle is centered
radius = min(width, height) // 2  # Assuming the circle is the largest possible inside the image
mask = np.zeros((height, width), dtype=np.uint8)
cv2.circle(mask, center, radius, 255, thickness=-1)  # Fill circle with white (255)

# Step 3: Extract pixels inside the circle
pixels_inside_circle = image[mask == 255]  # Get pixels where the mask is 255 (inside the circle)

# Step 4: Count the most frequent color
pixels_as_tuples = [tuple(color) for color in pixels_inside_circle]  # Convert pixels to tuples
# Remove all instances of the target pixel
def is_purple(pixel):
    r, g, b = pixel
    return b > 120 and r > 80 and r < 150 and abs(r - b) < 60

pixels_as_tuples = [tuple(pixel) for pixel in pixels_inside_circle if not is_purple(pixel)]

if pixels_as_tuples:
    most_common_color, count = Counter(pixels_as_tuples).most_common(1)[0]
    print(f"Most Common Color (excluding purple): {most_common_color}, Count: {count}")
else:
    print("No valid colors found after excluding purple pixels.")
