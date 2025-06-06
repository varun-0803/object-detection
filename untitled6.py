import cv2
import matplotlib.pyplot as plt

# Read the image from file
img = cv2.imread(r"C:\Users\KML\Downloads\Dog_Breeds.jpg")  # Replace with your image path

# Convert BGR (OpenCV default) to RGB for matplotlib display
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur
blurred = cv2.GaussianBlur(gray, (7, 7), 1.5)

# Edge detection using Canny
edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

# Display images
plt.figure(figsize=(12, 8))

plt.subplot(2,2,1)
plt.title('Original Image')
plt.imshow(img_rgb)
plt.axis('off')

plt.subplot(2,2,2)
plt.title('Grayscale Image')
plt.imshow(gray, cmap='gray')
plt.axis('off')

plt.subplot(2,2,3)
plt.title('Blurred Image')
plt.imshow(blurred, cmap='gray')
plt.axis('off')

plt.subplot(2,2,4)
plt.title('Edges (Canny)')
plt.imshow(edges, cmap='gray')
plt.axis('off')

plt.show()
