import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Load image
print("Current working dir:", os.getcwd())

img = cv2.imread("test.jpg")

print("Shape:", img.shape)
print("Data type:", img.dtype)

# Show using matplotlib (convert BGR → RGB)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis("off")
plt.show()

# Access a pixel
print("Pixel at (100,100):", img[100,100])

# Modify a region
img[100:150, 50:150] = [0, 0, 255]  # red square (BGR)

img_rgb2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb2)
plt.title("Modified Image")
plt.axis("off")
plt.show()