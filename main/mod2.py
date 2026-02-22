import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("test.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

kernel = np.ones((10,10), np.float32) / 100
blurred = cv2.filter2D(img_rgb, -1, kernel)

plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.imshow(img_rgb)
plt.title("Original")

plt.subplot(1,2,2)
plt.imshow(blurred)
plt.title("Blurred")
plt.show()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

magnitude = np.sqrt(sobelx**2 + sobely**2)

plt.imshow(magnitude, cmap='gray')
plt.title("Edge Magnitude")
plt.axis("off")
plt.show()