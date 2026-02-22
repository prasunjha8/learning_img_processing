import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("test.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


kernel = np.ones((7,7), np.float32) / 49
blurred = cv2.filter2D(img_rgb, -1, kernel)
kernel2 = np.ones((3,3), np.float32) / 9
blurred2 = cv2.filter2D(img_rgb, -1, kernel2)
kernel3 = np.array([[ 0, -1,  0],[-1,  5, -1],[ 0, -1,  0]])
manker = cv2.filter2D(img_rgb, -1, kernel3)

plt.figure(figsize=(8,4))
plt.subplot(2,3,1)
plt.imshow(img_rgb)
plt.title("Original")

plt.subplot(2,3,2)
plt.imshow(blurred)
plt.title("Blurred")
plt.show()

plt.subplot(2,3,3)
plt.imshow(blurred2)
plt.title("Blurred2")
plt.show()

plt.subplot(2,3,4)
plt.imshow(manker)
plt.title("manual kernal")
plt.show()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)


magnitude = np.sqrt(sobelx**2 )

plt.imshow(magnitude, cmap='gray')
plt.title("Edge Magnitude")
plt.axis("off")
plt.show()