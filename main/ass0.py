import cv2 
import numpy as np 
import matplotlib.pyplot as plt
import os

print("Current working dir:", os.getcwd())
img = cv2.imread("test.jpg")

print("Shape:", img.shape)
print("Data type:", img.dtype)
#unit8 represents 0-255 values for each color channel because in bits lets say 2^8 = 256 , thus possible till 0-256

print("channels:", img.shape[2])
print("Total pixels:", img.shape[0] * img.shape[1])

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(img_gray)
plt.title("Original Image")
plt.axis("off")
plt.show()

h = img.shape[0]
cutoff = int(0.25 * h)
img[0:cutoff, :] = [0, 0, 0]
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(img_gray)
plt.title("modified Image")
plt.axis("off")
plt.show()
cv2.imwrite("output_module0.jpg", img)