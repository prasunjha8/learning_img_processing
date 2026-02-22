import cv2
import matplotlib.pyplot as plt

img = cv2.imread("test.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Resize to half size
small_nearest = cv2.resize(img_rgb, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
small_linear = cv2.resize(img_rgb, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
small_cubic = cv2.resize(img_rgb, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)

plt.figure(figsize=(12,8))

plt.subplot(1,3,1)
plt.imshow(small_nearest)
plt.title("Nearest")

plt.subplot(1,3,2)
plt.imshow(small_linear)
plt.title("Bilinear")

plt.subplot(1,3,3)
plt.imshow(small_cubic)
plt.title("Bicubic")

plt.show()