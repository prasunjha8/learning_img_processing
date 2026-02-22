import cv2
import matplotlib.pyplot as plt

img = cv2.imread("test.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print("Original Image Shape:", img_rgb.shape)
print("new height:", img_rgb.shape[0]*2, "new width:", img_rgb.shape[1]*2)
rows, cols = img.shape[:2]
center = (cols / 2, rows / 2)
M = cv2.getRotationMatrix2D(center, 45, 1)
print("Rotation Matrix:\n", M)

small_nearest = cv2.resize(img_rgb, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
small_linear = cv2.resize(img_rgb, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
small_cubic = cv2.resize(img_rgb, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
rotated_image = cv2.warpAffine(img, M, (cols, rows))

plt.figure(figsize=(12,10))

plt.subplot(3,3,2)
plt.imshow(img_rgb)
plt.title("original")

plt.subplot(3,3,4)
plt.imshow(small_nearest)
plt.title("Nearest")

plt.subplot(3,3,5)
plt.imshow(small_linear)
plt.title("Bilinear")

plt.subplot(3,3,6)
plt.imshow(small_cubic)
plt.title("Bicubic")

plt.subplot(3,3,8)
plt.imshow(rotated_image)
plt.title("rotated_image")

plt.show()