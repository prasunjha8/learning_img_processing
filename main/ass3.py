import cv2
import numpy as np
import matplotlib.pyplot as plt


img_rgb = cv2.imread("test.jpg", 0)

kernel = np.ones((10,10), np.float32) / 100
#blurred = cv2.filter2D(img_rgb, -1, kernel)
blurred = cv2.GaussianBlur(img_rgb, (11,11), 0)

f = np.fft.fft2(img_rgb)
fshift = np.fft.fftshift(f)

f2 = np.fft.fft2(blurred)
fshift2 = np.fft.fftshift(f2)

magnitude_spectrum = 20 * np.log(np.abs(fshift))
magnitude_spectrum_blurred = 20 * np.log(np.abs(fshift2))

plt.figure(figsize=(8,4))
plt.subplot(1,4,1)
plt.imshow(img_rgb)
plt.title("Original")

plt.subplot(1,4,2)
plt.imshow(magnitude_spectrum)
plt.title("magnitude_spectrum")

plt.subplot(1,4,3)
plt.imshow(blurred, cmap='gray')
plt.title("Blurred")

plt.subplot(1,4,4)
plt.imshow(magnitude_spectrum_blurred, cmap='gray') 
plt.title("magnitude_spectrum_blurred")
plt.show()


rows, cols = img_rgb.shape
crow, ccol = rows // 2 , cols // 2

mask = np.ones((rows, cols), np.uint8)
size = 30
mask[crow-size:crow+size, ccol-size:ccol+size] = 0
fshift_filtered = fshift * mask
f_ishift = np.fft.ifftshift(fshift_filtered)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)


plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.imshow(img_rgb, cmap='gray')
plt.title("Original")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(mask, cmap='gray')
plt.title("High-Pass Mask")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(img_back, cmap='gray')
plt.title("High-Pass Result")
plt.axis("off")

plt.show()