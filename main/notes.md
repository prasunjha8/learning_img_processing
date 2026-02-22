Foundations of Image Processing & Computer Vision



Note: Image processing is not about calling OpenCV functions. It is about manipulating memory, geometry, and frequencies. This document explains the physics and math behind the code.

Meta-Lesson: Environment & Execution Control

Before touching a single pixel, you must control your execution environment. Software engineering is 80% environment control and 20% algorithms.

Virtual Environments (venv): Environments are isolation chambers. They prevent dependency chaos. Never stack environments (e.g., Conda + pyenv + venv). Pick one and stick to it.

Activation: source venv/bin/activate. Always verify with which python.

The "Silent None" Bug: If cv2.imread("image.jpg") fails to find the file because you are running the script from the wrong directory, it does not throw an error. It silently returns None. When code fails, check:

Which Python interpreter is running?

What is the current working directory (os.getcwd())?

Is the image path correct relative to the execution directory?

Module 0: What an Image Actually Is (Memory & Matrices)

An image is not a picture; it is a 3D Tensor (Matrix). Everything in computer vision stems from manipulating this memory space.

1. Core Concepts

Shape (H, W, C):

Height (Rows / y-axis)

Width (Columns / x-axis)

Channels (Depth / Colors)

Data Type (uint8): Pixels are typically 8-bit unsigned integers.

$2^8 = 256$. The range is strictly 0 to 255.

Each channel takes exactly 1 Byte of memory.

Memory size = $Height \times Width \times Channels$ bytes (img.nbytes).

Color Space: OpenCV reads images in BGR (Blue, Green, Red) format, not RGB. Matplotlib expects RGB. You must convert before plotting: cv2.cvtColor(img, cv2.COLOR_BGR2RGB).

2. Matrix Discipline (Slicing)

To modify an image, you do not "draw." You slice the tensor and assign values.

# Select rows 50 to 149, cols 50 to 149, and set to Red (BGR)
img[50:150, 50:150] = [0, 0, 255] 


Note: Grayscale conversion weights ($0.299R + 0.587G + 0.114B$) exist because the human eye is biologically more sensitive to green light.

Module 1: Image Geometry & Interpolation

When you resize or rotate an image, you are transforming coordinate systems.

1. Interpolation (Guessing missing pixels)

When making an image larger, the computer must invent pixels that did not exist.

INTER_NEAREST: Picks the closest existing pixel. Fast but blocky.

INTER_LINEAR: Takes a weighted average of the 4 nearest neighbors. Smooth.

INTER_CUBIC: Fits a cubic polynomial to the 16 nearest neighbors. Smoothest, but computationally heavier.

2. Coordinate Mapping & Boundaries

When you rotate an image (e.g., 45 degrees), some pixels map to coordinates outside the new bounding box.

The "black corners" that appear are not void fillers; they are regions where the inverse-mapped source coordinates fell outside the valid boundaries of the original image matrix. OpenCV fills these with a default border value (0 = black).

3. Matplotlib Grid Geometry

Plotting is architecture. Define one grid and stick to it.

fig, axs = plt.subplots(3, 3, figsize=(12,10)) # 12 inches wide, 10 tall
axs[0, 0].imshow(img) # Top left


Module 2: Convolution & Calculus on Pixels

Almost all classical computer vision (blur, sharpen, edge detection) is Convolution: sliding a small matrix (a kernel) over the image, multiplying overlapping values, and summing them up.

1. Blurring (Low-Pass Filter)

A normalized box filter (e.g., a $5 \times 5$ matrix of 1s divided by 25) averages the neighborhood.

Why normalize (sum = 1)? If the kernel sum is $> 1$, the image gets brighter. If $< 1$, it gets darker. Summing to $1$ preserves energy/brightness.

Blurring removes rapid intensity changes (high frequencies) and softens edges.

2. Sharpening (High-Pass Filter)

kernel = np.array([[ 0, -1,  0],
                   [-1,  5, -1],
                   [ 0, -1,  0]], np.float32)


This kernel takes the center pixel ($5\times$) and subtracts the neighbors.

Mathematically: $Original + (Original - Surrounding\_Average)$.

It boosts the difference signal (high frequencies/edges), making the image sharper.

3. Data Types in Convolution (float32 vs uint8)

Why do we use np.float32 for kernels and cv2.CV_64F for output depths?

Fractions: A normalized kernel requires decimals (e.g., $1/25 = 0.04$). Integers would truncate this to $0$.

Overflow/Underflow: Convolution can produce numbers $> 255$ or $< 0$. If you use uint8, these values instantly clip, destroying mathematical information. Compute in float, then normalize back to uint8.

4. Sobel & Edge Detection (Calculus)

Edges are places where brightness changes rapidly. Vision is a gradient machine.

Sobel X: Computes $\partial I / \partial x$. Highlights vertical edges (intensity changing left-to-right).

Sobel Y: Computes $\partial I / \partial y$. Highlights horizontal edges.

Magnitude: $\sqrt{G_x^2 + G_y^2}$. Gives the absolute edge strength regardless of orientation.

Module 3: Frequency Domain (Fourier Transforms)

Images can be thought of as a sum of 2D sine/cosine waves.

Smooth regions = Low frequency.

Sharp edges / Noise = High frequency.

1. The Fourier Transform (np.fft.fft2)

Converts an image from the Spatial Domain (pixels) to the Frequency Domain (waves).

fftshift moves the lowest frequencies (the DC component, representing overall brightness) to the dead center of the image.

2. The Convolution Theorem

Convolution in the spatial domain = Multiplication in the frequency domain.

Applying a Blur kernel in space is mathematically identical to multiplying the frequency spectrum by a mask that dims the outer edges (high frequencies).

3. Frequency Surgery (High-Pass Masking)

By manipulating the frequency domain directly, we can edit the image's DNA:

Transform to frequency space.

Create a mask: a canvas of 1s, with a black square of 0s in the center.

Multiply the shifted FFT by this mask (zeroing out the low frequencies).

Inverse FFT back to the spatial domain.

Result: An image containing only edges and texture. The overall brightness and smooth gradients are completely gone.
