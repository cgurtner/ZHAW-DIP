import os
import cv2
import skimage
from scipy.fft import fft2, ifft2
from scipy.signal import convolve2d
import numpy as np
import matplotlib.pyplot as plt
import time


folder = os.path.join(os.getcwd(), 'Lab04/')

# 1.1
filename = os.path.join(folder, 'MenInDesert.jpg')
img = skimage.io.imread(filename)

## plt.imshow(img)
## plt.show()

print(img.shape) # 736x552, color-space: RGB

# 1.2
## transform to grayscale
grayscale = skimage.color.rgb2gray(img)
transform = fft2(grayscale)

print(transform.dtype) # complex128
print(transform.shape) # it still 736x552 becuse no zero-padding parameter was provided in fft2()

# 1.3.
## magnitude
magnitude = np.abs(transform)
log_magnitude = np.log(magnitude + 1) # +1 to ensure 0 are calculated too

fig, axarr = plt.subplots(1, 3, figsize=(15, 5))

# Original Image
axarr[0].imshow(grayscale, cmap='gray')
axarr[0].set_title('Original Image')
axarr[0].axis('off')

# Magnitude Liniear
axarr[1].imshow(np.fft.fftshift(magnitude), cmap='gray')
axarr[1].set_title('Magnitude Linear')
axarr[1].axis('off')

# Magnitude Log
axarr[2].imshow(np.fft.fftshift(log_magnitude), cmap='gray')
axarr[2].set_title('Magnitude Log')
axarr[2].axis('off')

plt.tight_layout()
plt.show()

# 1.4
inverse = ifft2(transform)
print(transform.dtype) # also complex128 like the original

# 1.5
start_time = time.time()
filtered_image = convolve2d(grayscale, np.ones((9, 9))/81, mode='same', boundary='symm')
filtered_image_time = time.time() - start_time

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(grayscale, cmap='gray')
plt.title('Original')

plt.subplot(1, 2, 2)
plt.imshow(filtered_image, cmap='gray')
plt.title('9x9 Filter')

plt.tight_layout()
plt.show()

# 1.6
filter = np.ones((9, 9))/81
start_time = time.time()
spectrum = np.pad(filter, 
        (
            (0, grayscale.shape[0] - filter.shape[0]), 
            (0, grayscale.shape[1] - filter.shape[1])
        ), 'constant')

fourier_spectrum = fft2(spectrum)
G = transform * fourier_spectrum

filtered_image_fft = np.real(ifft2(G))
filtered_image_fft_time = time.time() - start_time

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(grayscale, cmap='gray')
plt.title('Original')

plt.subplot(1, 2, 2)
plt.imshow(filtered_image_fft, cmap='gray')
plt.title('2D-FFT')

plt.show()

# 1.7 compare
diff = np.abs(filtered_image - filtered_image_fft)
mean_diff = np.mean(diff)
print(mean_diff) # 0.05832842458476699 => relatively small difference

# 1.8

## 2x
upscaled_image = cv2.resize(grayscale, (grayscale.shape[1]*2, grayscale.shape[0]*2), interpolation=cv2.INTER_LINEAR)
upscaled_filter = cv2.resize(filter, (filter.shape[1]*2, filter.shape[0]*2), interpolation=cv2.INTER_LINEAR)

start_time = time.time()
convolve2d(upscaled_image, upscaled_filter, mode='same', boundary='symm')
filtered_image_time_double = time.time() - start_time

start_time = time.time()
upscale_padded = np.pad(upscaled_filter, 
        (
            (0, upscaled_image.shape[0] - upscaled_filter.shape[0]), 
            (0, upscaled_image.shape[1] - upscaled_filter.shape[1])
        ), 'constant')

transform_upscaled = fft2(upscaled_image)
fourier_spectrum_upscaled = fft2(upscale_padded)

G = transform_upscaled * fourier_spectrum_upscaled

filtered_image_fft = np.real(ifft2(G))
filtered_image_fft_time_double = time.time() - start_time

## 4x
upscaled_image = cv2.resize(grayscale, (grayscale.shape[1]*4, grayscale.shape[0]*4), interpolation=cv2.INTER_LINEAR)
upscaled_filter = cv2.resize(filter, (filter.shape[1]*4, filter.shape[0]*4), interpolation=cv2.INTER_LINEAR)

start_time = time.time()
convolve2d(upscaled_image, upscaled_filter, mode='same', boundary='symm')
filtered_image_time_4x = time.time() - start_time

start_time = time.time()
upscale_padded = np.pad(upscaled_filter, 
        (
            (0, upscaled_image.shape[0] - upscaled_filter.shape[0]), 
            (0, upscaled_image.shape[1] - upscaled_filter.shape[1])
        ), 'constant')

transform_upscaled = fft2(upscaled_image)
fourier_spectrum_upscaled = fft2(upscale_padded)

G = transform_upscaled * fourier_spectrum_upscaled

filtered_image_fft = np.real(ifft2(G))
filtered_image_fft_time_4x = time.time() - start_time

print('Spatial: ', filtered_image_time) 
print('FFT: ', filtered_image_fft_time) 

print('Spatial 2x ', filtered_image_time_double) 
print('FFT 2x ', filtered_image_fft_time_double) 

print('Spatial 4x ', filtered_image_time_4x) 
print('FFT 4x ', filtered_image_fft_time_4x) 

# Spatial:  0.055377960205078125
# FFT:  0.03349113464355469

# Spatial 2x  0.7539660930633545
# FFT 2x  0.1135411262512207

# Spatial 4x  14.490129947662354
# FFT 4x  0.5556559562683105