from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage, fft
from PIL import Image


# ----------------------------------------------------- load image ---------------------------------------------
folder = os.path.join(os.getcwd(), 'Lab04/')
file_name = os.path.join(folder, 'blurred_image.jpg')

# Open the image with pillow and convert to numpy array
image = Image.open(file_name)
color_pixels = np.asarray(Image.open(file_name))
gray_pixels = np.asarray(Image.open(file_name).convert('L'))

# summarize some details about the image
print(image.format)
print('numpy array:', gray_pixels.dtype)
print(gray_pixels.shape)

# -------------------------------------------- generate the motion blur filter -----------------------------------------
nFilter = 91
angle = 45
my_filter = np.zeros((nFilter, nFilter))
my_filter[nFilter//2, :] = 1.0 / nFilter
my_filter = scipy.ndimage.rotate(my_filter, angle, reshape=False)

nRows = gray_pixels.shape[0]
nCols = gray_pixels.shape[1]
nFFT = 1024

image_spectrum = scipy.fft.fft2(gray_pixels, (nFFT, nFFT))
filter_spectrum = scipy.fft.fft2(my_filter, (nFFT, nFFT))

modified_image_spectrum = image_spectrum * filter_spectrum
modified_image = scipy.fft.ifft2(modified_image_spectrum)
modified_image = np.real(modified_image)[nFilter:nRows + nFilter, nFilter:nCols + nFilter]

# --------------------------------------------------- reconstruct the image --------------------------------------------

K = 0.02
H_star = np.conj(filter_spectrum)
F_hat = (H_star * image_spectrum) / (np.abs(filter_spectrum)**2 + K)

reconstructed_image = np.real(scipy.fft.ifft2(F_hat))
reconstructed_image = reconstructed_image[:gray_pixels.shape[0], :gray_pixels.shape[1]]  # Crop to original size

# --------------------------------------------------------- display images ---------------------------------------------
fig = plt.figure(1)
plt.subplot(2, 2, 1)
plt.title('Original Image')
plt.imshow(gray_pixels, cmap='gray')

plt.subplot(2, 2, 2)
plt.title('Motion Blur Filter')
plt.imshow(my_filter, cmap='gray')

plt.subplot(2, 2, 3)
plt.title('Modified Image')
plt.imshow(modified_image, cmap='gray')

plt.subplot(2, 2, 4)
plt.title('Reconstructed Image')
plt.imshow(reconstructed_image, cmap='gray')

plt.show()
