import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

from scipy import ndimage as ndi

import skimage as skimage
import skimage.segmentation as segmentation
import skimage.feature as feature    
from scipy.ndimage import distance_transform_edt, label
from skimage.feature import peak_local_max  
from skimage.segmentation import watershed
import os


def compute_binary_image_otsu(image, show=False):
    if len(image.shape) == 3:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        
    threshold, image_binary = cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    
    if show:
        plt.figure()
        plt.imshow(image_binary, 'gray')
        plt.title("Binary Image")
        plt.show()

        plt.figure()
        plt.hist(image.ravel(), bins=256, range=(0, 255))
        plt.axvline(threshold, color='r', linestyle='dashed', linewidth=2)
        plt.title("Histogram with Otsu's Threshold")
        plt.show()

    return image_binary


def distance_transform_implementation_example(binaryImage, show=False):
    g = 1000 * np.uint16(binaryImage > 0) 
    for iy in range(1, g.shape[0]-1):
        for ix in range(1, g.shape[1]-1):
            neighborMinValue = np.min([g[iy-1, ix-1], g[iy-1, ix], g[iy, ix-1], g[iy-1, ix+1], g[iy+1, ix+1], g[iy+1, ix], g[iy, ix+1], g[iy+1, ix-1]])
            if g[iy, ix]:
                g[iy, ix] = neighborMinValue + 1

    for iy in range(g.shape[0]-2, 1, -1):
        for ix in range(g.shape[1]-2, 1, -1):
            neighborMinValue = np.min([g[iy-1, ix-1], g[iy-1, ix], g[iy, ix-1], g[iy-1, ix+1], g[iy+1, ix+1], g[iy+1, ix], g[iy, ix+1], g[iy+1, ix-1]])
            if g[iy, ix]:
                g[iy, ix] = neighborMinValue + 1

    if show:
        plt.figure()
        plt.imshow(g, 'gray')
        plt.show()

    return g


def main():
    image = plt.imread("yeast.tif")
    if len(image.shape) == 3:
        image = image[:, :, 0]

    imageBinary = compute_binary_image_otsu(image, show=True)

    imageDistance = distance_transform_edt(imageBinary)

    local_maxi = peak_local_max(imageDistance, indices=True, labels=imageBinary)
    peakCoords = local_maxi

    mask = np.zeros_like(imageBinary, dtype=bool)
    mask[tuple(peakCoords.T)] = True

    seedMask, _ = label(mask)

    labeled_regions = watershed(-imageDistance, seedMask, mask=imageBinary)
    labels = labeled_regions * imageBinary

    plt.figure()
    fig, axes = plt.subplots(ncols=5, figsize=(12, 3), sharex=True, sharey=True)
    ax = axes.ravel()

    ax[0].imshow(-image, cmap=plt.cm.gray, interpolation='none')
    ax[0].set_title('Original Image')
    ax[1].imshow(imageBinary, cmap=plt.cm.gray, interpolation='none')
    ax[1].set_title('Otsu: image_binary')
    ax[2].imshow(imageDistance / np.amax(imageDistance), cmap=plt.cm.jet, interpolation='none')
    ax[3].scatter(peakCoords[:, 1], peakCoords[:, 0], s=10, marker='o')
    ax[4].imshow(labels, cmap=plt.cm.nipy_spectral, interpolation='none')
    ax[4].set_title('Separated objects')

    for a in ax:
        a.set_axis_off()

    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

