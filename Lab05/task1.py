# Lab05
## Task 1

import os
import skimage
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt

folder = os.path.join(os.getcwd() + '/Lab05/pics/')
squaresFilePath = folder + 'squares.tif'
original = skimage.io.imread(squaresFilePath)

## 1

struct = np.ones((5,5))

img_erode = ndimage.binary_erosion(input=original, structure=struct)
plt.imshow(img_erode)
plt.show()
img_erode_5x5 = ndimage.binary_erosion(input=img_erode, structure=np.ones((2,2)))
plt.imshow(img_erode_5x5)
plt.show()

img_diff = img_erode ^ img_erode_5x5 # woher kommen die kanten nach dem diff?
plt.imshow(img_diff)
plt.show()

img_dilation = ndimage.binary_dilation(input=img_diff, structure=struct)

plt.imshow(img_dilation)
plt.show()

## 2



