# Lab05
## Task 1 Locate Squares of Given Size

import os
import skimage
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt

folder = os.path.join(os.getcwd() + '/Lab05/pics/')
squaresFilePath = folder + 'squares.tif'
original = skimage.io.imread(squaresFilePath)

## 1

struct5x5 = np.ones((5,5))
struct6x6 = np.ones((6,6))

img5x5_erode = ndimage.binary_erosion(original, structure=struct5x5)
img5x5_dilation = ndimage.binary_dilation(img5x5_erode, structure=struct5x5)

plt.imshow(img5x5_dilation)
plt.show()

img6x6_erode = ndimage.binary_erosion(original, structure=struct6x6)
img6x6_dilation = ndimage.binary_dilation(img6x6_erode, structure=struct6x6)

plt.imshow(img6x6_dilation)
plt.show()

img5x5_isolated = img5x5_dilation ^ img6x6_dilation

plt.imshow(img5x5_isolated)
plt.show()

## 2
def search5x5(image):
    struct = np.array(
        [
            [0, 1, 0], 
            [1, 1, 1], 
            [0, 1, 0]
        ]
    )

    count = 0
    while np.sum(image) > 0:
        seed_x, seed_y = np.transpose(np.nonzero(image))[0]
        seed = np.zeros_like(image)
        seed[seed_x, seed_y] = 1
        
        prev_sum = 0
        while np.sum(seed) != prev_sum:
            prev_sum = np.sum(seed)
            dilated = ndimage.binary_dilation(seed, structure=struct)
            seed = dilated & image
        
        image[seed == 1] = 0
        count += 1

    return count

num5x5 = search5x5(img5x5_isolated)
print(num5x5)

## 3
### Applying a 5x5 filter and use convolution.