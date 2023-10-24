# Lab05
## Task 2 Counting Blood Cells

import os
from skimage import io, segmentation
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt

folder = os.path.join(os.getcwd(), 'Lab05/pics/')
image = io.imread(folder + 'bloodCells.png', as_gray=True)

min, max = image.min(), image.max()
threshold = (min + max) / 2
binarized_global = image > threshold

removed_on_border_before_filling = segmentation.clear_border(binarized_global)

filled_holes = ndimage.binary_fill_holes(binarized_global)
filled = segmentation.clear_border(filled_holes)

label_image, num_features = ndimage.label(filled)

fig, axarr = plt.subplots(1, 3, figsize=(20, 5))

axarr[0].imshow(image, cmap='gray')
axarr[0].set_title('Original')
axarr[0].axis('off')

axarr[1].imshow(removed_on_border_before_filling, cmap='gray')
axarr[1].set_title('Not filled')
axarr[1].axis('off')

axarr[2].imshow(filled, cmap='gray')
axarr[2].set_title('Filled')
axarr[2].axis('off')

fig.suptitle('Labeled Blood Cells (Total: {})'.format(num_features))
plt.show()

counts = np.bincount(label_image.ravel())
counts = counts[1:] # remove the background count

print(counts)

plt.figure(figsize=(10, 6))
plt.hist(counts)
plt.title('Count of Blood Cells')
plt.xlabel('Pixels')
plt.ylabel('Blood Cells')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.show()