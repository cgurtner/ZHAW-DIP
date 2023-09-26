import os
import cv2
import skimage
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

folder = os.path.join(os.getcwd(), 'Lab01/pics/')

# 1.1 Greyscale Images
filename = os.path.join(folder, 'lena_gray.gif')

## 1.1.2 With scikit-image
skimage_lena_grey = skimage.io.imread(filename) 
print(type(skimage_lena_grey)) # numpy.ndarray

## 1.1.2 With PIL
pil_lena_gray = Image.open(filename)
print(type(pil_lena_gray)) # PIL.GifImagePlugin.GifImageFile
print(pil_lena_gray.mode) # is "L" => grayscale, not indexed with a color palette

## 1.1.3 
print(skimage_lena_grey.shape) # (1, 512, 512) => image is 512x512 and has 1 frame (.gif)
frame = skimage_lena_grey[0] # get the first frame of the .gif

print(frame.min()) # 32
print(frame.max()) # 218 ==> It does not span the full grayscale of 0...255

## 1.1.4
# plt.imshow(frame, cmap='gray')
# plt.show()

## 1.1.5
skimage.io.imsave(os.path.join(folder, 'lena_gray.tiff'), skimage_lena_grey)
skimage.io.imsave(os.path.join(folder, 'lena_gray.bmp'), skimage_lena_grey)
skimage.io.imsave(os.path.join(folder, 'lena_gray.png'), skimage_lena_grey)

# 1.2 Color Images
filename = os.path.join(folder, 'lena_color.gif')
skimage_lena_color = skimage.io.imread(filename) 

frame = skimage_lena_color[0] # get first frame of colored .gif
if len(frame.shape) == 3 and frame.shape[2] == 3: # Check if it's RGB
    print('1.2 This image is RGB with 3 channels!')

## 1.2.2
# plt.imshow(frame)
# plt.show()

red_channel = frame[:, :, 0]
green_channel = frame[:, :, 1]
blue_channel = frame[:, :, 2]

print(red_channel, green_channel, blue_channel)

## 1.2.3
converted_gray = skimage.color.rgb2gray(frame)
print(converted_gray.shape) # (512, 512) ==> grayscale, 1 channel

# plt.imshow(converted_gray)
# plt.show()

# 1.3
folder = os.path.join(os.getcwd(), 'Lab01/pics/brain/')
images = sorted(os.listdir(folder))

## remove .DS_Store file if it exists because I'm working with macOS
images = [img for img in images if img != '.DS_Store']

slices = []

for img in images:
    path = os.path.join(folder, img)
    img = skimage.io.imread(path)

    ## 1.3.1 display individual images
    # plt.imshow(img)
    # plt.show()

    slices.append(img)

## 1.3.2 save sequenze as .gif
output_file = 'brain_sequence.gif'
pil_img = [Image.fromarray(slice) for slice in slices]
pil_img[0].save(output_file, save_all=True, append_images=pil_img[1:], duration=100, loop=100)

## 1.3.3 show single images in a stack
stacked_singles = np.vstack(slices)
# plt.imshow(stacked_singles)
# plt.show()

## 1.3.4 parallel to frontal plane
stack = np.stack(slices, axis=2)

s, M, N = stack.shape  # s is x-direction, M is y-direction, N is the number of slices
center = s // 2  # center slice

frontal = stack[center, :, :]

dim_max = max(M, N)
frontal_slice = np.flip(cv2.resize(frontal, (dim_max, dim_max), interpolation=cv2.INTER_LINEAR))

# plt.imshow(frontal_slice)
# plt.show()

## 1.3.5 parallel to the sagittal plane
center = M // 2 # re-center according to M

sagittal = stack[:, center, :]
max = max(sagittal.shape)

sagittal_resize = np.flip(cv2.resize(sagittal, (max, max), interpolation=cv2.INTER_LINEAR))
# plt.imshow(sagittal_resize)
# plt.show()
