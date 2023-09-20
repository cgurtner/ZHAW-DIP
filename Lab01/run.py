import os
import skimage
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
