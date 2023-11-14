import cv2
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from PIL import Image
import os

def basic_thresholding(image, epsilon=1):
    Tn = np.mean(image.flatten()).astype(int)
    delta = np.inf

    while delta >= epsilon:
        G1 = image[image <= Tn]
        G2 = image[image > Tn]

        m1 = np.mean(G1) if len(G1) > 0 else 0
        m2 = np.mean(G2) if len(G2) > 0 else 0

        Tn_plus_1 = 0.5 * (m1 + m2)
        delta = abs(Tn_plus_1 - Tn)
        Tn = Tn_plus_1

    bin_img = image > Tn
    return bin_img, Tn

def my_otsu(image):
    pixel_counts = np.bincount(image.flatten(), minlength=256)
    total_pixels = image.size
    
    current_max_variance = 0
    threshold = 0
    
    cumulative_sum = np.cumsum(pixel_counts)
    cumulative_mean = np.cumsum(np.arange(len(pixel_counts)) * pixel_counts)
    
    for k in range(len(pixel_counts)):
        prob1 = cumulative_sum[k] / total_pixels
        prob2 = 1 - prob1
        
        mean1 = cumulative_mean[k] / cumulative_sum[k] if cumulative_sum[k] > 0 else 0
        mean2 = (cumulative_mean[-1] - cumulative_mean[k]) / (total_pixels - cumulative_sum[k]) if cumulative_sum[k] != total_pixels else 0
        
        variance_between = prob1 * prob2 * (mean1 - mean2) ** 2
        
        if variance_between > current_max_variance:
            current_max_variance = variance_between
            threshold = k
    
    binary_image = image >= threshold
    return binary_image, current_max_variance, threshold


def main():
    dir = os.path.join(os.getcwd() + '/Lab08/Otsu/')
    image = np.asarray(Image.open(dir + 'thGonz.tif'))

    if len(image.shape)==3:
        image=image[:,:,0]

    binary_image, t = basic_thresholding(image)
    print("Basic Thresholding. Output Threshold: "+str(t))

    plt.imshow(binary_image.astype(int), cmap="gray")
    plt.show()

    binary_image, between_class_variance, threshold, separability = my_otsu(image)

    print("Otsu's Method. Output Threshold: "+str(threshold))
    print("Separability: "+str(separability))

    plt.plot(between_class_variance*100)
    plt.show()

    plt.imshow(binary_image.astype(int), cmap="gray")
    plt.show()

if __name__ == "__main__":
    main()
