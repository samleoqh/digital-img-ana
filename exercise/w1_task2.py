"""
Author: Qinghui Liu
Date: 30.08.2017
INF4300 Course W1 exercise task 2:
----
Make a function that takes an 8-bits greyscal image as input argument,
and returns a histogram of the graylevel intensities.

"""
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

def histogram(img_gray):
    """
    :param img_gray: 2D numpy array, uint8, (gray image)
    :return:
    intens: 1D numpy array, intens values for histogram
    hist: 1D numpy array, unnormalized hist values
    """
    assert len(img_gray.shape) == 2, "not a gray img"
    assert img_gray.dtype == 'uint8', "not uint8 type"

    intens = np.arange(start=0, stop= 256, step=1)

    hist = np.zeros(len(intens),dtype='int64')

    for i, j in enumerate(intens):
        hist[i] = np.sum((img_gray == j)*1)

    return intens, hist

input_dir = './images'  # change it according to your own image folder
output_dir = './output' # change it according to your own output folder

def main():
    print("Solution to week1 exercise task2")
    num_fig = 1
    #read graylevel image
    img_coins= os.path.join(input_dir,'coins.png')
    img1 = cv2.imread(img_coins, cv2.IMREAD_GRAYSCALE)

    fig = plt.figure(1)
    plt.subplot(1,2,1)
    plt.imshow(img1, cmap='gray')
    plt.subplot(1,2,2)
    intensity, hist = histogram(img1)
    plt.bar(intensity,hist)
    fig.savefig(os.path.join(output_dir,'coins_hist.png'))
    plt.show()


if __name__ == "__main__":
    main()




