"""
Author: Qinghui Liu
Date: 30.08.2017
INF4300 Course W1 exercise No1
Need to refactor code
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import signal #refer to https://docs.scipy.org/doc/scipy/reference/signal.html

input_dir = './images'  # change it according to your own image folder
output_dir = './output' # change it according to your own output folder

def main():
    print("Solution to week1 exercise#1 of INF4300")
    num_fig = 1
    #read graylevel image
    img_football= os.path.join(input_dir,'football.jpg')
    img1 = cv2.imread(img_football, cv2.IMREAD_GRAYSCALE)

    #show gray images
    #can wrape this snippet into a function for plotting images
    fig = plt.figure(num_fig)
    num_fig += 1
    plt.subplot(1,2,1)
    plt.imshow(img1,cmap='gray') # note the parameter: cmap
    plt.subplot(1,2,2)
    plt.imshow(img1,interpolation='none') # refer to opencv documents
    plt.show()

    #save images with other png formate
    out_file1=os.path.join(output_dir,'football.png')
    fig.savefig(out_file1,bbox_inches='tight',pad_inches=0)

    #construct mean kernel
    mean_filter = np.ones((5,5))/25

    # full padding mode
    mean_img1 = signal.convolve2d(img1,mean_filter,mode='full')
    # valid padding mode
    mean_img2 = signal.convolve2d(img1,mean_filter,mode='valid')
    # same padding mode
    mean_img3 = signal.convolve2d(img1,mean_filter,mode='same')
    fig = plt.figure(num_fig)
    num_fig += 1
    plt.subplot(1,3,1)
    plt.imshow(mean_img1,cmap='gray',interpolation='none')
    plt.subplot(1,3,2)
    plt.imshow(mean_img2,cmap='gray',interpolation='none')
    plt.subplot(1,3,3)
    plt.imshow(mean_img3,cmap='gray')
    plt.show()


if __name__ == "__main__":
    main()
