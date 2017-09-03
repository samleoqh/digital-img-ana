"""
Author: Qinghui Liu
Date: 03.09.2017
INF4300 Course W1 exercise task 4:
----
computer gradient magnitude of a image (football.jpg)
rescale it to take values 0~255
threshold it at a graylevel e.g 122,
use this to obtain an image containing only the seam of the ball.
"""

import os
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import cv2

input_dir = './images'  # change it according to your own image folder
output_dir = './output' # change it according to your own output folder


def main():
    print('Solution to week1 task4')

    #read image
    img = cv2.imread(os.path.join(input_dir,'football.jpg'),cv2.IMREAD_GRAYSCALE)
    #img = img.astype(np.float32)

    #gradient mask using sobel operators
    maskx = np.array([[-1,-2,-1],
                      [0,0,0],
                      [-1,2,-1]])
    masky = np.array([[-1,0,1],
                      [-2,0,2],
                      [-1,0,1]])
    # compute gx - vertical gradient, gy - horizontal gradient
    gx = signal.convolve2d(img, maskx, mode='same', boundary='symm')
    gy = signal.convolve2d(img, masky, mode='same', boundary='symm')

    #print(gx[1:3,1:3], gy[1:3,1:3])

    # compute gm - gradient magnitude
    gm = np.sqrt(gx**2 + gy**2)
    # scale its range in 0~255
    gm = 255*(gm - np.min(gm))/(np.max(gm)-np.min(gm))


    th1 = 180
    imgth1 = (gm < th1) * 0 + (gm >= th1) * 255
    Otsu_th,imgth2 = cv2.threshold(gx.astype(np.uint8), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    plt.figure(1)
    plt.subplot(2,2,4), plt.imshow(gm,'gray'), plt.title('gm image')
    plt.subplot(2,2,2), plt.imshow(imgth1,'gray'), plt.title('th = 180')
    plt.subplot(2,2,3), plt.imshow(imgth2,'gray'), plt.title('th = Otsu')
    plt.subplot(2,2,1),plt.hist(gm.ravel(),256), plt.title('histogram of gm')
    plt.show()


if __name__ == "__main__":
    main()