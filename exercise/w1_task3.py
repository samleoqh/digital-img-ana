"""
Author: Qinghui Liu
Date: 30.08.2017
INF4300 Course W1 exercise task 3:
----
a) use operators <, >, >=, <= to threshold images by an arbitrary threshold.
b) compute an optimal threshold with Otsu's algorithm
c) compare a & b by displaying them and see the differences, and display the differences

"""

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

input_dir = './images'  # change it according to your own image folder
output_dir = './output' # change it according to your own output folder

def main():
    print('solution to week1 task3')

    img = cv2.imread(os.path.join(input_dir,'coins.png'),cv2.IMREAD_GRAYSCALE)

    # manually set an arbitrary threshold
    arbi_th = 127
    th1 = (img <= arbi_th) * 0 + (img > arbi_th)*255

    Otsu_th, th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    gaus_blur = cv2.GaussianBlur(img, (5,5),0)
    ret3,th3 = cv2.threshold(gaus_blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU )
    # plot all the images and their histograms
    images = [img, 0, th1,
              img, 0, th2,
              gaus_blur,0,th3]
    titles = ["Original image", "histogram",'Arbitrary Threshold 127',
              "Original image", 'histogram',"Otsu's Threshold",
              "Gaus Blur img", 'Histogram',"Otsu's threshold"]

    print (arbi_th,Otsu_th,ret3)

    for i in range(3): # xrange for Python 2
        plt.subplot(3,3,i*3+1), plt.imshow(images[i*3],'gray')
        plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
        plt.subplot(3,3,i*3+2), plt.hist(images[i*3].ravel(),256)
        plt.title(titles[i*3+1]), plt.xticks([]),plt.yticks([])
        plt.subplot(3,3,i*3+3), plt.imshow(images[i*3+2],'gray')
        plt.title(titles[i*3+2]),plt.xticks([]),plt.yticks([])

    plt.show()

if __name__ == "__main__":
    main()