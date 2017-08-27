#opencv basic test script
#Qinghui Liu
#27/08/2017

#File IO
from __future__ import print_function

import cv2
import numpy as np
import os

img = np.zeros((3,3),dtype=np.uint8)
# img = [[0 0 0],
#        [0 0 0],
#        [0 0 0]]

#convert img to Blue-green-red BGR
img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
# img = [[[0 0 0],
#         [0 0 0],
#         [0 0 0]],
#        [[0 0 0],
#         [0 0 0],
#         [0 0 0]],
#        [[0 0 0],
#         [0 0 0],
#         [0 0 0]],
#       ]
#print (img.shape) (3,3,3)

img2 = cv2.imread('wow.png')
grayimg = cv2.imread('wow.png',cv2.IMREAD_GRAYSCALE)
cv2.imwrite('wow2.jpg',img2)
cv2.imwrite('wowgray.jpg',grayimg)

randomarray = bytearray(os.urandom(180000))
flatnparray= np.array(randomarray)
ranImg = flatnparray.reshape(300,600)
rgbImg = flatnparray.reshape(300,200,3)
cv2.imwrite('random.png',ranImg)
cv2.imwrite('grbi.png',rgbImg)
