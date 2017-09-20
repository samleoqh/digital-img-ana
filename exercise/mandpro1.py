# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: #
# Date: 20.09.2017                                                            #
# Author: Qignhui L                                                           #
#                                                                             #
# INF4300 Mandatory term project 2017 part-1                                  #
# Segmentation of textured regions in an image                                #
#                                                                             #
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: #
"""
Task:
Describe textured regions in an image, compute and visualize GLCM's from
each texture, extract GLCM feature images, and segment these images.
"""

# Standard library imports
import os
import numpy as np

# Third party imports
import matplotlib.pyplot as plt
import cv2


def main():
    # testing the subimage function
    filepath = "./images/mosaic2.png"  # image file full path
    output = "./output"
    suffix = os.path.splitext(os.path.basename(filepath))[0]

    for k, patch in enumerate(subimage(filepath), 1):
        path = os.path.join(output, "subimg-{0}-{1}.png".format(suffix, k))
        cv2.imwrite(path, patch)


def subimage(path_image, height=None, width=None, stepsize=1):
    """
    Perform image patch extraction with specific height-width sliding window by stepsize
    if no specific window size, the input image will be split into 4 patches equally as default
    :param path_image: full path and file name
    :param height:  patch height cropped
    :param width:   patch width cropped
    :param stepsize:  sliding step size
    :return: patches extracted
    """
    img = cv2.imread(path_image)
    h, w = img.shape[:2]

    if height is None or width is None:
        height = int(h / 2)
        width = int(w / 2)
        stepsize = width

    for x in range(0, h, stepsize):
        px = x
        end_x = x + height
        if end_x > h:
            end_x = h
            px = max(end_x - height, 0)

        for y in range(0, w, stepsize):
            py = y
            end_y = y + width
            if end_y > w:
                end_y = w
                py = max(end_y - width, 0)

            yield img[px:end_x, py:end_y]


if __name__ == '__main__':
    main()
