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


    # cut the input image into 4 parts equally by subimage() function
    # for k, patch in enumerate(subimage(filepath), 1):
    #     path = os.path.join(output, "subimg-{0}-{1}.png".format(suffix, k))
    #     cv2.imwrite(path, patch)


def requantize(image, level_num=8):
    """
    Perform requantization on input gray image
    :param img: Gray image or 2-D array
    :param level_num:
    :return: 2-D image
    """
    M, N = image.shape
    level_space = np.linspace(0,255,level_num)
    out_img = np.zeros([M,N],dtype='uint8')
    for i in range(M):
        for j in range(N):
            out_img[i,j] = min(level_space,key=lambda x: abs(x-image[i,j]))

    return out_img.astype('uint8')


def glcm_window_img(gray_image, neighbor=2, current_row=0, current_col=0, fill='constant'):
    """
    Get an glcm window image given defined neghbourhood or window size on the original image with
    specific pixel location index

    :param gray_image: input gray level image
    :param neighbor: window order - size defined by neighborhood
    :param current_row: give current location of pixel - row index
    :param current_col: give current location of pixel - column index
    :param fill: boundary filling flag, now only have 2 types, constant or mirror
    :return: window image
    """
    max_row, max_col = gray_image.shape
    win_shape = (2 * neighbor + 1, 2 * neighbor + 1)  # window shape - 5 x 5
    win_img = np.zeros(win_shape)

    for row_offset in range(-1 * neighbor, neighbor + 1):
        for col_offset in range(-1 * neighbor, neighbor + 1):

            cp_i = current_row + row_offset # pixel row index on the image with offsets
            cp_j = current_col + col_offset # pixel col index on the image with offsets

            if 0 <= cp_i < max_row and 0 <= cp_j < max_col:
                win_img[neighbor + row_offset, neighbor + col_offset] = gray_image[cp_i, cp_j]
            else:
                if fill is 'constant':
                    win_img[neighbor + row_offset, neighbor + col_offset] = 0
                elif fill is 'mirror':
                    if cp_i >= max_row:
                        cp_i = max_row - row_offset
                    if cp_j >= max_col:
                        cp_j = max_col - col_offset
                    if cp_i < 0:
                        cp_i = -1 - cp_i
                    if cp_j < 0:
                        cp_j = -1 - cp_j

                    win_img[neighbor + row_offset, neighbor + col_offset] = gray_image[cp_i, cp_j]

    return win_img


def direction_glcm(win_image, direction='horizontal', step=1, weight=1):
    """
    Perfome directional glcm computing

    four directions

                diagonal1 45 degree
               .
             .
           .
         .
       .
     ............ horizontal
     ..
     .  .
     .    .
     .      .
     .        .
     .          .diagonal2 -45 degree

    vertical

    :param win_image: input window image
    :param direction: giving 4 direction
    :param step: pixel pare step, default 1 >= 1 <= image size
    :param weight: default weight value >0
    :return:
    """
    M, N = win_image.shape
    levels = list(np.unique(win_image))
    num_levels = len(levels)

    glcm = np.zeros([num_levels, num_levels], dtype='uint8')

    if direction is 'horizontal':
        i_range = list(range(M))
        j_range = list(range(N - step))
        for i in i_range:
            for j in j_range:
                pixel_1 = win_image[i, j]
                pixel_2 = win_image[i, j + step]
                co_i = levels.index(pixel_1)
                co_j = levels.index(pixel_2)
                glcm[co_i, co_j] += weight
    elif direction is 'vertical':
        i_range = list(range(M - step))
        j_range = list(range(N))
        for i in i_range:
            for j in j_range:
                pixel_1 = win_image[i, j]
                pixel_2 = win_image[i + step, j]
                co_i = levels.index(pixel_1)
                co_j = levels.index(pixel_2)
                glcm[co_i, co_j] += weight
    elif direction is 'diagonal1':
        i_range = list(range(step, M))
        j_range = list(range(N - step))
        for i in i_range:
            for j in j_range:
                pixel_1 = win_image[i, j]
                pixel_2 = win_image[i - step, j + step]
                co_i = levels.index(pixel_1)
                co_j = levels.index(pixel_2)
                glcm[co_i, co_j] += weight
    elif direction is 'diagonal2':
        i_range = list(range(M - step))
        j_range = list(range(N - step))
        for i in i_range:
            for j in j_range:
                pixel_1 = win_image[i, j]
                pixel_2 = win_image[i + step, j + step]
                co_i = levels.index(pixel_1)
                co_j = levels.index(pixel_2)
                glcm[co_i, co_j] += weight

    return glcm



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
