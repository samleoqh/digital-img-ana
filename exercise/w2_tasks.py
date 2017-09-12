# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: #
# Date: 10.09.2017                                                            #
# Author: Qignhui L                                                           #
#                                                                             #
# Solution proposal of the week2 exercise programming in                      #
# Digital image analysis at UiO                                               #
#                                                                             #
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: #
"""
Task1:
To implement your own GLCM-function that takes as an input an image window
and the number of image grayscales and outputs a co-occurance matrix.
Derive variance, contrast and entropy from the GLCM of a sliding windown of a suitable size.
"""

# Standard library imports
import os
import numpy as np

# Third party imports
import matplotlib.pyplot as plt
import cv2


def main():
    input_dir = './images'  # change it according to your own image folder
    output_dir = './output'  # change it according to your own output folder
    #img_name = os.path.join(input_dir, 'zebra_1.tif')
    img_name = os.path.join(input_dir, 'coins.png')
    features = ['min', 'max', 'mean', 'std_dev', 'skewness', 'kurtosis',
                 'entropy', 'energy', 'smoothness', 'coefficient']

    # Arbitrarily create an 2-D array for test
    # img = np.array([[1, 2, 3],
    #                   [4, 5, 6],
    #                   [7, 8, 9]])

    img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
    img_inf = basic_info(img)
    fig_num = imagesc_hist(img, "Original Image")
    print("-----original image ------------")
    for keys in features:
        print(keys,":",img_inf[keys])

    #img2 = cv2.equalizeHist(img)
    img2 = hist_equalization(img)
    img_inf = basic_info(img2)
    fig_num = imagesc_hist(img2,"Hist equalized",fig_num)
    print("-----hist equalized image ------------")
    for keys in features:
        print(keys,":",img_inf[keys])

    img3 = requantize(img2, level_num=8)
    img_inf = basic_info(img3)
    fig_num = imagesc_hist(img3, "Requantized", fig_num)
    print("-----requantized image ------------")
    for keys in features:
        print(keys, ":", img_inf[keys])


    plt.show()

def basic_info(image):
    """
    Perform some basic (1-order statistic) texture features for gray-level images.
    :param image: 2-D gray-level image
    :return: info_list, some basic features
    """
    info_list = {'min': 0, 'max': 0,
                 'mean': 0, 'std_dev': 0,
                 'skewness': 0, 'kurtosis': 0,
                 'entropy': 0, 'energy': 0,
                 'smoothness': 0, 'coefficient': 0}
    #hist, - = np.histogram(image.flatten(), bins=256, range=[0, 255], density=False)

    info_list['min'] = image.min()
    info_list['max'] = image.max()

    #hist = cv2.calcHist(image,[0],None,[256],[0,256])
    hist,_ = np.histogram(image.flatten(), bins=256, range=[0, 255], density=False)
    hx = hist.ravel()/hist.sum()
    # mean = np.mean(image.flatten())
    x = np.arange(256)
    mean = hx.dot(x)

    info_list['mean'] = mean
    variance = ((x - mean)**2).dot(hx)
    std = np.sqrt(variance)
    info_list['std_dev'] = std
    info_list['skewness'] = ((x - mean)**3).dot(hx)/std**3          # different with lecture notes
    info_list['kurtosis'] = (((x - mean)**4)*hx).sum()/std**4 - 3    # different with lecture notes
    info_list['energy'] = (hx*hx).sum()
    info_list['smoothness'] = 1 - 1/(1+variance)
    info_list['coefficient'] = std/mean

    # ref: https://stackoverflow.com/questions/16647116/faster-way-to-analyze-each-sub-window-in-an-image
    log_h = np.log2(hx+0.00001)
    info_list['entropy'] = -1 * (log_h*hx).sum()

    return info_list

def imagesc_hist(f_img=None, name='Demo', fig_num=0):
    """
    Visualizing rectangular 2D arrays in Python and Matplotlib
    the way you do with Matlab’s imagesc and then plot tis
    gray level histogram
    :type fig_num: object
    :param f_img: Gray level image or 2-D array
    :param name:  Suptitle of the figure to show
    :param fig_num: Number of the figure created
    :return: Figure number for next show
    """
    def extents(f):
        delta = f[1] - f[0]
        return [f[0] - delta / 2, f[-1] + delta / 2]

    if f_img is None:
        # Arbitrarily create an 2-D array for demo
        f_img = np.array([[1, 0, 2],
                          [7, 8, 0],
                          [2, 4, 3]])
        f_img.astype('uint8')

    column_x = np.linspace(0, f_img.shape[1] - 1, f_img.shape[1])  # equal to column axis
    row_y = np.linspace(f_img.shape[0] - 1, 0, f_img.shape[0])  # equal to row axis

    plt.figure(fig_num)
    plt.suptitle(name)
    plt.subplot(1,2,1)
    plt.title('Image')
    plt.imshow(f_img, aspect='auto', interpolation='none', cmap='gray',
               extent=extents(column_x) + extents(row_y), origin='upper')
    plt.xticks([]), plt.yticks([]) # uncomment it if showing xticks

    plt.subplot(1,2,2)
    plt.title('Histogram')
    plt.hist(f_img.ravel(), 256)
    fig_num += 1

    return fig_num

def requantize(img, level_num=8):
    """
    Perform requantization on input image
    :param img: Gray image or 2-D array
    :param level_num:
    :return: 2-D image
    """
    M, N = img.shape
    level_space = np.linspace(0,255,level_num)
    out_img = np.zeros([M,N],dtype='uint8')
    for i in range(M):
        for j in range(N):
            out_img[i,j] = min(level_space,key=lambda x: abs(x-img[i,j]))

    return out_img.astype('uint8')


def hist_equalization(img):
    """
    Perform histogram equalization -- similar with cv2.equalizeHist(img)
    :param image: 2-D gray image, M x N shape
    :return: 2-D array, grey image M xN shape
    """
    M, N = img.shape

    histogram, _ = np.histogram(img.flatten(), bins=256, range=[0, 256], density=True)

    # Cumulative mass function on this histogram.
    cumulative = np.cumsum(histogram)

    # Use this to equalize the image.
    out_img = np.zeros([M, N])
    for i in range(M):
        for j in range(N):
            out_img[i, j] = np.floor(255. * cumulative[img[i, j]])

    return out_img



if __name__ == "__main__":
  main()