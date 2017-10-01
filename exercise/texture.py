# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: #
# Date: 29.09.2017                                                            #
# Author: Qignhui L                                                           #
#                                                                             #
# INF4300 Mandatory term project 2017 part-1                                  #
# Segmentation of textured regions in an image                                #
#                                                                             #
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: #
"""
wrapped up some functions for texture analysis
Import this lib into the main project

"""

# Standard library imports
import numpy as np

# Third party imports
import matplotlib.pyplot as plt
import cv2
# Make plot with vertical (default) colorbar
from matplotlib import cm
import matplotlib.colors as mcolors

# using numba to optimize code
from numba import jit
from skimage.feature import greycomatrix, greycoprops


# these below two functions copied
# from https://stackoverflow.com/questions/18704353/correcting-matplotlib-colorbar-ticks

def colorbar_index(ncolors, cmap):
    cmap = cmap_discretize(cmap, ncolors)
    mappable = cm.ScalarMappable(cmap=cmap)
    mappable.set_array([])
    mappable.set_clim(-0.5, ncolors + 0.5)
    colorbar = plt.colorbar(mappable)
    colorbar.set_ticks(np.linspace(0, ncolors, ncolors))
    colorbar.set_ticklabels(range(ncolors))


def cmap_discretize(cmap, N):
    """Return a discrete colormap from the continuous colormap cmap.

        cmap: colormap instance, eg. cm.jet.
        N: number of colors.

    Example
        x = resize(arange(100), (5,100))
        djet = cmap_discretize(cm.jet, 5)
        imshow(x, cmap=djet)
    """

    if type(cmap) == str:
        cmap = plt.get_cmap(cmap)

    colors_i = np.concatenate((np.linspace(0, 1., N), (0., 0., 0., 0.)))
    colors_rgba = cmap(colors_i)
    indices = np.linspace(0, 1., N + 1)
    cdict = {}
    for ki, key in enumerate(('red', 'green', 'blue')):
        cdict[key] = [(indices[i], colors_rgba[i - 1, ki], colors_rgba[i, ki])
                      for i in range(N + 1)]
    # Return colormap object.
    return mcolors.LinearSegmentedColormap(cmap.name + "_%d" % N, cdict, 1024)


def requantize(image, level_num=8):
    """
    Perform requantization on input gray image
    :param img: Gray image or 2-D array
    :param level_num:
    :return: 2-D image
    """
    M, N = image.shape
    level_space = np.linspace(0, 255, level_num)
    out_img = np.zeros([M, N], dtype='uint8')
    for i in range(M):
        for j in range(N):
            out_img[i, j] = min(level_space, key=lambda x: abs(x - image[i, j]))

    return out_img.astype('uint8')


@jit
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

            cp_i = current_row + row_offset  # pixel row index on the image with offsets
            cp_j = current_col + col_offset  # pixel col index on the image with offsets

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


@jit
def direction_glcm(win_image, direction='horizontal', step=1):
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
    weight = 1

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


@jit
def isotropic_glcm(win_image, weights=[0.25, 0.25, 0.25, 0.25], stepsize=1):
    if weights is None:
        weights = [0.25, 0.25, 0.25, 0.25]
    glcm = direction_glcm(win_image, 'horizontal', step=stepsize) * weights[0]
    glcm += direction_glcm(win_image, 'vertical', step=stepsize) * weights[1]
    glcm += direction_glcm(win_image, 'diagonal1', step=stepsize) * weights[2]
    glcm += direction_glcm(win_image, 'diagonal2', step=stepsize) * weights[3]
    return glcm


@jit
def symmetrise(glcm):
    return glcm + glcm.T


@jit
def normalize(glcm):
    return glcm / float(sum(glcm.flatten()))


@jit
def glcm_measures(glcm, name=None, normed=False, symmetric=False):
    """
    Perform computing all kinds of measures/features based on a given glcm
    :param glcm: input glcm matrix
    :param name: measure's name, if none, return all measures
    :param normed: if normalize the glcm before measuring it
    :param symmetric: if symmetrise the glcm before measuring it
    :return: a measure values or a measure list
    """
    measure_list = dict(max_prob=0, contrast=0, dissimilarity=0, homogeneity=0, ASM=0, energy=0, entropy=0,
                        correlation=0, cluster_shade=0, variance_i=0, variance_j=0, mean_i=0, mean_j=0)

    M, N = glcm.shape

    np.seterr(divide='ignore', invalid='ignore')

    if symmetric:
        # symmetrisation
        glcm = symmetrise(glcm)
        if normed:
            glcm = normalize(glcm)
    else:
        if normed:
            glcm = normalize(glcm)

    flat_glcm = glcm.flatten()
    index_i = np.arange(0, M)  # row index
    index_j = np.arange(0, N)  # column index = row

    sum_v = np.sum(glcm, axis=0)  # sum column[] , vertical
    sum_h = np.sum(glcm, axis=1)  # sum row[] , horizontal

    mean_i = np.dot(index_i, sum_h.flatten())
    mean_j = np.dot(index_j, sum_v.flatten())

    var_i = np.dot((index_i - mean_i) ** 2, sum_h.flatten())
    var_j = np.dot((index_j - mean_j) ** 2, sum_v.flatten())

    stdev_i = np.sqrt(var_i)
    stdev_j = np.sqrt(var_j)

    contrast_weights = np.zeros([M, N])
    dissi_weights = np.zeros([M, N])
    homo_weights = np.zeros([M, N])
    cluster_weights = np.zeros([M, N])
    correl_weights = np.outer((index_i - mean_i), (index_j - mean_j)) / (stdev_i * stdev_j)

    for i in range(M):
        for j in range(N):
            contrast_weights[i, j] = (i - j) ** 2
            dissi_weights[i, j] = abs(i - j)
            homo_weights[i, j] = 1 / (1 + (i - j) ** 2)
            cluster_weights[i, j] = (i + j - mean_i - mean_j) ** 3

    max_prob = np.max(flat_glcm)
    ASM = np.dot(flat_glcm, flat_glcm)
    energy = np.sqrt(ASM)

    # ln = np.log(flat_glcm) here, log(0) = -inf, will have some problem, using np.ma.log instead
    # np.ma.log(0) = -- : not -inf. ? can pass
    ln = np.ma.log(flat_glcm)
    entropy = -np.dot(flat_glcm, ln)

    contrast = np.dot(flat_glcm, contrast_weights.flatten())
    dissimilarity = np.dot(flat_glcm, dissi_weights.flatten())
    homogeneity = np.dot(flat_glcm, homo_weights.flatten())
    correlation = np.dot(flat_glcm, correl_weights.flatten())
    # cluster_shade = np.dot(flat_glcm, cluster_weights.flatten())
    cluster_shade = np.dot(glcm.flatten(), cluster_weights.flatten())

    measure_list['max_prob'] = max_prob
    measure_list['contrast'] = contrast
    measure_list['dissimilarity'] = dissimilarity
    measure_list['homogeneity'] = homogeneity
    measure_list['ASM'] = ASM
    measure_list['energy'] = energy
    measure_list['entropy'] = entropy
    measure_list['correlation'] = correlation
    measure_list['cluster_shade'] = cluster_shade
    measure_list['variance_i'] = var_i
    measure_list['variance_j'] = var_j
    measure_list['mean_i'] = mean_i
    measure_list['mean_j'] = mean_j

    if name in measure_list.keys():
        return measure_list[name]
    else:
        return measure_list


# construct glcm feature images
# @jit(nopython=True)
@jit
def scale_image(image, min_val=0, max_val=255):
    im_max = np.nanmax(image)  # if using np.max, sometimes will return Nan values
    im_min = np.nanmin(image)
    scale_img = min_val + (1 - (im_max - image) / (im_max - im_min)) * max_val

    return scale_img.astype('uint8')


def mask_featured_image(image, feature_img, threshold=40, above=True):
    mask = (feature_img < threshold) * (1 - above * 1) + (feature_img >= threshold) * (above * 1)
    return image * mask


@jit
def get_glcm(win_img, type_name=None, weights=None, stepsize=1):
    if type_name is 'horizontal':
        glcm = direction_glcm(win_img, direction='horizontal', step=stepsize)
    elif type_name is 'vertical':
        glcm = direction_glcm(win_img, direction='vertical', step=stepsize)
    elif type_name is 'diagonal1':
        glcm = direction_glcm(win_img, direction='diagonal1', step=stepsize)
    elif type_name is 'diagonal2':
        glcm = direction_glcm(win_img, direction='diagonal2', step=stepsize)
    else:
        glcm = isotropic_glcm(win_img, weights, stepsize)
    return glcm


@jit
def construct_glcm_feature_img(gray_img, win_order=3, feature='correlation', glcm_type='isotropic', weights=None,
                               fill_type='mirror', norm=True, symm=True, stepsize=1):
    M, N = gray_img.shape
    feature_img = np.zeros([M, N])

    for i in range(M):
        for j in range(N):
            win_img = glcm_window_img(gray_img, neighbor=win_order, current_row=i, current_col=j, fill=fill_type)
            glcm = get_glcm(win_img, glcm_type, weights, stepsize=stepsize)
            feature_img[i, j] = glcm_measures(glcm, name=feature, normed=norm, symmetric=symm)

    return scale_image(feature_img)

@jit
def construct_texture_img(gray_img, win_order=3, feature='contrast', angles=0,
                               fill_type='mirror', norm=True, symm=True, stepsize=1,levels=7):
    # THIS  FUNCTION  is implemented by skimage's functions
    # please first re-scale input image to 0~levels-1 gray-level,
    M, N = gray_img.shape
    feature_img = np.zeros([M, N])

    for i in range(M):
        for j in range(N):
            win_img = glcm_window_img(gray_img, neighbor=win_order, current_row=i, current_col=j, fill=fill_type)

            glcm = greycomatrix(win_img.astype('uint8'), [stepsize], [angles],
                         levels=levels, symmetric=symm, normed=norm)
            feature_img[i, j] = greycoprops(glcm, feature)[0, 0]

    return scale_image(feature_img)


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


@jit
def first_order_features(image, feature_name='entropy'):
    """
    Perform some basic (1-order statistic) texture features for gray-level images.
    :param image: 2-D gray-level image
    :return: info_list, some basic features
    """
    info_list = {'min': 0, 'max': 0, 'variance': 0,
                 'mean': 0, 'std_dev': 0,
                 'skewness': 0, 'kurtosis': 0,
                 'entropy': 0, 'energy': 0,
                 'smoothness': 0, 'coefficient': 0}
    # hist, - = np.histogram(image.flatten(), bins=256, range=[0, 255], density=False)

    info_list['min'] = image.min()
    info_list['max'] = image.max()

    # hist = cv2.calcHist(image,[0],None,[256],[0,256])
    hist, _ = np.histogram(image.flatten(), bins=256, range=[0, 255], density=False)
    hx = hist.ravel() / hist.sum()
    # mean = np.mean(image.flatten())
    x = np.arange(256)
    mean = hx.dot(x)

    info_list['mean'] = mean
    variance = ((x - mean) ** 2).dot(hx)
    std = np.sqrt(variance)
    info_list['variance'] = variance
    info_list['std_dev'] = std
    info_list['skewness'] = ((x - mean) ** 3).dot(hx) / std ** 3  # different with lecture notes
    info_list['kurtosis'] = (((x - mean) ** 4) * hx).sum() / std ** 4 - 3  # different with lecture notes
    info_list['energy'] = (hx * hx).sum()
    info_list['smoothness'] = 1 - 1 / (1 + variance)
    info_list['coefficient'] = float(std) / mean

    # ref: https://stackoverflow.com/questions/16647116/faster-way-to-analyze-each-sub-window-in-an-image
    log_h = np.log2(hx + 0.00001)
    info_list['entropy'] = -1 * (log_h * hx).sum()

    if feature_name in info_list.keys():
        return info_list[feature_name]
    elif feature_name is 'ALL':
        return info_list
    else:
        return info_list['entropy']


@jit
def construct_first_order_feature_image(gray_img, feature='entropy', win_order=3, fill_type='mirror'):
    M, N = gray_img.shape
    feature_img = np.zeros([M, N])

    for i in range(M):
        for j in range(N):
            win_img = glcm_window_img(gray_img, neighbor=win_order, current_row=i, current_col=j, fill=fill_type)
            feature_img[i, j] = float(first_order_features(win_img, feature))

    return scale_image(feature_img)
