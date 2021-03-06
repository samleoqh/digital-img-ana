""""
Author:Qinghui Liu
Date:05/09/2017
image patch extracting with sliding windows
labeled images will be automatically converted to gray-scaled image
following below rules.

Building            (RGB: 0, 0, 255)    ---> 1  or 0(b001 -1)
Tree                (RGB: 0, 255, 0)    ---> 2  or 1(b010 -1)
Low vegetation      (RGB: 0, 255, 255)  ---> 3  or 2(b011 -1)
Clutter/background  (RGB: 255, 0, 0)    ---> 4  or 3(b100 -1)
Impervious surfaces (RGB: 255, 255, 255)---> 7  or 4(b111 -3)
Car                 (RGB: 255, 255, 0)  ---> 6  or 5(b110 -1)

there are no '5' - b101 and '0' -b000,
OpenCV stores images in BGR order instead of RGB.
"""

import cv2
import os
from imutils import paths
import numpy as np
import matplotlib.pyplot as plt


def main():
    """
    test the patch_extract function
    :return: none
    """
    # initialize some parameters needed by patch_extract function
    inputdir = "./images/train"  # image home folder
    output = "./output"
    (ph, pw) = (256, 256)  # patch size
    step = 5800  # sliding step

    myDataset = ImageData(inputdir,output)
    print(myDataset)

    total_patch = myDataset.patch_extract(patchSize=(ph, pw), stepSize=step)
    print("Total number of patches extracted is : {0}".format(total_patch))

    # exam the gray patch generated to
    # img = cv2.imread("./output/patch_0_0_top_potsdam_2_10_label.png",cv2.IMREAD_GRAYSCALE)
    # plt.imshow(img, 'gray')
    # plt.show()

class ImageData(object):
    """ Image dataset class providing some basic image pre processing functions """

    def __init__(self, path="./image",outdir="./output"):
        self.folder = path
        self.outputs = outdir
        self.num_files = sum([len(files) for r, d, files in os.walk(self.folder)])

    def __str__(self):
        return "The path to the data set is '%s' which has total %d files" % (self.folder, self.num_files)

    def __setattr__(self, key, value):
        self.__dict__[key] = value

    @classmethod
    def rgb2gray(cls, rgb_img):
        assert len(rgb_img.shape) == 3, 'RGB images only'
        # mapping 3-channel RGB values to a gray scalar to form a new gray scale image
        gray_img = np.zeros((rgb_img.shape[0], rgb_img.shape[1]), dtype=np.uint8)
        gray_img[:] = rgb_img[:, :, 2] / 255 + \
                      (rgb_img[:, :, 1] / 255) * 2 + \
                      (rgb_img[:, :, 0] / 255) * 4

        return gray_img

    @classmethod
    def gray2rgb(cls, gray_img):
        assert len(gray_img.shape) == 2, 'gray images only'
        assert np.max(gray_img) < 8, 'gray level range 0-7 '
        # mapping gray image to 3-channel RGB image
        # only support gray scale range from 0-7.
        size = gray_img.shape
        newRGB = np.uint8(gray_img)
        newRGB = np.unpackbits(newRGB)  # uint8 numbers converted to 8-bit [0,0,0,0,0,0,1,0] - 2
        newRGB = np.reshape(newRGB, (size[0], size[1], 8))
        newRGB = newRGB[:, :, -3:]
        newRGB *= 255
        # if use scipy.misc.imsave() instead of cv2.imwrite() to save file, need change BGR to RGB
        # rgb_img = np.zeros((gray_img.shape[0], gray_img.shape[1], 3), dtype=np.uint8)
        # rgb_img[:, :, 0] = newRGB[:, :, 2]
        # rgb_img[:, :, 1] = newRGB[:, :, 1]
        # rgb_img[:, :, 2] = newRGB[:, :, 0]

        return newRGB

    # @classmethod
    def patch_extract(self, patchSize=(256, 256), stepSize=200):
        """
        :param patchSize: patch size - height and width
        :param stepSize: sliding step, should less than patch size
        :return: total patch numbers extracted successfully
        Note: the generated patch files will be stored as new .png files
         in the same folder with the original file
        """
        ph = patchSize[0]  # patch height
        pw = patchSize[1]  # patch width
        total_patch = 0
        valid_exts = (".jpg", ".jpeg", ".png", ".bmp",".tif")
        for _, imgPath in enumerate(paths.list_files(self.folder, valid_exts)):
            path, filename = os.path.split(imgPath)  # filename = imgPath.split(os.path.sep)[-1]
            suffix = os.path.splitext(os.path.basename(filename))[0] + ".png"
            img = cv2.imread(imgPath)
            #img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)


            for x in range(0, img.shape[0], stepSize):
                px = x
                end_x = px + ph
                if end_x > img.shape[0]:
                    end_x = img.shape[0]
                    px = max(end_x - ph, 0)

                for y in range(0, img.shape[1], stepSize):
                    py = y
                    end_y = py + pw
                    if end_y > img.shape[1]:
                        end_y = img.shape[1]
                        py = max(end_y - pw, 0)

                    patch = img[px:end_x, py:end_y]

                    file_name = "patch_" + str(px) + "_" + str(py) + "_" + suffix

                    if "label" in filename:
                        patch = ImageData.rgb2gray(patch) # convert a RGB image into a gray level image

                    cv2.imwrite(os.path.join(self.outputs, file_name), patch)

                    total_patch += 1

        return total_patch


if __name__ == "__main__":
    main()
