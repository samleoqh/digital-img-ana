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
"""

import cv2
import os
from imutils import paths
import scipy.misc


def main():
    """
    test the patch_extract function
    :return: none
    """
    # initialize some parameters needed by patch_extract function
    dir = "./images/remotesensing"  # image home folder
    (ph, pw) = (256, 256)  # patch size
    step = 5800  # sliding step

    myDataset = ImageData(dir)
    print(myDataset)

    total_patch = myDataset.patch_extract(patchSize=(ph, pw), stepSize=step)
    print("Total number of patches extracted is : {0}".format(total_patch))


class ImageData:
    def __init__(self, path="./image"):
        self.folder = path
        self.num_files = sum([len(files) for r, d, files in os.walk(self.folder)])

    def __str__(self):
        return "The path to the data set is '%s' which has total %d files" % (self.folder,self.num_files)

    def __setattr__(self, key, value):
        self.__dict__[key] = value

    @classmethod
    def rgb2gray(cls,rgb_img):
        assert len(rgb_img.shape) == 3, 'RGB images only'
        gray_img = rgb_img[:, :, 2] / 255 + \
                   (rgb_img[:, :, 1] / 255) * 2 + \
                   (rgb_img[:, :, 0] / 255) * 4
        return gray_img

    #@classmethod
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
        for _, imgPath in enumerate(paths.list_images(self.folder)):
            path, filename = os.path.split(imgPath)  # filename = imgPath.split(os.path.sep)[-1]
            suffix = os.path.splitext(os.path.basename(filename))[0] + ".png"
            img = cv2.imread(imgPath)

            for x in range(0, img.shape[0], stepSize):
                px = x
                endX = px + ph
                if endX > img.shape[0]:
                    x = endX
                    endX = img.shape[0]
                    px = endX - ph
                else:
                    x += stepSize
                    y = 0

                for y in range(0, img.shape[1], stepSize):
                    py = y
                    endY = py + pw
                    if endY > img.shape[1]:
                        y = endY
                        endY = img.shape[1]
                        py = endY - pw
                    else:
                        y += stepSize

                    patch = img[px:endX, py:endY]

                    # you can change the new filename format as you please
                    newName = "patch_" + str(px) + "_" + str(py) + "_" + suffix

                    # exam if it is a labeled image by its filename containing 'label' str
                    if "label" in filename:
                        # mapping 3-channel RGB values to a gray scalar to form a new grayscale image
                        # newPatch = patch[:, :, 2] / 255 + \
                        #            (patch[:, :, 1] / 255) * 2 + \
                        #            (patch[:, :, 0] / 255) * 4
                        newPatch = ImageData.rgb2gray(patch)

                        scipy.misc.imsave(os.path.join(path, newName),
                                          newPatch)  # don't know how to use cv2 to save as a gray file
                    else:
                        cv2.imwrite(os.path.join(path, newName), patch)

                    total_patch += 1

        return total_patch


if __name__ == "__main__":
    main()


# newPatch = np.uint8(newPatch)
# newRGB = np.unpackbits(newPatch) # uint8 numbers converted to 8-bit [0,0,0,0,0,0,1,0] - 2
# newRGB = np.reshape(newRGB,(ph,pw,8))
# newRGB = newRGB[:,:,5:8]
# newRGB *= 255
