""""
Author:Qinghui Liu
Date:05/09/2017
image patch extracting with sliding windows
labeled images will be automatically converted to gray-scaled image
following below rules.

Building            (RGB: 0, 0, 255)    ---> 1 (b001 -1)
Tree                (RGB: 0, 255, 0)    ---> 2 (b010 -1)
Low vegetation      (RGB: 0, 255, 255)  ---> 3 (b011 -1)
Clutter/background  (RGB: 255, 0, 0)    ---> 4 (b100 -1)
Impervious surfaces (RGB: 255, 255, 255)---> 7 (b111 -3)
Car                 (RGB: 255, 255, 0)  ---> 6 (b110 -1)

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
    step = 200  # sliding step

    total_patch = patch_extract(input_dir=dir, patchSzie=(ph,pw), stepSize=step)
    print("total patches: {0}".format(total_patch))


def patch_extract(input_dir, patchSize=(256,256), stepSize=200):
    """
    :param input_dir: path to your image folder
    :param patchSize: patch size - height and width
    :param stepSize: sliding step, should less than patch size
    :return: total patch numbers extracted successfully
    """
    ph = patchSize[0] # patch height
    pw = patchSize[1] # patch width
    total_patch = 0
    for _, imgPath in enumerate(paths.list_images(input_dir)):
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

                # exam if it is labeled images by its filename containing 'label' str
                if "label" in filename:
                    # mapping 3-channel RGB value to a gray scalar to form a gray image
                    newPatch = patch[:, :, 2] / 255 + \
                               (patch[:, :, 1] / 255) * 2 + \
                               (patch[:, :, 0] / 255) * 4

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
