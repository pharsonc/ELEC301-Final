import os
import pickle
import numpy as np
from scipy.misc import imread


def get_image_list(path):
    """
    Args:
        path: Path of txt file as string.
    Return:
        A list of file names in from the input txt file (without extension).
    """
    with open(path) as f:
        data = f.readlines()
    # Getting just the file names
    return [line.split()[0] for line in data if not line.startswith('#')]


def get_images(im_list, folder_path):
    """
    Args:
        im_list (list): A list of file names.
        folder_path (int): 0 - /annotations/trimaps/; 1 - /images/
    Returns:
        im_arr (list): A list of 2D numpy arrays representing matrices/images
    """
    if folder_path:
        ext = '.jpg'
        folder = './images/'
    else:
        ext = '.png'
        folder = './annotations/trimaps/'

    im_arr = []
    for name in im_list:
        path = folder + name + ext
        im = imread(path)
        if(len(im.shape) == 2):
            im_arr.append(im)
        # Dealing with 3D matrices
        elif(im.shape[2] == 3):
            new_im = np.zeros((im.shape[0], im.shape[1]))
            for r in range(im.shape[0]):
                for c in range(im.shape[1]):
                    new_im[r][c] = getIfromRGB(im[r][c])
            im_arr.append(new_im)
        else:
            print("weird shape not rgb")
    return im_arr


def get_images2(im_list):
    """
    Args:
        im_list (list): A list of file names.
        folder_path (int): 0 - /annotations/trimaps/; 1 - /images/
    Returns:
        im_arr (list): A list of 2D numpy arrays representing matrices/images
    """
    im_ext = '.jpg'
    im_folder = './images/'

    lbl_ext = '.png'
    lbl_folder = './annotations/trimaps/'

    im_arr = []
    lbl_arr = []
    for name in im_list:
        im_path = im_folder + name + im_ext
        lbl_path = lbl_folder + name + lbl_ext
        im = imread(im_path)
        lbl = imread(lbl_path)
        if(len(im.shape) == 2):
            im_arr.append(im)
            lbl_arr.append(lbl)
        # Dealing with 3D matrices
        elif(im.shape[2] == 3):
            new_im = np.zeros((im.shape[0], im.shape[1]))
            for r in range(im.shape[0]):
                for c in range(im.shape[1]):
                    new_im[r][c] = getIfromRGB(im[r][c])
            im_arr.append(new_im)
            lbl_arr.append(lbl)
        else:
            print("weird shape not rgb")
    return zip(im_arr, lbl_arr)


def getRGBfromI(RGBint):
    blue = RGBint & 255
    green = (RGBint >> 8) & 255
    red = (RGBint >> 16) & 255
    return red, green, blue


def getIfromRGB(rgb):
    red = rgb[0]
    green = rgb[1]
    blue = rgb[2]
    RGBint = (red << 16) + (green << 8) + blue
    return RGBint


# Getting test images
# def get_test_images():
# 	num = range(1,45)
# 	im_list = [str(n) + '.jpg' for n in num]


# Testing: Load first 5 images from list.txt
# images = get_image_list('./annotations/list.txt')
# arr = get_images(images[:5], 1)
