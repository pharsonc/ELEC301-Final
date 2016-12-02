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
		im_list: A list of file names.
		folder_path: 0 - /annotations/trimaps; 1 - /images/
	Returns:
		An array of numpy matrices representing all images
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
		if(im.shape([2]) == 3):
			new_im = np.zeros(im.shape([0]), im.shape([1]))
			for r in im.shape([0]):
				for c in im.shape([1]):
					new_im[r][c] = getIfromRGB(im[r][c])
			im_arr.append(new_im)
		else if(im.shape([2]) == 1):
			im_arr.append(im)
		else:
			print("weird shape not rgb")
	return im_arr


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


# Testing: Load first 5 images from list.txt
# images = get_image_list('./annotations/list.txt')
# arr = get_images(images[:5], 1)
