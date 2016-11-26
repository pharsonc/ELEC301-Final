import os
import pickle
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
    	folder_path: 0 - /annotations/trimaps; 1 - /images
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
		im_arr.append(im)
	return im_arr

# Testing: Load first 5 images from list.txt
# images = get_image_list('./annotations/list.txt')
# arr = get_images(images[:5], 1)



