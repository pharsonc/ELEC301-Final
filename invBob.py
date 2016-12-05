import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


def inv_Bob(lis,frow,fcol):
	"""

	:param lis: A list of matrices
	:param frow: The number of rows in the final image
	:param fcol: The number of columns in the final image
	:return:
	"""
	row = 0
	col = 0
	image = np.zeros((frow,fcol))
	# for r in lis.shape[0]:
	for r in range(len(lis)):
		#print row
		#print col
		#print fcol, frow
		image[row][col] = lis[r]
		if(col<=fcol-2):
			col = col + 1
		else:
			row = row + 1
			col = 0
	return image


def inv_col(labels,n):
	done = 0
	seperate = []
	for ni in n:
		seperate.append(labels[done:done+ni])
		done = done + ni
	return seperate


def disp(image):
	plt.imshow(image, interpolation='nearest')
	plt.show()
	return


def retrieve_image(labels,n,frow,fcol):
	"""

	:param labels:
	:param n: A list containing the number of pixels in each image
	:param frow: The number of rows in the final image
	:param fcol: The number of columns in the final image
	:return: Displays image
	"""
	lis = inv_col(labels,n)
	im_lis = []
	for l in lis:
		image = inv_Bob(l,frow,fcol)
		im_lis.append(image)
	return im_lis


