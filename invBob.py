import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

def inv_Bob(lis,frow,fcol):

	row = 0
	col = 0
	image = np.zeros(frow,fcol)
	for r in lis.shape[0]:
		image[row][col] = lis[r]
		if(col<fcol):
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

def retrieve_image(labels,n,frow,fcol)
	lis = inv_col(labels,n)
	image = inv_Bob(lis,frow,fcol)
	disp(image)
	return

