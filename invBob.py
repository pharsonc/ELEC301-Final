import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
"""
Functions to process the output of our SVM to obtain images
"""


def inv_Bob(lis, frow, fcol):
    """
    Args:
        lis (list): A list representing a column
        frow (int): The number of rows in the final image
        fcol (int): The number of columns in the final image
    Returns:
        image (np.matrix): A numpy matrix representing an image
    """
    arr = np.array(lis)
    image = arr.reshape((frow, fcol))
    return image


def inv_col(labels, n):
    """
    Args:
        labels (np.ndarray): A 1D array of predicted labels for some number of classified images
        n: An array whose ith element is the size (rows * cols) of the ith image
    Returns:
        separate (list): A list of columns represented as lists
    """
    done = 0
    separate = []
    for ni in n:
        separate.append(labels[done:done + ni])
        done = done + ni
    return separate


def disp(image):
    plt.imshow(image, interpolation='nearest')
    plt.show()
    return


def retrieve_image(labels, im_dims):
    """

    Args:
        labels (np.ndarray): A 1D array of predicted labels for some number of images
        im_dims (list): A list of tuples representing image dimension for each image
    Returns:
        im_lis (list): A list of numpy matrices representing images
    """
    n = [row * col for (row, col) in im_dims]
    lis = inv_col(labels, n)
    im_lis = []
    for ind in xrange(len(im_dims)):
        frow, fcol = im_dims[ind]
        image = inv_Bob(lis[ind], frow, fcol)
        im_lis.append(image)
    return im_lis
