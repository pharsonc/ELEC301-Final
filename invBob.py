import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
"""
Functions to process the output of our SVM to obtain images
"""


def inv_Bob(lis, frow, fcol):
    """

    :param lis: A list representing a column
    :param frow: The number of rows in the final image
    :param fcol: The number of columns in the final image
    :return: A numpy matrix representing an image
    """
    row = 0
    col = 0
    image = np.zeros((frow, fcol))
    # for r in lis.shape[0]:
    for r in range(len(lis)):		# Number of elements in column
        # print row
        # print col
        # print fcol, frow
        image[row][col] = lis[r]
        if(col <= fcol - 2):
            col = col + 1
        else:
            row = row + 1
            col = 0
    return image


def inv_col(labels, n):
    """

    :param labels: An array of predicted labels for some number of classified images
    :param n: An array whose ith element is the size (rows * cols) of the ith image
    :return: A list of columns represented as lists
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


def retrieve_image(labels, n, frow, fcol):
    """

    :param labels: An array of predicted labels for some number of classified images
    :param n: An array whose ith element is the size (rows * cols) of the ith image
    :param frow: The number of rows in the final image
    :param fcol: The number of columns in the final image
    :return: Displays image
    """
    lis = inv_col(labels, n)
    im_lis = []
    for l in lis:
        image = inv_Bob(l, frow, fcol)
        im_lis.append(image)
    return im_lis
