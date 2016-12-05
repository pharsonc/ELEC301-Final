# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 14:58:02 2016

@author: Joseph
"""

from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import scale
# from sklearn.svm.libsvm import predict
import scipy
import matplotlib
import numpy as np


def Bob(training, labels):
    """
    Bob the builder builds your data matrix.
    @param training The training images in raw form.
    @return A transformed set of data matrices.
    """
    # bigData = numpy.zeros(training.length)
    bigData = []
    for image, label in zip(training, labels):
        nrow = image.shape[0]
        ncol = image.shape[1]
        # data = np.zeros((6, ncol * nrow))
        # for y in xrange(nrow):
        #     for x in xrange(ncol):
        #         # Pixel
        #         data[0][ncol * y + x] += image[x][y]
        #         # Top neighbor
        #         if (x != 0):
        #             data[1][ncol * y + x] += image[x - 1][y]
        #         # Bottom neighbor
        #         if (x != ncol - 1):
        #             data[2][ncol * y + x] += image[x + 1][y]
        #         # Left neighbor
        #         if (y != 0):
        #             data[3][ncol * y + x] += image[x][y - 1]
        #         # Right neighbor
        #         if (y != nrow - 1):
        #             data[4][ncol * y + x] += image[x][y + 1]
        #         data[5][ncol * y + x] += label[x][y]
        # bigData.append(np.transpose(data))
    data = np.zeros((ncol * nrow, 6))
    for row in xrange(nrow):
        for col in xrange(ncol):
            # Pixel
            data[ncol * row + col][0] += image[row][col]
            # Left neighbor
            if (col != 0):
                data[ncol * row + col][1] += image[row][col - 1]
            # Right neighbor
            if (col != ncol - 1):
                data[ncol * row + col][2] += image[row][col + 1]
            # Top neighbor
            if (row != 0):
                data[ncol * row + col][3] += image[row - 1][col]
            # Bottom neighbor
            if (row != nrow - 1):
                data[ncol * row + col][4] += image[row + 1][col]
            # Label
            data[ncol * row + col][5] += label[row][col]
    return bigData
