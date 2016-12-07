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
    @param training (list): A list of 2D numpy arrays representing matrices/training images.
    @return bigData (list): A transformed list of data (numpy) matrices.
    """
    # bigData = numpy.zeros(training.length)
    bigData = []
    for image, label in zip(training, labels):
        nrow = image.shape[0]
        ncol = image.shape[1]
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
        bigData.append(data)
    return bigData
