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
        data = np.zeros((6, ncol * nrow))
        for x in xrange(nrow):
            for y in xrange(ncol):
                data[0][nrow * y + x] += image[x][y]
                if (x != 0):
                    data[1][nrow * y + x] += image[x - 1][y]
                if (x != nrow - 1):
                    data[2][nrow * y + x] += image[x + 1][y]
                if (y != 0):
                    data[3][nrow * y + x] += image[x][y - 1]
                if (y != ncol - 1):
                    data[4][nrow * y + x] += image[x][y + 1]
                data[5][nrow * y + x] += label[x][y]
        bigData.append(np.transpose(data))
    return bigData
