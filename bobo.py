# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 14:58:02 2016

@author: Joseph
"""

from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import scale
#from sklearn.svm.libsvm import predict
import scipy
import matplotlib
import numpy


def Bob(training, labels):
    """
    Bob the builder builds your data matrix.
    @param training The training images in raw form.
    @return A transformed set of data matrices.
    """
    #bigData = numpy.zeros(training.length)
    bigData = []
    for image, label in zip(training, labels):
        data = numpy.zeros((6, image.shape[1]*image.shape[0]))
        for x in xrange(image.shape[0]):
            for y in xrange(image.shape[1]):
                data[0][image.shape[0]*y + x] += image[x][y]
                if (x != 0):
                    data[1][image.shape[0]*y + x] += image[x-1][y]
                if (x != image.shape[0]):
                    data[2][image.shape[0]*y + x] += image[x+1][y]
                if (y != 0):
                    data[3][image.shape[0]*y + x] += image[x][y-1]
                if (y != image.shape[1]):
                    data[4][image.shape[0]*y + x] += image[x][y+1]
                data[6][image.shape[0]*y + x] += label[x][y]
        bigData.append(data)
    return bigData