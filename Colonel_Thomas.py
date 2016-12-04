# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 17:16:35 2016

@author: ChristopherChee
"""
import numpy as np
from sklearn.svm import SVM


def Colonel_Thomas(matrices):
    """
    Method to form the data matrix for SVM
    @params matrices: a list of data matrices, one for each image
    @return data: a matrix of stacked data points
    """
    # determine size of data matrix (spliced matrices)
    length = 0
    for matrix in matrices:
        length = length + matrix.shape[2]
    # stack matrices
    data = np.zeroes(6, length)
    pointer = 0
    for matrix in matrices:
        blocksize = matrix.shape[2]
        data[:, pointer:blocksize] += matrix
        pointer += blocksize
    return data


def Train_Thomas(training_data):
    """
    "Kernel" SVM method for training data.
    @params training_data: a matrix of stacked data points
    @return SVM: kernel SVM object for given data matrices
    """
    kernel = SVM()
    model = kernel.fit(training_data[0:5], training_data[[6]])
    return model
