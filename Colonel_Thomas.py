# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 17:16:35 2016

@author: ChristopherChee
"""
import numpy as np
from sklearn import svm


def Colonel_Thomas(matrices):
    """
    Method to form the data matrix for SVM
    @params matrices: a list of data matrices, one for each image
    @return data: a matrix of stacked data points
    """
    # determine size of data matrix (spliced matrices)
    # length = 0
    # for matrix in matrices:
    #     length = length + matrix.shape[1]
    # # stack matrices
    # data = np.zeros((6, length))
    # pointer = 0
    # for matrix in matrices:
    #     blocksize = matrix.shape[1]
    #     data[:, pointer:blocksize] += matrix
    #     pointer += blocksize
    data = matrices[0]
    for matrix in matrices[1:]:
        data = np.vstack((data, matrix))
    return data


def Train_Thomas(training_data):
    """
    "Kernel" SVM method for training data.
    @params training_data: a matrix of stacked data points
    @return SVM: kernel SVM object for given data matrices
    """
    kernel = svm.SVC()
    model = kernel.fit(training_data[:, 0:4], training_data[:, 5])
    return model
