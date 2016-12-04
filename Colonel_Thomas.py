import numpy as np
from sklearn.svm import SVM
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 17:16:35 2016

@author: ChristopherChee
"""

"""
"Kernel" SVM method for training data.
@params matrices: a set of data matrices, one for each image
@return SVM: kernel SVM object for given data matrices
"""


def Colonel_Thomas(matrices):
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
    kernel = SVM()
    model = kernel.fit(training_data[0:5], training_data[[6]])
    return model
