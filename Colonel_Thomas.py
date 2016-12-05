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
    @params matrices: a list of data (numpy) matrices, one for each image
    @return data: a numpy matrix of stacked data points
    """
    data = matrices[0]
    for matrix in matrices[1:]:
        data = np.vstack((data, matrix))
    return data


def Train_Thomas(training_data):
    """
    "Kernel" SVM method for training data.
    @params training_data: a numpy matrix of stacked data points
    @return model: kernel SVM model for given data matrices
    """
    # kernel = svm.SVC()
    kernel = svm.LinearSVC()    # For testing
    model = kernel.fit(training_data[:, 0:5], training_data[:, 5])
    return model
