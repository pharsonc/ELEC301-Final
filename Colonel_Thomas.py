# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 17:16:35 2016

@author: ChristopherChee
"""
import numpy as np
from sklearn import svm


# def Colonel_Thomas(matrices):
#     """
#     Method to form the data matrix for SVM
#     @params matrices (list): a list of data (numpy) matrices, one for each image
#     @return data (ndarray): a 2D numpy array representing matrix of stacked data points
#     """
#     data = matrices[0]
#     for matrix in matrices[1:]:
#         data = np.vstack((data, matrix))
#     return data


def Colonel_Thomas(matrices):
    """
    Method to form the data matrix for SVM
    @params matrices (list): a list of data (numpy) matrices, one for each image
    @return data (ndarray): a 2D numpy array representing matrix of stacked data points
    """
    lengths = [mat.shape[0] for mat in matrices]
    data = np.zeros((sum(lengths), 6))
    ptr = 0
    for i in xrange(len(lengths)):
        print i
        data[ptr:ptr+lengths[i]] = matrices[i]
        ptr += lengths[i]
    return data


def save_matrix(matrix, file):
    """
    Saves the matrix to a text file.

    Args:
        matrix (ndarray): 2D numpy array (matrix) to be saved
        file (str): File path to save the matrix to
    Returns:
        None
    """
    matrix.dump(file)
    return


def load_matrix(file):
    """
    Loads a matrix from the input file.

    Args:
        file (str): Path to text file containing the matrix
    Returns:
        matrix (ndarray): 2D numpy array (matrix) read from file
    """
    matrix = np.load(file)
    return matrix


def Train_Thomas(training_data):
    """
    "Kernel" SVM method for training data.
    @params training_data (ndarray): a 2D numpy array of stacked data points
    @return model: kernel SVM model for given data matrix
    """
    # kernel = svm.SVC()
    kernel = svm.LinearSVC()    # For testing
    model = kernel.fit(training_data[:, 0:5], training_data[:, 5])
    return model
