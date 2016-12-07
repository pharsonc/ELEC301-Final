# -*- coding: utf-8 -*-
"""
Created on Thu Dec 01 22:49:48 2016

@author: Joseph
"""
from get_images import *
from bobo import *
from Colonel_Thomas import *
from invBob import *

# from sklearn import svm
# from sklearn.model_selection import KFold
# from sklearn.metrics import confusion_matrix
# from sklearn.preprocessing import scale
# from sklearn.svm.libsvm import predict
# import scipy
# import numpy

########################### Testing ###########################
"""
	Running the code the first time will form the data matrix
	and save it to 'training_data.dat'.

	To run the SVM, comment out everything in step 1,
	uncomment step 2, and run.
"""
#################### 1. FORMING DATA MATRIX ###################
full_list = get_image_list('./annotations/list.txt')
# full_list = full_list[:2]
all_images = get_images(full_list, 1)
all_labels = get_images(full_list, 0)
data_matrix_arr = Bob(all_images, all_labels)
training_data = Colonel_Thomas(data_matrix_arr)

# Saving data matrix to file
save_matrix(training_data, 'training_data.dat')

#################### 2. RUNNING SVM ###########################
# test = load_matrix('training_data.dat')
# model = Train_Thomas(training_data)

# # Very hard-coded test
# # Predicting on first image in training data
# test_image = all_images[0]
# rows = test_image.shape[0]
# cols = test_image.shape[1]
# test_data_matrix = data_matrix_arr[0][:, 0:5]
# test_label = model.predict(test_data_matrix)
# # Reshaping etc
# # rows = 500
# # cols = 394
# # test_label = 2*np.ones((1, rows*cols))
# reshaped = retrieve_image(test_label, [rows * cols], rows, cols)
# disp(reshaped[0])
