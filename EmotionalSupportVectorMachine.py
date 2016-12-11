# -*- coding: utf-8 -*-
"""
Created on Thu Dec 01 22:49:48 2016

@author: Joseph
"""
from get_images import *
from bobo import *
from Colonel_Thomas import *
from invBob import *

from sklearn.externals import joblib
# from sklearn import svm
# from sklearn.model_selection import KFold
# from sklearn.metrics import confusion_matrix
# from sklearn.preprocessing import scale
# from sklearn.svm.libsvm import predict
# import scipy
# import numpy

########################### Testing ###########################
"""
	Initial run: The training data matrix is computed and
	stored in 'training_data.dat' - unless we change what our
	data points are, this won't change -> section 1 can be
	commented out.

	Uncomment section 2 and run: the trained model is saved
	into 'classifier.pkl'. Need to re-run this section if we
	change our SVM stuff.

	Comment out section 2, run section 3: Predicts labels.
"""
################## 1. FORMING DATA MATRICES ###################
# TRAINING DATA
full_list = get_image_list('./annotations/list.txt')
image_zip = get_images2(full_list)
data_matrix_arr = Bob2(image_zip)
training_data = Colonel_Thomas(data_matrix_arr)

# Saving data matrices to file
save_matrix(training_data, 'training_data.dat')

#################### 2. MAKING MODEL  #########################
# training_data_ld = load_matrix('training_data.dat')
# model = Train_Thomas(training_data_ld)
#
# # Save model
# joblib.dump(model, 'classifier.pkl')

#################### 3. TESTING MODEL  ########################
# # Load model
# model = joblib.load('classifier.pkl')

# # Very hard-coded test
# # Predicting on first image in training data
# test_image = get_images(['Abyssinian_100'], 1)
# rows = test_image[0].shape[0]
# cols = test_image[0].shape[1]

# # Just want the data matrix (ignore labels column)
# test_data_matrix = Bob(test_image, test_image)
# test_data_matrix = test_data_matrix[0][:, 0:5]
# test_label = model.predict(test_data_matrix)

# reshaped = retrieve_image(test_label, [rows * cols], rows, cols)
# disp(reshaped[0])

# for n in range(len(reshaped)):
# 	im = Image.fromarray(reshaped[n])
# 	filename = 'segmented', n, '.png'
# 	im.save(filename)
