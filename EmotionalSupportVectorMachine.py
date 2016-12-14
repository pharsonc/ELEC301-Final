# -*- coding: utf-8 -*-
"""
Created on Thu Dec 01 22:49:48 2016

@author: Joseph
"""
from get_images import *
from bobo import *
from Colonel_Thomas import *
from invBob import *

from scipy.misc import imsave
from sklearn.externals import joblib
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
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
# # TRAINING DATA
# full_list = get_image_list('./annotations/list.txt')
# # full_list = full_list[0:200]
# image_zip = get_images(full_list)
# data_matrix_arr = Bob2(image_zip)
# training_data = Colonel_Thomas(data_matrix_arr)
# #
# # # Saving data matrices to file
# save_matrix(training_data, 'training_data.dat')

# ################### 2. MAKING MODEL  #########################
# training_data_ld = load_matrix('training_data.dat')
# model = Train_Thomas(training_data_ld)
# model = Train_Thomas(training_data)
#
# # Save model
# joblib.dump(model, 'classifier.pkl')

# ################### 3a. CROSS-VALIDATION  ####################
# RUN THIS BY ITSELF

def dice(im1, im2):
	im1 = np.asarray(im1).astype(np.bool)
	im2 = np.asarray(im2).astype(np.bool)

	if im1.shape != im2.shape:
		raise ValueError("Shape mismatch")

	# Compute Dice coefficient
	intersection = np.logical_and(im1, im2)

	return 2. * intersection.sum() / (im1.sum() + im2.sum())


full_list = get_image_list('./annotations/list.txt')
full_list = full_list[0:10]
X, y, dims = get_images2(full_list)

X = np.asarray(X)
y = np.asarray(y)
dims = np.asarray(dims)

# K-fold cross-validation
num_folds = 5
kf = KFold(n_splits=num_folds)
dice_coeffs = []

for train_index, test_index in kf.split(X):
	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = y[train_index], y[test_index]
	dims_train, dims_test = dims[train_index], dims[test_index]

	data_matrix_arr = Bob2(zip(X_train, y_train))
	data_matrix = Colonel_Thomas(data_matrix_arr)

	model = Train_Thomas(data_matrix)

	# Processing test data
	test_data_matrix_arr = Bob2(zip(X_test, X_test))
	test_data_matrix = Colonel_Thomas(test_data_matrix_arr)
	test_data_matrix = test_data_matrix[:, 0:5]

	test_label = model.predict(test_data_matrix)

	ground_truth = retrieve_image(y_test, dims_test)
	reshaped = retrieve_image(test_label, dims_test)

	dice_coeff = 0.0
	for i in range(len(reshaped)):
		seg = reshaped[i]
		gt = ground_truth[i]
		dice_coeff += dice(seg, gt) / len(reshaped)
	dice_coeffs.append(dice_coeff)

print dice_coeffs

# ################### 3b. TESTING MODEL  #######################
# # Load model
# model = joblib.load('classifier.pkl')

# # Get test images
# test_images, test_image_dims = get_test_images2()
#
# test_data_arr = Bob(test_images, test_images)
# test_data_matrix = Colonel_Thomas(test_data_arr)
# test_data_matrix = test_data_matrix[:, 0:5]
# test_label = model.predict(test_data_matrix)
#
# reshaped = retrieve_image(test_label, test_image_dims)
#
# for n in range(len(reshaped)):
# 	filename = 'segmented' + str(n+1) + '.png'
# 	imsave(filename, reshaped[n])
