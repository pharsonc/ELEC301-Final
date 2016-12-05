# -*- coding: utf-8 -*-
"""
Created on Thu Dec 01 22:49:48 2016

@author: Joseph
"""
from get_images import *
from bobo import *
from Colonel_Thomas import *

# from sklearn import svm
# from sklearn.model_selection import KFold
# from sklearn.metrics import confusion_matrix
# from sklearn.preprocessing import scale
# from sklearn.svm.libsvm import predict
# import scipy
# import numpy

# Testing
full_list = get_image_list('./annotations/list.txt')
full_list = full_list[:2]
all_images = get_images(full_list, 1)
all_labels = get_images(full_list, 0)
data_matrix_arr = Bob(all_images, all_labels)
training_data = Colonel_Thomas(data_matrix_arr)
model = Train_Thomas(training_data)
print 'done'