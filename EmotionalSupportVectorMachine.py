# -*- coding: utf-8 -*-
"""
Created on Thu Dec 01 22:49:48 2016

@author: Joseph
"""
from get_images import *
from bobo import *

from sklearn import svm
#from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
#from sklearn.preprocessing import scale
#from sklearn.svm.libsvm import predict
import scipy
import numpy

# change the path to whatever you need it to be
spiral = scipy.io.loadmat(
    'C:/Users/Joseph/Documents/ELEC 301/Homework 11/TwoSpirals.mat')
labels = spiral['y']
data = spiral['X']
train = svm.SVC()
train.fit(data, labels.ravel())
conf = confusion_matrix(train.predict(data), labels)
# kf = KFold(n_splits=5, shuffle=True)
# kf.get_n_splits(data)
# conf = numpy.zeros((2,2))
# conf = numpy.add(conf, (1.0/5.0)*confusion_matrix(train.predict(data), labels))
"""
for train_index, test_index in kf.split(data):
    train = svm.SVC()
    train.fit(data[train_index], labels[train_index].ravel())
    y_pred = train.predict(data[test_index])
    y_test = labels[test_index]
    conf = numpy.add(conf, (1.0/5.0)*confusion_matrix(y_pred, y_test))
"""
print conf
print 'Error: ', (conf[0, 1] + conf[1, 0]) / conf.sum()

# Testing
full_list = get_image_list('/annotations/list.txt')
all_images = get_image(full_list, 1)
all_labels = get_image(full_list, 0)
data_matrix_arr = Bob(all_images, all_labels)
# Call Chris' function to get a nice data matrix
