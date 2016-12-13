from __future__ import print_function

import sys
import os
import time

import numpy as np
from scipy.misc import imread, imresize, imsave
import theano
import theano.tensor as T

import lasagne


def get_image_list(path):
    """
    Args:
        path: Path of txt file as string.
    Return:
        A list of file names in from the input txt file (without extension).
    """
    with open(path) as f:
        data = f.readlines()
    # Getting just the file names
    return [line.split()[0] for line in data if not line.startswith('#')]


def get_images(im_list):
    """
    Args:
        im_list (list): A list of file names.
    Returns:
        (list): A zipped list of 2D numpy arrays representing matrices/images
    """
    im_ext = '.jpg'
    im_folder = './images/'

    lbl_ext = '.png'
    lbl_folder = './annotations/trimaps/'

    # n = len(im_list)
    # im_arr = np.zeros((n, 1, 30, 30))
    # lbl_arr = np.zeros((n,))
    im_arr = []
    lbl_arr = []
    for name in im_list:
    # for i in range(n):
    #     name = im_list[i]
        im_path = im_folder + name + im_ext
        lbl_path = lbl_folder + name + lbl_ext

        im = imread(im_path)
        lbl = imread(lbl_path)
        # Resizing
        im = imresize(im, (30, 30))
        lbl = imresize(lbl, (30, 30))
        lbl = np.remainder(lbl, 2)
        if(len(im.shape) == 2):
            im_arr.append(im)
            lbl_arr.append(lbl)
        # Dealing with 3D matrices
        elif(im.shape[2] == 3):
            new_im = np.zeros((im.shape[0], im.shape[1]))
            for r in xrange(im.shape[0]):
                for c in xrange(im.shape[1]):
                    new_im[r][c] = getIfromRGB(im[r][c])
            # new_im = feature.canny(new_im, sigma=3) #canny edge detection
            im_arr.append(new_im)
            lbl_arr.append(lbl)
        else:
            print("weird shape not rgb")
    return im_arr, lbl_arr


def getIfromRGB(rgb):
    red = rgb[0]
    green = rgb[1]
    blue = rgb[2]
    RGBint = (red << 16) + (green << 8) + blue
    return RGBint


def get_test_images2():
    """
    Returns:
        im_arr (list): A list of 2D numpy arrays representing test images
        dims (list): A list of tuples (# rows, # cols)
    """
    num = range(1, 45)
    names = ['./test_images/' + str(n) + '.jpg' for n in num]
    im_arr = []
    dims = []
    for name in names:
        im = imread(name)
        dims.append(im.shape[0:2])
        im = imresize(im, (30, 30))
        if(len(im.shape) == 2):
            im_arr.append(im)
        # Dealing with 3D matrices
        elif(im.shape[2] == 3):
            new_im = np.zeros((im.shape[0], im.shape[1]))
            for r in xrange(im.shape[0]):
                for c in xrange(im.shape[1]):
                    new_im[r][c] = getIfromRGB(im[r][c])
            im_arr.append(new_im)
        else:
            print("weird shape not rgb")
    return im_arr, dims


def get_neighborhood(image):
    ncol, nrow = (30, 30)
    nbr_list = []
    for row in xrange(ncol):
        for col in xrange(nrow):
            data = np.zeros((3, 3))
            # Pixel
            data[1, 1] = image[row, col]
            # Top left
            if (col != 0) and (row != 0):
                data[0, 0] = image[row - 1, col - 1]
            # Top mid
            if (row != 0):
                data[0, 1] = image[row - 1, col]
            # Top right
            if (col != ncol - 1) and (row != 0):
                data[0, 2] = image[row - 1, col + 1]
            # Left
            if (col != 0):
                data[1, 0] = image[row, col - 1]
            # Right
            if (col != ncol - 1):
                data[1, 2] = image[row, col + 1]
            # Bottom left
            if (col != 0) and (row != nrow - 1):
                data[2, 0] = image[row + 1, col - 1]
            # Bottom mid
            if (row != nrow - 1):
                data[2, 1] = image[row + 1, col]
            # Bottom right
            if (col != ncol - 1) and (row != nrow - 1):
                data[2, 2] = image[row + 1, col + 1]
            nbr_list.append(data)
    return nbr_list


def get_training(images, labels):
    n = len(images)
    y_train = np.zeros((n*900,), dtype=np.int32)
    big_data = []
    for j in xrange(n):
        # Get 3x3 neighborhood of each pixel
        image = images[j]
        # data = np.zeros((ncol * nrow, 6))
        image_nbrs = get_neighborhood(image)
        big_data.extend(image_nbrs)

        # Flattening labels
        y_train[j*900:(j+1)*900] = labels[j].reshape((900,))

    # X_len = len(big_data)
    # X_train = np.zeros((X_len, 1, 30, 30))
    # for i in xrange(X_len):
    X_train = np.array(big_data)
    X_train = X_train.reshape((-1, 1, 3, 3))

    return X_train, y_train


def get_test(images):
    big_data = []
    for image in images:
        image_nbrs = get_neighborhood(image)
        big_data.extend(image_nbrs)

    X_test = np.array(big_data)
    X_test = X_test.reshape((-1, 1, 3, 3))

    return X_test



# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.
# Notice that this function returns only mini-batches of size `batchsize`.
# If the size of the data is not a multiple of `batchsize`, it will not
# return the last (remaining) mini-batch.

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


# ##################### Build the neural network model #######################

def build_mlp(input_var=None):
    # This creates an MLP of two hidden layers of 800 units each, followed by
    # a softmax output layer of 10 units. It applies 20% dropout to the input
    # data and 50% dropout to the hidden layers.

    # Input layer, specifying the expected input shape of the network
    # (unspecified batchsize, 1 channel, 3 rows and 3 columns) and
    # linking it to the given Theano variable `input_var`, if any:
    l_in = lasagne.layers.InputLayer(shape=(None, 1, 3, 3),
                                     input_var=input_var)

    # Apply 20% dropout to the input data:
    l_in_drop = lasagne.layers.DropoutLayer(l_in, p=0.2)

    # Add a fully-connected layer of 800 units, using the linear rectifier, and
    # initializing weights with Glorot's scheme (which is the default anyway):
    l_hid1 = lasagne.layers.DenseLayer(
            l_in_drop, num_units=800,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

    # We'll now add dropout of 50%:
    l_hid1_drop = lasagne.layers.DropoutLayer(l_hid1, p=0.5)

    # Another 800-unit layer:
    l_hid2 = lasagne.layers.DenseLayer(
            l_hid1_drop, num_units=800,
            nonlinearity=lasagne.nonlinearities.rectify)

    # 50% dropout again:
    l_hid2_drop = lasagne.layers.DropoutLayer(l_hid2, p=0.5)

    # Finally, we'll add the fully-connected output layer, of 10 softmax units:
    l_out = lasagne.layers.DenseLayer(
            l_hid2_drop, num_units=800,
            nonlinearity=lasagne.nonlinearities.softmax)

    # Each layer is linked to its incoming layer(s), so we only need to pass
    # the output layer to give access to a network in Lasagne:
    return l_out


def main(model='mlp', num_epochs=500):
    # Getting training and test data
    full_list = get_image_list('./annotations/list.txt')
    full_list = full_list[0:10]
    images, labels = get_images(full_list)
    X_train, y_train = get_training(images, labels)

    test_images, test_image_dims = get_test_images2()
    X_test = get_test(test_images)

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    print("Building model and compiling functions...")
    network = build_mlp(input_var)

    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.01, momentum=0.9)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, 500, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1

        # # And a full pass over the validation data:
        # val_err = 0
        # val_acc = 0
        # val_batches = 0
        # for batch in iterate_minibatches(X_val, y_val, 500, shuffle=False):
        #     inputs, targets = batch
        #     err, acc = val_fn(inputs, targets)
        #     val_err += err
        #     val_acc += acc
        #     val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        # print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        # print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        # print("  validation accuracy:\t\t{:.2f} %".format(
        #     val_acc / val_batches * 100))

    # # After training, we compute and print the test error:
    # test_err = 0
    # test_acc = 0
    # test_batches = 0
    # for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
    #     inputs, targets = batch
    #     err, acc = val_fn(inputs, targets)
    #     test_err += err
    #     test_acc += acc
    #     test_batches += 1
    # print("Final results:")
    # print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    # print("  test accuracy:\t\t{:.2f} %".format(
    #     test_acc / test_batches * 100))

    # Prediction
    predictions = []
    for im in range(len(test_image_dims)):  # Iterating over 44 test images
        net_output = lasagne.layers.get_output(network, X_test[im*900:(im+1)*900])
        reshaped = net_output.eval().reshape((30, 30))
        predicted = imresize(reshaped, test_image_dims[im])
        predictions.append(predicted)

    for n in range(len(predictions)):
        filename = 'predictions' + str(n + 1) + '.png'
        imsave(filename, predictions[n])


main()
test = 1 + 1