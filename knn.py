# import numpy as np
# from scipy.misc import imread
# from sklearn.preprocessing import scale
from get_images import *
from sklearn.neighbors import KNeighborsClassifier


def get_pixel(pixel, image_list):
    """
    @params
    pixel (tuple): Pixel to get neighborhood of
    image_list: zipped list of images and ground truth matrices
    """
    colors = []
    labels = []

    for image, label in image_list:
        colors.append(image[pixel[0], pixel[1]])
        labels.append(label[pixel[0], pixel[1]])

    return zip(colors, labels)


def KFlanders(test, k, image_list):
    model = KNeighborsClassifier(k)

    predict = np.zeros([400, 400])
    for x in xrange(400):
        for y in xrange(400):
            pixeldata = get_pixel((x, y), image_list)
            # colors, labels = zip(*pixeldata)
            colors, labels = ([[a] for a, b in pixeldata], [b for a, b in pixeldata])
            model.fit(colors, labels)
            predict[x, y] += model.predict(test[x, y])

    return predict


full_list = get_image_list('./annotations/list.txt')
full_list = full_list[0:6]
image_zip = get_images3(full_list)

# Get test images
test_images, test_image_dims = get_test_images2()


# Run K-NN
labels = []
for i in range(len(test_images)):
    lbl = KFlanders(test_images[i], 3, image_zip)
    # Scale labels to original size
    rows, cols = test_image_dims[i]
    resized_lbl = imresize(lbl, (rows, cols))
    labels.append(resized_lbl)
print labels
