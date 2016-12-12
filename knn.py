# import numpy as np
# from scipy.misc import imread
# from sklearn.preprocessing import scale
from get_images import *

full_list = get_image_list('./annotations/list.txt')
image_zip = get_images3(full_list)

# TO-DO: Scaling training images

# Get test images
test_images = get_test_images2()
test_image_dims = get_image_dims(test_images)
# TO-DO: Scale test images


# Run K-NN

# Scale labels to original size