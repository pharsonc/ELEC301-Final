"""
Run this file to make sure the git/image stuff is all good
"""
from scipy.misc import imread
import matplotlib.pyplot as plt


def open_im(path):
    im = imread(path)
    plt.imshow(im)
    plt.show()


# Seeing if stuff works
open_im('./annotations/trimaps/Abyssinian_1.png')
