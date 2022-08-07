"""

The following code try to implement the CNN model described in
(S. Pereira et al.)  ( http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7426413&isnumber=7463083)
It consist mainly of a class for compiling/loading, fitting and saving two kind of models,
and saving the processed group of images used for test.
The segmentation is described in accordance to the rules of the Brats Contest.

"""

from __future__ import print_function
from skimage.color import rgb2gray
from glob import glob
from skimage import io, color, img_as_float
from skimage.exposure import adjust_gamma
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.metrics import classification_report
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Activation, Flatten
from keras.optimizers import SGD
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping
from keras.initializers import glorot_normal
from keras.models import model_from_json
from os.path import isdir
from os import makedirs
from errno import EEXIST
import numpy as np
import json
import argparse
import matplotlib.image as mpimg
from patch_library import PatchLibrary

__author__ = "Cesare Catavitello"

__license__ = "MIT"
__version__ = "1.0.1"
__maintainer__ = "Cesare Catavitello"
__email__ = "cesarec88@gmail.com"
__status__ = "Production"


def mkdir_p(path):
    """
    mkdir -p function, makes folder recur