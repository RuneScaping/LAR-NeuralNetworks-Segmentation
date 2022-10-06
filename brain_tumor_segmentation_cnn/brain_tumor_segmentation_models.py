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
    mkdir -p function, makes folder recursively if required
    :param path:
    :return:
    """
    try:
        makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == EEXIST and isdir(path):
            pass
        else:
            raise


class Brain_tumor_segmentation_model(object):
    """
    A class for compiling/loading, fitting and saving various models,
     viewing segmented images and analyzing results
    """

    def __init__(self, is_hgg=None, n_chan=4, loaded_model=False, model_name=None):
        """

        :param model_name: if loaded_model is True load the model name specified
        :param is_hgg: if True compile model for HGG if False for LGG
        :param n_chan:number of channels being assessed. defaults to 4
        :param loaded_model: True if loading a pre-existing model. defaults to False
        """
        self.n_chan = n_chan
        self.loaded_model = loaded_model
        self.is_hgg = is_hgg
        self.model = None

        if not self.loaded_model:
            if self.is_hgg is None:
                raise NameError('expected boolean True for HGG else False for LGG')
            self.model_name = None
            if is_hgg:
                self.model_name = 'HGG'
            else:
                self.model_name = 'LGG'
            self._make_model()
            self._compile_model()
            print('model for {} ready and compiled, waiting for training'.format(self.model_name))
        else:
            if model_name is None:
                model_to_load = str(raw_input('Which model should I load? '))
            else:
                model_to_load = model_name
            self.model = self.load_model_weights(model_to_load)

    def _make_model(self):
        if self.is_hgg:
            dropout_rate = 0.1
        else:
            dropout_rate = 0.5
        step = 0
        print('******************************************', step)
        step += 1
        model_to_make = Sequential()
        print('***************************