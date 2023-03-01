
"""

The following Convolutional Neural Network it has been implemented
taking inspiration from Ruohui Wang's paper (http://www.springer.com/cda/content/
document/cda_downloaddocument/9783319406626-c2.pdf?SGWID=0-0-45-1575688-p180031493)
The patch extraction is made using the canny filter for edge detection.


"""

from __future__ import print_function
from keras.models import Sequential, model_from_json
from keras.layers import Conv2D, Dense, Flatten, Activation
from keras.initializers import glorot_normal
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.preprocessing import normalize
from skimage.exposure import adjust_sigmoid
from skimage.filters import laplace
from skimage.color import rgb2gray, gray2rgb
from skimage.exposure import adjust_gamma
from skimage.io import imread, imsave
from skimage.feature import canny as canny_filter
from skimage import img_as_float, img_as_ubyte
from glob import glob
from errno import EEXIST
from os import makedirs
from os.path import isdir
import patch_extractor_edges
import numpy as np
import argparse
import json

__author__ = "Cesare Catavitello"

__license__ = "MIT"
__version__ = "1.0.2"
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
        makedirs( path )
    except OSError as exc:  # Python >2.5
        if exc.errno == EEXIST and isdir( path ):
            pass
        else:
            raise


# noinspection PyTypeChecker
class Edge_detector_cnn( object ):
    def __init__(self, loaded_model=False, model_name=None):
        self.loaded_model = loaded_model
        if not self.loaded_model:
            self.model = None
            self._make_model()
            self._compile_model()
        else:
            if model_name is None:
                model_to_load = str( raw_input( 'Which model should I load? ' ) )
            else:
                model_to_load = model_name
            self.model = self.load_model_weights( model_to_load )

    def _make_model(self):
        step = 0
        print( '******************************************', step )
        step += 1
        model_to_make = Sequential()
        print( '******************************************', step )
        step += 1
        model_to_make.add( Conv2D( 32, (5, 5),