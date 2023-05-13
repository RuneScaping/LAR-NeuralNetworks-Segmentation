import numpy as np
import random
import json
from glob import glob
import os
import progressbar
import argparse
from patch_library import PatchLibrary
import matplotlib.pyplot as plt
from sklearn.feature_extraction.image import extract_patches_2d
from skimage import io, color, img_as_float
from skimage.exposure import adjust_gamma
from keras.models import Model, model_from_json
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Dropout, Input, Reshape
from keras.layers.merge import Concatenate
from keras.optimizers import SGD
from keras import regularizers
from keras.constraints import max_norm
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import np_utils
from os import makedirs
from os.path import isdir
from errno import EEXIST


progress = progressbar.ProgressBar(widgets=[progressbar.Bar('*', '[', ']'), progressbar.Percentage(), ' '])

def mkdir_p(path):
   