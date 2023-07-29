import numpy as np
from keras.models import Model
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Dropout, Input
from keras.layers.merge import Concatenate
from keras.optimizers import SGD
from keras import regularizers
from keras.constraints import max_norm

__author__ = "Matteo Causio"

__license__ = "MIT"
__version__ = "1.0.1"
__maintainer__ = "Matteo Causio"
__status__ = "Production"


class T