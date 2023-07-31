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


class TwoBlocksDCNN(object):
    """

    """

    def __init__(self, dropout_rate, learning_rate, momentum_rate, decay_rate, l1_rate, l2_rate):
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.decay_rate = decay_rate
        self.l1_rate = l1_rate
        self.l2_rate = l2_rate
        self.model = self.twoBlocksDCNN()

    # model of TwoPathCNN
    def twoBlocksDCNN(self):
        """


        :param in_channels: int, number of input channel
        :param in_shape: int, dim of the input image
        :return: Model, TwoPathCNN compiled
        """
        input = Input(shape=(65, 65, 4))
        # localPath
        locPath = Conv2D(64, (7, 7), padding='valid', activation='relu', use_bias=True,
                         kernel_regularizer=regularizers.l1_l2(self.l1_rate, self.l2_rate),
                         kernel_constraint=max_norm(2.),
                         bias_constraint=max_norm(2.))(input)
        locPath = MaxPooling2D(pool_size=(4, 4), strides=1, padding='valid')(locPath)
        locPath = Dropout(self.dropout_rate)(locPath)
        locPath = Conv2D(64, (3, 3), padding='valid', activation='relu', use_bias=True,
                         kernel_regularizer=regularizers.l1_l2(self.l1_rate, self.l2_rate),
                         kernel_constraint=max_norm(2.),
                         bias_constraint=max_norm(2.))(locPath)
        locPath = MaxPooling2D(pool_si