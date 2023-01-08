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
        print('******************************************', step)
        step += 1
        model_to_make.add(Conv2D(64, (3, 3),
                                 kernel_initializer=glorot_normal(),
                                 bias_initializer='zeros',
                                 padding='same',
                                 data_format='channels_first',
                                 input_shape=(4, 33, 33)
                                 ))
        print(model_to_make.input_shape)
        print(model_to_make.output)
        print('******************************************', step)
        step += 1
        model_to_make.add(LeakyReLU(alpha=0.333))
        print(model_to_make.output)
        print('******************************************', step)
        step += 1
        model_to_make.add(Conv2D(filters=64,
                                 kernel_size=(3, 3),
                                 padding='same',
                                 data_format='channels_first',
                                 input_shape=(64, 33, 33)))
        print(model_to_make.output)
        print('******************************************', step)
        step += 1
        model_to_make.add(LeakyReLU(alpha=0.333))
        print(model_to_make.output)
        if self.is_hgg:
            model_to_make.add(Conv2D(filters=64,
                                     kernel_size=(3, 3),
                                     padding='same',
                                     data_format='channels_first',
                                     input_shape=(64, 33, 33)))
            print('******************************************', step)
            step += 1
            print(model_to_make.output)

            model_to_make.add(LeakyReLU(alpha=0.333))
            print('******************************************', step)
            step += 1
            print(model_to_make.output)

        model_to_make.add(MaxPool2D(pool_size=(3, 3),
                                    strides=(2, 2),
                                    data_format='channels_first',
                                    input_shape=(64, 33, 33)))
        print('******************************************', step)
        step += 1
        print(model_to_make.output)

        model_to_make.add(Conv2D(filters=128,
                                 kernel_size=(3, 3),
                                 padding='same',
                                 data_format='channels_first',
                                 input_shape=(64, 16, 16)))
        print('******************************************', step)
        step += 1
        print(model_to_make.output)

        model_to_make.add(LeakyReLU(alpha=0.333))
        print('******************************************', step)
        step += 1
        print(model_to_make.output)

        model_to_make.add(Conv2D(filters=128,
                                 kernel_size=(3, 3),
                                 padding='same',
                                 data_format='channels_first',
                                 input_shape=(128, 16, 16)))
        print('******************************************', step)
        step += 1
        print(model_to_make.output)

        if self.is_hgg:
            model_to_make.add(Conv2D(filters=128,
                                     kernel_size=(3, 3),
                                     padding='same',
                                     data_format='channels_first',
                                     input_shape=(128, 16, 16)))
            print('******************************************', step)
            step += 1
            print(model_to_make.output)
            model_to_make.add(LeakyReLU(alpha=0.333))
            print('******************************************', step)
            step += 1
            print(model_to_make.output)
        model_to_make.add(MaxPool2D(pool_size=(3, 3),
                                    strides=(2, 2),
                                    data_format='channels_first',
                                    input_shape=(128, 16, 16)))
        print('******************************************', step)
        step += 1
        print(model_to_make.output)
        print('******************************************', 'flattened')
        model_to_make.add(Flatten())
        print(model_to_make.output)
        model_to_make.add(Dense(units=256, input_dim=6272))
        print('******************************************', step)
        step += 1
        print(model_to_make.output)
        model_to_make.add(LeakyReLU(alpha=0.333))
        print('******************************************', step)
        step += 1
        print(model_to_make.output)
        model_to_make.add(Dropout(dropout_rate))
        print('******************************************', step)
        step += 1
        print(model_to_make.output)
        model_to_make.add(Dense(units=256, input_dim=256))
        print('******************************************', step)
        step += 1
        print(model_to_make.output)
        model_to_make.add(LeakyReLU(alpha=0.333))
        print('******************************************', step)
        step += 1
        print(model_to_make.output)
        model_to_make.add(Dropout(dropout_rate))
        print('******************************************', step)
        step += 1
        print(model_to_make.output)
        model_to_make.add(Dense(units=5,
                                input_dim=256))
        print('******************************************', step)
        step += 1
        print(model_to_make.output)
        model_to_make.add(Activation('softmax'))
        print('******************************************', step)
        step += 1
        print(model_to_make.output)
        self.model = model_to_make

    def _compile_model(self):
        # default decay = 1e-6, lr = 0.01 maybe 1e-2 for linear decay?
        sgd = SGD(lr=3e-3,
                  decay=0,
                  momentum=0.9,
                  nesterov=True)
        print(sgd)
        self.model.compile(optimizer=sgd,
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    @staticmethod
    def load_model_weights(model_name):
        """

        :param model_name: filepath to model and weights, not including extension
        :return: Model with loaded weights. can fit on model using loaded_model=True in fit_model method
        """
        print('Loading model {}'.format(model_name))
        model_to_load = '{}.json'.format(model_name)
        weights = '{}.hdf5'.format(model_name)
        with open(model_to_load) as f:
            m = f.next()
        model_comp = model_from_json(json.loads(m))
        model_comp.load_weights(weights)
        print('Done.')
        return model_comp

    def fit_model(self, X_train, y_train):
        """

        :param X_train: list of patches to train on in form (n_sample, n_channel, h, w)
        :param y_train: list of labels corresponding to X_train patches in form (n_sample,)
        :return: Fits specified model
        """

        print(X_train.shape)
        print('*' * 100)
        print(y_train.shape)
        print('*' * 100)
        Y_train = np_utils.to_categorical(y_train, 5)

        shuffle = zip(X_train, Y_train)
        np.random.shuffle(shuffle)

        X_train = np.array([shuffle[i][0] for i in xrange(len(shuffle))])
        Y_train = np.array([shuffle[i][1] for i in xrange(len(shuffle))])
        EarlyStopping(monitor='val_loss', patience=2, mode='auto')

        if self.is_hgg:
            n_epochs = 20
        else:
            n_epochs = 25

        self.model.fit(X_train, Y_train, epochs=n_epochs, batch_size=128, verbose=1)

    def save_model(self, model_name):
        """
        Saves current model as json and weigts as h5df file
        :param model_name: name to save model and weigths under, including filepath but not extension
        :return:
        """
        model_to_save = '{}.json'.format(model_name)
        weights = '{}.hdf5'.format(model_name)
        json_string = self.model.to_json()
        try:
            self.model.save_weights(weights)
        except:
            mkdir_p(model_name)
            self.model.save_weights(weights)

        with open(model_to_save, 'w') as f:
        