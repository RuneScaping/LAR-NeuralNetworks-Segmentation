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



class BrainSegDCNN(object):
    """

    """
    def __init__(self):
        self.dropout_rate = None
        self.learning_rate = None
        self.momentum_rate = None
        self.decay_rate = None
        self.l1_rate = None
        self.l2_rate = None
        self.batch_size = None
        self.nb_epoch = None
        self.nb_sample = None
        self.model = None
        self.subpatches_33 = None

    def __init__(self, dropout_rate, learning_rate, momentum_rate, decay_rate, l1_rate, l2_rate, batch_size, nb_epoch,
                 nb_sample, cascade_model=False):
        """
        The field cnn1 is initialized inside the method compile_model
        :param dropout_rate: rate for the dropout layer
        :param learning_rate: learning rate for training
        :param momentum_rate: rate for momentum technique
        :param decay_rate: learning rate decay over each update
        :param l1_rate: rate for l1 regularization
        :param l2_rate: rate for l2 regolarization
        :param batch_size:
        :param nb_epoch: number of epochs
        :param cascade_model: True if the model is input cascade
        """
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.decay_rate = decay_rate
        self.l1_rate = l1_rate
        self.l2_rate = l2_rate
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        self.nb_sample = nb_sample
        self.cascade_model = cascade_model
        self.model = self.compile_model()

    # model of TwoPathCNN
    def one_block_model(self, input_tensor):
        """
        Method to model one cnn. It doesn't compile the model.
        :param input_tensor: tensor, to feed the two path
        :return: output: tensor, the output of the cnn
        """

        # localPath
        loc_path = Conv2D(64, (7, 7), data_format='channels_first', padding='valid', activation='relu', use_bias=True,
                         kernel_regularizer=regularizers.l1_l2(self.l1_rate, self.l2_rate),
                         kernel_constraint=max_norm(2.),
                         bias_constraint=max_norm(2.), kernel_initializer='lecun_uniform', bias_initializer='zeros')(input_tensor)
        loc_path = MaxPooling2D(pool_size=(4, 4), data_format='channels_first', strides=1, padding='valid')(loc_path)
        loc_path = Dropout(self.dropout_rate)(loc_path)
        loc_path = Conv2D(64, (3, 3), data_format='channels_first', padding='valid', activation='relu', use_bias=True,
                          kernel_initializer='lecun_uniform', bias_initializer='zeros',
                          kernel_regularizer=regularizers.l1_l2(self.l1_rate, self.l2_rate),kernel_constraint=max_norm(2.),
                          bias_constraint=max_norm(2.))(loc_path)
        loc_path = MaxPooling2D(pool_size=(2, 2), data_format='channels_first', strides=1, padding='valid')(loc_path)
        loc_path = Dropout(self.dropout_rate)(loc_path)
        # globalPath
        glob_path = Conv2D(160, (13, 13), data_format='channels_first', strides=1, padding='valid', activation='relu', use_bias=True,
                           kernel_initializer='lecun_uniform', bias_initializer='zeros',
                           kernel_regularizer=regularizers.l1_l2(self.l1_rate, self.l2_rate),
                           kernel_constraint=max_norm(2.),
                           bias_constraint=max_norm(2.))(input_tensor)
        glob_path = Dropout(self.dropout_rate)(glob_path)
        # concatenation of the two path
        path = Concatenate(axis=1)([loc_path, glob_path])
        # output layer
        output = Conv2D(5, (21, 21), data_format='channels_first', strides=1, padding='valid', activation='softmax', use_bias=True,
                        kernel_initializer='lecun_uniform', bias_initializer='zeros')(path)
        return output

    def compile_model(self):
        """
        Model and compile the first CNN and the whole two blocks DCNN.
        Also initialize the field cnn1
        :return: Model, Two blocks DeepCNN compiled
        """
        if self.cascade_model:
            # input layers
            input65 = Input(shape=(4, 65, 65))
            input33 = Input(shape=(4, 33, 33))
            # first CNN modeling
            output_cnn1 = self.one_block_model(input65)
            # first cnn compiling
            cnn1 = Model(inputs=input65, outputs=output_cnn1)
            sgd = SGD(lr=self.learning_rate, momentum=self.momentum_rate, decay=self.decay_rate, nesterov=False)
            cnn1.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
            # initialize the field cnn1
            self.cnn1 = cnn1
            print 'First CNN compiled!'
            # concatenation of the output of the first CNN and the input of shape 33x33
            conc_input = Concatenate(axis=1)([input33, output_cnn1])
            # second cnn modeling
            output_dcnn = self.one_block_model(conc_input)
            output_dcnn = Reshape((5,))(output_dcnn)
            # whole dcnn compiling
            dcnn = Model(inputs=[input65, input33], outputs=output_dcnn)
            sgd = SGD(lr=self.learning_rate, momentum=self.momentum_rate, decay=self.decay_rate, nesterov=False)
            dcnn.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
            print 'Cascade DCNN compiled!'
            return dcnn
        else:
            # input layers
            input33 = Input(shape=(4, 33, 33))
            # first CNN modeling
            output_cnn1 = self.one_block_model(input33)
            # first cnn compiling
            cnn1 = Model(inputs=input33, outputs=output_cnn1)
            sgd = SGD(lr=self.learning_rate, momentum=self.momentum_rate, decay=self.decay_rate, nesterov=False)
            cnn1.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
            # initialize the field cnn1
            self.cnn1 = cnn1
            print 'Two pathway CNN compiled!'
            return cnn1

    def fit_model(self, x33_train, y_train, x33_uniftrain, y_uniftrain, x65_train=None, x65_uniftrain=None):
        '''
        Fit the model in both modality single or cascade. For cascade model need either 65x65 and 33x33 patches
        :param x33_train:33x33 patches
        :param x65_train:65x65 patches
        :param y_train: labels
        :param x33_uniftrain:33x33 uniformly distribuited patches
        :param x65_uniftrain:65x65 uniformly distribuited patches
        :param y_uniftrain:uniformly distribuited labels
        '''
        if self.cascade_model:
            if x65_train == None and x65_uniftrain == None:
                print 'Error: patches 65x65, necessary to fit cascade model, not inserted.'
            X33_train, X65_train, Y_train, X33_uniftrain, X65_uniftrain, Y_uniftrain = self.init_cascade_training(x33_train,
                                    x65_train, y_train, x33_uniftrain, x65_uniftrain, y_uniftrain)
            # Stop the training if the monitor function doesn't change after patience epochs
            earlystopping = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')
            # Save model after each epoch to check/bm_epoch#-val_loss
            checkpointer = ModelCheckpoint(filepath="/home/ixb3/Scrivania/check/bm_{epoch:02d}-{val_loss:.2f}.hdf5", verbose=1)
            # Fit the first cnn
            self.fit_cnn1(X33_train, Y_train, X33_uniftrain, Y_uniftrain)
            # Fix all the weights of the first cnn
            self.cnn1 = self.freeze_model(self.cnn1)

            # First-phase training of the second cnn
            self.model.fit(x=[X65_uniftrain, X33_uniftrain], y=Y_uniftrain, batch_size=self.batch_size, epochs=self.nb_epoch,
                       callbacks=[earlystopping, checkpointer], validation_split=0.3, verbose=1)
            # fix all the layers of the dcnn except the output layer for the second-phase
            self.freeze_model(self.model, freeze_output=False)
            # Second-phase training of the second cnn
            self.model.f