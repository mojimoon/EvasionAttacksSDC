import numpy as np

import tensorflow as tf
tf.compat.v1.enable_eager_execution()
import os
import sys

import pickle
import gzip

from os import path
import random
from tensorflow.keras import backend as K
K.set_learning_phase(0)

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


import time

import imageio as im
import skimage.transform as st

from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.python.keras.utils import np_utils
from tensorflow.keras.models import load_model

IMAGE_SIZE = 128
NUM_CHANNELS = 3
NUM_LABELS = 3

class SDC_model_epoch:
    """
    Building the best model arhitecture for Epoch model
    """
    def __init__(self, restore, session = None, image_size_param = IMAGE_SIZE, num_channels_param = NUM_CHANNELS, num_labels_param = NUM_LABELS):

        self.image_size = image_size_param 
        self.num_channels = num_channels_param
        self.num_labels = num_labels_param

        model = Sequential()
        
        model.add(Conv2D(32, (3, 3), input_shape =(image_size_param, image_size_param, num_channels_param)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.4))

        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.4))

        model.add(Conv2D(128, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(3))

        model.load_weights(restore)

        self.model = model
        self.model_path = restore

        self.fn = lambda correct, predicted: tf.nn.softmax_cross_entropy_with_logits_v2(labels = correct, logits = predicted)
        self.sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss=self.fn, optimizer=self.sgd, metrics=['accuracy'], run_eagerly=True)

    def predict(self, data):
        return self.model(data) # return logits
    
    def evaluate(self, data, labels, batch_size = 128):
        return self.model.evaluate(data, labels, batch_size = batch_size)
    
    def prob(self, data):
        logits = self.predict(data)
        return tf.nn.softmax(logits) # return probabilities
    
    def retrain(self, candidateX, candidatey, testX, testy, epochs, batch_size = 128):
        best_acc = 0
        for i in range(epochs):
            self.model.fit(candidateX, candidatey, batch_size = batch_size, epochs = 1)
            loss, acc = self.evaluate(testX, testy)
            # print('Epoch:', i, 'Loss:', loss, 'Accuracy:', acc)
            if acc > best_acc:
                best_acc = acc
                # self.model.save(self.model_path)
        return best_acc


class SDC_model_nvidia:
    """
    Building the best model arhitecture for Nvidia model
    """
    def __init__(self, restore, session = None):

        self.image_size = 128
        self.num_channels = 3 
        self.num_labels = 3

        model = Sequential()
        
        model.add(BatchNormalization(input_shape = (128, 128, 3)))
    
        model.add(Conv2D(24, (5, 5)))
        model.add(Activation('relu'))

        model.add(Conv2D(36, (5, 5)))
        model.add(Activation('relu'))

        model.add(Conv2D(48, (5, 5)))
        model.add(Activation('relu'))

        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))

        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))

        model.add(Flatten())
        model.add(Dense(582))
        model.add(Activation('relu'))

        model.add(Dense(100))
        model.add(Activation('relu'))

        model.add(Dense(50))
        model.add(Activation('relu'))

        model.add(Dense(10))
        model.add(Activation('relu'))

        model.add(Dense(3))

        model.load_weights(restore)

        self.model = model

    def predict(self, data):
        return self.model(data)