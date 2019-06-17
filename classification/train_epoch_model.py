import tensorflow as tf
import numpy as np
import imageio as im
import os
import skimage.transform as st

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import load_model
from os import path

from utilities import *
    
def train(data, file_name, params, num_epochs = 50, batch_size = 128, train_temp = 1, init = None):
    """
    Standard neural network training procedure.
    """
    model = Sequential()
    
    model.add(Conv2D(params[0], (3, 3), input_shape = data.train_data.shape[1:]))
    
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(params[1], (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(params[2], (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(params[3]))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(params[4]))
    
    if init != None:
        model.load_weights(init)

    def fn(correct, predicted):
        return tf.nn.softmax_cross_entropy_with_logits(labels = correct,
                                                       logits = predicted/train_temp)

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    
    model.compile(loss=fn,
                  optimizer=sgd,
                  metrics=['accuracy'])
    
    model.fit(data.train_data, data.train_labels,
              batch_size=batch_size,
              validation_split = 0.1,
              nb_epoch=num_epochs,
              shuffle=True)
    

    if file_name != None:
        model.save(file_name)

    return model
    
if not os.path.isdir('models'):
    os.makedirs('models')

IMAGE_FILE = 'straight_right_left.csv'
IMAGE_FOLDER = '/home/alesia/Documents/sdc/'

train(SDC_data(IMAGE_FILE, IMAGE_FOLDER), "models/sdc_epoch", [32, 64, 128, 1024, 3], num_epochs = 50)

