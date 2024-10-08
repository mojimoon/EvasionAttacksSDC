import tensorflow as tf
import numpy as np
import imageio as im
import os
import skimage.transform as st

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.python.keras.utils import np_utils
from tensorflow.keras.models import load_model
from os import path

from utilities import SDC_data

    
def train(data, file_name, params, num_epochs = 50, batch_size = 128, train_temp = 1, init = None):
    """
    Standard neural network training procedure.
    """
    model = Sequential()
    
    model.add(Conv2D(params[0], (3, 3), input_shape = data.attack_data.shape[1:]))
    
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

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'mean_squared_error')

    model.fit(data.attack_data, data.attack_labels, batch_size = batch_size, epochs = num_epochs)
    
    if file_name != None:
        model.save(file_name)

    return model
    
if not os.path.isdir('models'):
    os.makedirs('models')

IMAGE_FILE = '/home/alesia/Documents/sdc/interpolated.csv'
IMAGE_FOLDER = '/home/alesia/Documents/sdc/'


train(SDC_data(IMAGE_FILE, IMAGE_FOLDER), "models/sdc", [32, 64, 128, 1024, 1], num_epochs = 50)

