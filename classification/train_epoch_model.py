import tensorflow as tf
import numpy as np
import imageio as im
import os
import skimage.transform as st

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD
from tensorflow.python.keras.utils import np_utils
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from os import path

from utilities import SDC_data, Dave_data
    
def train(data, file_name, params, num_epochs = 50, batch_size = 256, init = None):
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

    def fn(correct, predicted):
        return tf.nn.softmax_cross_entropy_with_logits_v2(labels = correct,
                                                       logits = predicted)

    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    
    model.compile(loss=fn,
                  optimizer=sgd,
                  metrics=['accuracy'])
    

    model.fit(data.attack_data, data.attack_labels, 
                batch_size=batch_size, 
                epochs=num_epochs, 
                validation_split = 0.2,
                shuffle = True,
                callbacks = [ModelCheckpoint(file_name + "_b256.h5", save_best_only = True),
                                EarlyStopping(patience = 10, restore_best_weights = True)])

    if file_name != None:
        model.save(file_name)

    return model
    
if not os.path.isdir('models'):
    os.makedirs('models')

IMAGE_FILE = '/home/jzhang2297/data/dave_test/driving_dataset/data.txt'
IMAGE_FOLDER = '/home/jzhang2297/data/dave_test/driving_dataset/'

train(Dave_data(IMAGE_FILE, IMAGE_FOLDER), "models/sdc_epoch", [32, 64, 128, 1024, 3], num_epochs = 50)

