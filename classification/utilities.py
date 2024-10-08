import numpy as np

import tensorflow as tf
import os
import sys

import pickle
import gzip

from os import path
import random

# for Python 3.6
# from keras import backend as K
# from keras.optimizers import SGD
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Activation, Flatten
# from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
# from keras.utils import np_utils
# from keras.models import load_model

# for Python 2.7
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.python.keras.utils import np_utils
from tensorflow.keras.models import load_model
K.set_learning_phase(0)

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

import time
import math
from tqdm import tqdm

import imageio as im
import skimage.transform as st

from sklearn.model_selection import train_test_split

def read_images_steering_directions(steering_image_log, image_folder):
    """
        Read the images with corresponding direction
        :param steering_image_log: the file that contains the path to image and all necessary information about it including steering angle and path to image
        :param image_folder: the path to the folder that contains all images
        :return: imgs, steerings: images, labels
    """

    STEERING_ANGEL_THRESHOLD = 0.15
    NUM_CLASSES = 3
    
    steerings = []
    imgs = []
    
    with open(steering_image_log) as f:
        for line in f.readlines()[1:]:
            
            fields = line.split(",")
            
            if 'center' not in line:
                continue
            
            #getting the value of steering angle
            steering = fields[6]
            steering_label = np.zeros(NUM_CLASSES)
            #checking if the direction is 'right'
            if float(steering) > STEERING_ANGEL_THRESHOLD:
                steering_label[2] = 1

            elif float(steering) < -1 * STEERING_ANGEL_THRESHOLD:
                steering_label[0] = 1

            else:
                steering_label[1] = 1

            #path to the image
            url = fields[5]
            #reading the image
            img = im.imread(path.join(image_folder, url))
            
            #image preprocessing
            crop_img = img[200:,:]
            img = st.resize(crop_img, (128, 128))
            
            steerings.append(steering_label)
            imgs.append(img)
            
    return imgs, steerings

def read_dave2_data(steering_image_log, image_folder, n=0):
    steerings = []
    imgs = []

    with open(steering_image_log) as f:
        if n != 0:
            selected_lines = random.sample(f.readlines(), n)
        else:
            selected_lines = f.readlines()
        for line in tqdm(selected_lines):
            steering = float(line.split()[1]) * math.pi / 180
            steering_label = np.zeros(3)

            if steering > 0.15: # right
                steering_label[2] = 1
            elif steering < -0.15: # left
                steering_label[0] = 1
            else: # straight
                steering_label[1] = 1
            
            img = im.imread(path.join(image_folder, line.split()[0]))
            crop_img = img[80:,:]
            img = st.resize(crop_img, (128, 128))

            steerings.append(steering_label)
            imgs.append(img)

    return imgs, steerings

def softmax(x):
    """Compute softmax values for each sets of scores in x."""

    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

class SDC_data:
    def __init__(self, image_file, image_folder):
             
        data, labels = read_images_steering_directions(image_file,image_folder)
        
        self.attack_data = np.asarray(data)
        self.attack_labels = np.asarray(labels)

        print('Data shape:', self.attack_data.shape)
        print('Labels shape:', self.attack_labels.shape)

class Dave_data:
    def __init__(self, image_file, image_folder, n=0):
        data, labels = read_dave2_data(image_file, image_folder, n)
        
        self.attack_data = np.asarray(data)
        self.attack_labels = np.asarray(labels)

        # print('Data shape:', self.attack_data.shape)
        # print('Labels shape:', self.attack_labels.shape)

        self.train_X, self.test_X, self.train_y, self.test_y = train_test_split(self.attack_data, self.attack_labels, test_size=0.2, random_state=42)
        self.train_X, self.val_X, self.train_y, self.val_y = train_test_split(self.train_X, self.train_y, test_size=0.2, random_state=42)
        # train : val : test = 64 : 16 : 20


def generate_data(data, samples, targeted=True, start=0, inception=False):
    """
    Generate the input data to the attack algorithm.
    data: the images to attack
    samples: number of samples to use
    targeted: if true, construct targeted attacks, otherwise untargeted attacks
    start: offset into data to use
    inception: if targeted and inception, randomly sample 100 targets intead of 1000
    """
    inputs = []
    targets = []
    for i in range(samples):
        if targeted:
            if inception:
                seq = random.sample(range(1,1001), 10)
            else:
                seq = range(data.attack_labels.shape[1])

            for j in seq:
                if (j == np.argmax(data.attack_labels[start+i])) and (inception == False):
                    continue
                inputs.append(data.attack_data[start+i])
                targets.append(np.eye(data.attack_labels.shape[1])[j])
        else:
            inputs.append(data.attack_data[start+i])
            targets.append(data.attack_labels[start+i])

    inputs = np.array(inputs)
    targets = np.array(targets)

    return inputs, targets

'''
def data_generator(xs, ys, target_size=(100,100), batch_size=64):
    gen_state = 0
    while 1:
        if gen_state + batch_size > len(xs):
            paths = xs[gen_state: len(xs)]
            y = ys[gen_state: len(xs)]
            X = [preprocess(x) for x in paths]
            gen_state = 0
        else:
            paths = xs[gen_state: gen_state + batch_size]
            y = ys[gen_state: gen_state + batch_size]
            X = [preprocess(x) for x in paths]
            gen_state += batch_size
        yield np.array(X), np.array(y)

def val_generator(xs, ys, batch_size=64):
    gen_state = 0
    while 1:
        if gen_state + batch_size > len(xs):
            paths = xs[gen_state: len(xs)]
            y = ys[gen_state: len(xs)]
            X = [preprocess(x) for x in paths]
            yield np.array(X), np.array(y) #python2, use yield then return
            return
        else:
            paths = xs[gen_state: gen_state + batch_size]
            y = ys[gen_state: gen_state + batch_size]
            X = [preprocess(x) for x in paths]
            gen_state += batch_size
            yield np.array(X), np.array(y)
'''

def val_generator(xs, ys, batch_size=64):
    _size = len(xs)
    for start_idx in range(0, _size, batch_size):
        end_idx = min(start_idx + batch_size, _size)
        yield xs[start_idx:end_idx], ys[start_idx:end_idx]

def data_generator(xs, ys, batch_size=64):
    _size = len(xs)
    while True:
        for start_idx in range(0, _size, batch_size):
            end_idx = min(start_idx + batch_size, _size)
            yield xs[start_idx:end_idx], ys[start_idx:end_idx]
