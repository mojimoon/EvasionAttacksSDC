import numpy as np
import tensorflow as tf
import os
import sys

from model import SDC_model_epoch
from utilities import Dave_data
from sklearn.metrics import accuracy_score

pwd = os.getcwd()
IMAGE_FILE = '/home/jzhang2297/data/dave_test/driving_dataset/data.txt'
IMAGE_FOLDER = '/home/jzhang2297/data/dave_test/driving_dataset/'
MODEL_NAME = os.path.join(pwd, "models/sdc_epoch_tf_1.14.h5")

def test(data, model_name):
    """
    Test the model on the data
    """
    with tf.Session() as sess:
        model = SDC_model_epoch(model_name, session = sess)
        print(model.evaluate(data.test_X, data.test_y))


def main():
    test(Dave_data(IMAGE_FILE, IMAGE_FOLDER), MODEL_NAME)

if __name__ == "__main__":
    main()