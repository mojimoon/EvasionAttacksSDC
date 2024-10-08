import numpy as np

from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.python.keras.utils import np_utils
from tensorflow.keras.models import load_model

import imageio as im
import skimage.transform as st

from os import path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import roc_curve,auc
from scipy import interp
from itertools import cycle

from utilities import read_images_steering_directions, generate_data, SDC_data
   
IMAGE_FILE = 'straight_right_left.csv'
IMAGE_FOLDER = '/home/alesia/Documents/sdc/'


data = SDC_data(IMAGE_FILE, IMAGE_FOLDER)

train_labels = np.asarray(data.attack_labels)   



NUM_ATTACKS = 600 
NUM_CLASSES = 3
NUM_INPUTS = 300

LABELS_FILE = 'results/res_attack_labels_epoch.txt'
PROBAS_FILE = 'results/res_attack_probas_epoch.txt'
DISTANCES_FILE = 'results/res_attack_epoch.txt'


shape = (NUM_ATTACKS, NUM_CLASSES)

train_labels_600 = np.zeros(shape)


for i in range(0, NUM_ATTACKS, 2):
    train_labels_600[i] = train_labels[int(i/2)]

for i in range(1, NUM_ATTACKS, 2):
    train_labels_600[i] = train_labels[int(np.floor(i/2))]
    

predicted_probabilities_array = np.zeros(shape)

for k in range (NUM_INPUTS):

    inputs, targets = generate_data(data, samples = 1, targeted=True, start=k, inception=False)

predicted_probabilities_array = np.zeros(shape)
probabilities_attack_array = np.zeros(shape)

probabilities_array_071 = np.zeros(shape)
probabilities_array_072 = np.zeros(shape)
probabilities_array_075 = np.zeros(shape)

#get prediction probas
with open(LABELS_FILE, 'r') as f:
    predictions = f.readlines()

#get resulting distances
with open(DISTANCES_FILE, 'r') as f:
    distances = f.read()

distances = distances[1:len(distances)-1]
distances_str = distances.split(',')

for i in range(len(distances_str)):
    distances_str[i] = float(distances_str[i])

#get resulting probabilities
with open(PROBAS_FILE, 'r') as f:
    adversaries = f.readlines()

adversaries = [x.strip() for x in adversaries] 

for  i in range(0, NUM_ATTACKS):

    str_adv = adversaries[i]
    str_adv = str_adv.replace('[','')
    str_adv = str_adv.replace(']','')
    probabilities_attack_array[i] = str_adv.split(',')

    str_pred = predictions[i]
    str_pred = str_pred.replace('[','')
    str_pred = str_pred.replace(']','')
    predicted_probabilities_array[i] = str_pred.split(',')
    
    if(distances_str[i]< 0.71):
        probabilities_array_071[i] =   probabilities_attack_array[i]
    else:
        probabilities_array_071[i] =  predicted_probabilities_array[i]

    if(distances_str[i]< 0.72):
        probabilities_array_072[i] =  probabilities_attack_array[i]
    else:
        probabilities_array_072[i] =  predicted_probabilities_array[i]
    
    if(distances_str[i]< 0.75):
        probabilities_array_075[i] = probabilities_attack_array[i]
    else:
        probabilities_array_075[i] =  predicted_probabilities_array[i]

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

fpr["micro"], tpr["micro"], _ = roc_curve(train_labels_600.ravel(), predicted_probabilities_array.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

fpr["071"], tpr["071"], thres_071 = roc_curve(train_labels_600.ravel(), probabilities_array_071.ravel(), drop_intermediate = False)
roc_auc["071"] = auc(fpr["071"], tpr["071"])

fpr["072"], tpr["072"], thres_072 = roc_curve(train_labels_600.ravel(), probabilities_array_072.ravel(),  drop_intermediate = False)
roc_auc["072"] = auc(fpr["072"], tpr["072"])

fpr["075"], tpr["075"], thres_075 = roc_curve(train_labels_600.ravel(), probabilities_array_075.ravel(),  drop_intermediate = False)
roc_auc["075"] = auc(fpr["075"], tpr["075"])

plt.figure()
lw = 2

plt.plot(fpr["micro"], tpr["micro"], label='No attack (AUC = {0:0.2f})'''.format(roc_auc["micro"]), color='purple', lw=lw)
plt.plot(fpr["071"], tpr["071"], label = 'Attack d = 0.71 (AUC = {0:0.2f})'''.format(roc_auc["071"]), color='darkorange', lw=lw)
plt.plot(fpr["072"], tpr["072"], label = 'Attack d = 0.72 (AUC = {0:0.2f})'''.format(roc_auc["072"]), color='red', lw=lw)
plt.plot(fpr["075"], tpr["075"], label = 'Attack d = 0.75 (AUC = {0:0.2f})'''.format(roc_auc["075"]), color='blue', lw=lw)


plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([-0.03, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.legend(loc="lower right", fontsize=11)
plt.savefig('results/epoch_roc.png', dpi = 250)















