import tensorflow as tf

# for Python 3.6
from keras.models import Sequential
from keras.layers import Conv2D, Activation, MaxPooling2D, Dropout, Flatten, Dense
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping

# for Python 2.7
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, Activation, MaxPooling2D, Dropout, Flatten, Dense
# from tensorflow.keras.optimizers import SGD
# from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from kerastuner import HyperModel, RandomSearch

import pandas as pd
import numpy as np
import imageio as im
import os
import skimage.transform as st
from utilities import Dave_data

class CNNHyperModel(HyperModel):
    def build(self, hp):
        model = Sequential()
        
        # Hyperparameters
        num_conv_layers = hp.Int('num_conv_layers', 2, 4)
        # filters = [hp.Int(f'filters_{i}', 32, 128, step=32) for i in range(num_conv_layers)]
        # wtf this is Python 2, cannot use f-string
        filters = [hp.Int('filters_' + str(i), 32, 128, step=32) for i in range(num_conv_layers)]
        dense_units = hp.Int('dense_units', 64, 256, step=64)
        
        model.add(Conv2D(filters[0], (3, 3), input_shape=(data.attack_data.shape[1:]), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        for i in range(1, num_conv_layers):
            model.add(Conv2D(filters[i], (3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.25))
        
        model.add(Flatten())
        model.add(Dense(dense_units, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(3, activation='softmax'))  # 3 classes

        sgd = SGD(lr=hp.Float('lr', 1e-4, 1e-2, sampling='LOG'), decay=1e-6, momentum=0.9, nesterov=True)

        model.compile(loss='categorical_crossentropy',  # Use categorical_crossentropy for one-hot encoded labels
                      optimizer=sgd,
                      metrics=['accuracy'])
        return model

def train(data, file_name, num_epochs=50, batch_size=256):
    tuner = RandomSearch(
        CNNHyperModel(),
        objective='val_loss',
        max_trials=10,
        executions_per_trial=1,
        directory='my_dir',
        project_name='cnn_hyperparam_tuning'
    )

    # Split the data into training and validation sets
    train_data, val_data = data.attack_data[:int(len(data.attack_data) * 0.8)], data.attack_data[int(len(data.attack_data) * 0.8):]
    train_labels, val_labels = data.attack_labels[:int(len(data.attack_labels) * 0.8)], data.attack_labels[int(len(data.attack_labels) * 0.8):]

    # Callback to log results
    class CSVLogger(tf.keras.callbacks.Callback):
        def __init__(self, filename):
            super().__init__()
            self.filename = filename
            self.results = []

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            self.results.append(logs)

        def on_train_end(self, logs=None):
            df = pd.DataFrame(self.results)
            df.to_csv(self.filename, index=False)

    logger = CSVLogger(file_name + "_results.csv")

    tuner.search(train_data, train_labels,
                 epochs=num_epochs,
                 batch_size=batch_size,
                 validation_data=(val_data, val_labels),
                 callbacks=[ModelCheckpoint(file_name + "_best_model.h5", save_best_only=True),
                            EarlyStopping(patience=10, restore_best_weights=True),
                            logger])

    # Get the best model
    best_model = tuner.get_best_models(num_models=1)[0]

    # Save the best model if needed
    best_model.save(file_name)

    # Log hyperparameters and results
    tuner_results = tuner.oracle.get_best_trials(num_trials=tuner.oracle.trials)
    trials_data = []
    for trial in tuner_results:
        trials_data.append({
            'trial_id': trial.trial_id,
            'num_conv_layers': trial.hyperparameters.values['num_conv_layers'],
            # 'filters': [trial.hyperparameters.values[f'filters_{i}'] for i in range(trial.hyperparameters.values['num_conv_layers'])],
            'filters': [trial.hyperparameters.values['filters_' + str(i)] for i in range(trial.hyperparameters.values['num_conv_layers'])],
            'dense_units': trial.hyperparameters.values['dense_units'],
            'learning_rate': trial.hyperparameters.values['lr'],
            'val_loss': trial.metrics.get("val_loss"),
            'val_accuracy': trial.metrics.get("val_accuracy"),
        })

    # Saving hyperparameters and metrics to a CSV file
    trials_df = pd.DataFrame(trials_data)
    trials_df.to_csv(file_name + "_trials_results.csv", index=False)

    return best_model

IMAGE_FILE = '/home/jzhang2297/data/dave_test/driving_dataset/data.txt'
IMAGE_FOLDER = '/home/jzhang2297/data/dave_test/driving_dataset/'

data = Dave_data(IMAGE_FILE, IMAGE_FOLDER)

model = train(data, 'model/tuned_cnn', num_epochs=50, batch_size=256)
