import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

root = '/home/jzhang2297/wangfan/EvasionAttacksSDC/classification'

input_path = 'logs/log5.txt'
save_path = 'reports/tmp.png'

def read_training_history(input_path):
    '''
    match lines like:
    40865/40865 [==============================] - 498s 12ms/step - loss: 0.0452 - acc: 0.9838 - val_loss: 0.8909 - val_acc: 0.7492
    '''

    history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
    with open(os.path.join(root, input_path), 'r') as f:
        for line in f:
            if 'val_acc' in line:
                line = line.split(' - ')
                for item in line:
                    if 'val_acc' in item:
                        history['val_accuracy'].append(float(item.split(': ')[1]))
                    elif 'val_loss' in item:
                        history['val_loss'].append(float(item.split(': ')[1]))
                    elif 'acc' in item:
                        history['accuracy'].append(float(item.split(': ')[1]))
                    elif 'loss' in item:
                        history['loss'].append(float(item.split(': ')[1]))
    return history

def plot_training_history(history, save_path):
    plt.figure()
    sns.set()
    plt.plot(history['loss'], label='loss')
    plt.plot(history['accuracy'], label='accuracy')
    plt.plot(history['val_loss'], label='val_loss')
    plt.plot(history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.xticks(np.arange(1, len(history['loss'])+1, 1)) # 1-indexed
    plt.title('Training History')
    plt.legend()
    plt.savefig(os.path.join(root, save_path))
    plt.close()

if __name__ == '__main__':
    history = read_training_history(input_path)
    print(history)
    plot_training_history(history, save_path)
