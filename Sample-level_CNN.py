'''
# **Spoken digit classifier using Sample-level CNN**

Notice: This code is written for practice purposes.

Sample-level CNN classifies the spoken digit (0-9) on sample-level, without converting the signal into time-frequency level.


*   The source of the dataset of mnist-speech is as follows:
https://github.com/Jakobovski/free-spoken-digit-dataset
(4 speakers, 2,000 recordings, 50 of each digit per speaker)
*   The dataset is divided manually.
(Train: 1,200, Valid: 400, Test: 400)
*   The layer of CNN is constructed with reference to the following paper:
https://arxiv.org/pdf/1703.01789.pdf
'''

import tensorflow as tf
import numpy as np
from random import shuffle
from scipy.io.wavfile import read
from tensorflow.python.keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
import os


# Function to load dataset.
# The data is zero-padded after loaded.
# Padded data size per one wav file: 19683
def make_dataset(file_path, resample_size):
    # list of wav file
    file_list = os.listdir(file_path)

    # shuffle the dataset
    shuffle(file_list)
    data, labels = [], []


    for file_name in file_list:
        _, wav_file = read(file_path + file_name)

        # initialize the array which will be used as padded-data
        padded = np.zeros(resample_size)

        # Get the wav file length, and then insert it into the middle of zero-padded data
        wav_file_len = len(wav_file)
        padded_center = resample_size // 2
        wav_center = wav_file_len // 2
        front = padded_center - wav_center
        padded[front:front + wav_file_len] = wav_file

        # Append the data and the label as the dataset
        data.append(padded)
        labels.append(file_name[0])

    data = np.array(data, dtype=float)
    labels = np.array(labels, dtype=int)

    total_data = len(data)
    data = data.reshape(total_data, resample_size, 1)

    return data, labels


# Function of the layers
# The layer of CNN is constructed with reference to the following paper:
# https://arxiv.org/pdf/1703.01789.pdf
def top_layer(X_input, filters=128, kernel_size=3, strides=3, padding='same', activation='relu'):
    X = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)(X_input)
    X = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.Activation(activation)(X)

    return X


def module_layer(X, filters=128, kernel_size=3, conv_strides=1,
                 pool_size=3, pool_strides=3, padding='same', activation='relu'):
    X = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=conv_strides, padding=padding)(X)
    X = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.Activation(activation)(X)
    X = tf.keras.layers.MaxPool1D(pool_size=pool_size, strides=pool_strides)(X)

    return X


def bottom_layer(X, num_classes=10, filters=512, kernel_size=1, strides=1, padding='same',
                 conv_active='relu', fc_active='sigmoid'):
    X = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)(X)
    X = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.Activation(conv_active)(X)
    X = tf.keras.layers.Dropout(0.5)(X)
    X = tf.keras.layers.Flatten()(X)
    X = tf.keras.layers.Dense(num_classes, activation=fc_active)(X)

    return X


# Loads dataset and sets hyperparameter
# The layer of CNN in this paper is called m^nDCNN.
# The input of this network should be the power of m,
# which refers to the filter length and pooling length of intermediate convolution layer,
# and n refers to the number of modules. This paper recommends to use the filter length as 3.
# Therefore, the longest data of this dataset is around 18000,
# I choose the length of zero-padded data (resample_size) as 19683, which is the power of 3.


# path direction
train_path_dir = 'recordings/training/'
valid_path_dir = 'recordings/validation/'
test_path_dir = 'recordings/test/'

# hyper-parameters
resample_size = 19683
num_classes = 10
lr = 0.01
input_shape = (resample_size, 1)
batch_size = 100
num_epochs = 30
m = 3

# divide the dataset into train, validation and test.
train_data, train_labels = make_dataset(train_path_dir, resample_size)
valid_data, valid_labels = make_dataset(valid_path_dir, resample_size)
test_data, test_labels = make_dataset(test_path_dir, resample_size)

train_labels_one_hot = tf.keras.utils.to_categorical(train_labels, num_classes)
valid_labels_one_hot = tf.keras.utils.to_categorical(valid_labels, num_classes)
test_labels_one_hot = tf.keras.utils.to_categorical(test_labels, num_classes)

print('The number of train dataset: {}'.format(len(train_data)))
print('The number of valid dataset: {}'.format(len(valid_data)))
print('The number of test dataset: {}'.format(len(test_data)))

# SampleCNN layer
# Set the optimizer as SGD and print summary
with tf.name_scope("SampleCNN"):
    X_input = tf.keras.layers.Input(input_shape)

    X = top_layer(X_input, filters=128, kernel_size=m, strides=m, padding='same', activation='relu')
    # modules start
    X = module_layer(X, filters=128, kernel_size=m, conv_strides=1,
                     pool_size=m, pool_strides=m, padding='same', activation='relu')
    X = module_layer(X, filters=128, kernel_size=m, conv_strides=1,
                     pool_size=m, pool_strides=m, padding='same', activation='relu')
    X = module_layer(X, filters=256, kernel_size=m, conv_strides=1,
                     pool_size=m, pool_strides=m, padding='same', activation='relu')
    X = module_layer(X, filters=256, kernel_size=m, conv_strides=1,
                     pool_size=m, pool_strides=m, padding='same', activation='relu')
    X = module_layer(X, filters=256, kernel_size=m, conv_strides=1,
                     pool_size=m, pool_strides=m, padding='same', activation='relu')
    X = module_layer(X, filters=256, kernel_size=m, conv_strides=1,
                     pool_size=m, pool_strides=m, padding='same', activation='relu')
    X = module_layer(X, filters=256, kernel_size=m, conv_strides=1,
                     pool_size=m, pool_strides=m, padding='same', activation='relu')
    X = module_layer(X, filters=512, kernel_size=m, conv_strides=1,
                     pool_size=m, pool_strides=m, padding='same', activation='relu')
    # modules end
    X = bottom_layer(X, num_classes=num_classes, filters=512, kernel_size=1, strides=1, padding='same',
                     conv_active='relu', fc_active='sigmoid')

    model = tf.keras.models.Model(inputs=X_input, outputs=X)

sgd = tf.keras.optimizers.SGD(lr=lr, momentum=0.9, nesterov=True)
model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=sgd,
              metrics=['accuracy']
              )

model.summary()

# Starts training
history = model.fit(x=train_data, y=train_labels_one_hot, batch_size=batch_size, epochs=num_epochs,
                    validation_data=(valid_data, valid_labels_one_hot))

# Evaluates the training result
test_loss, test_acc = model.evaluate(x=test_data, y=test_labels_one_hot)

# plot loss and accuracy of train and valid
fig, loss_ax = plt.subplots()

acc_ax = loss_ax.twinx()

loss_ax.plot(history.history['loss'], 'y', label='train loss')
loss_ax.plot(history.history['val_loss'], 'r', label='val loss')

acc_ax.plot(history.history['accuracy'], 'b', label='train acc')
acc_ax.plot(history.history['val_accuracy'], 'g', label='val acc')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuray')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')

plt.show()

