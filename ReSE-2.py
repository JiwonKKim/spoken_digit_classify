import tensorflow as tf
import numpy as np
from random import shuffle
from scipy.io.wavfile import read
from scipy.io.wavfile import write
from scipy.signal import resample
from tensorflow.python.keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
import os


def make_dataset(file_path, resample_size, max_len):
    file_list = os.listdir(file_path)
    shuffle(file_list)
    # file_list.sort()
    data, labels = [], []

    for file_name in file_list:
        _, wav_file = read(file_path + file_name)
        padded = np.zeros(resample_size)
        wav_file_len = len(wav_file)

        if wav_file_len > max_len:
            max_len = wav_file_len

        padded_center = resample_size // 2
        wav_center = wav_file_len // 2
        front = padded_center - wav_center
        padded[front:front + wav_file_len] = wav_file
        # padded = padded / np.max(padded)
        data.append(padded)
        labels.append(file_name[0])

    data = np.array(data, dtype=float)
    labels = np.array(labels, dtype=int)

    total_data = len(data)
    data = data.reshape(total_data, resample_size, 1)

    return data, labels, max_len

# def make_dataset(file_path, resample_size, max_len):
#     file_list = os.listdir(file_path)
#     shuffle(file_list)
#     # file_list.sort()
#     data, labels = [], []
#
#     for file_name in file_list:
#         _, wav_file = read(file_path + file_name)
#         wav_file = resample(wav_file, resample_size)
#         # wav_file = wav_file / np.max(wav_file)
#         data.append(wav_file)
#         labels.append(file_name[0])
#
#     data = np.array(data, dtype=float)
#     labels = np.array(labels, dtype=int)
#
#     total_data = len(data)
#     data = data.reshape(total_data, resample_size, 1)
#
#     return data, labels, max_len


train_path_dir = 'recordings/training/'
valid_path_dir = 'recordings/validation/'
test_path_dir = 'recordings/test/'

resample_size = 19683
# resample_size = 59049

max_len_train = 0
max_len_valid = 0
max_len_test = 0

train_data, train_labels, max_len_train = make_dataset(train_path_dir, resample_size, max_len_train)
valid_data, valid_labels, max_len_valid = make_dataset(valid_path_dir, resample_size, max_len_valid)
test_data, test_labels, max_len_test = make_dataset(test_path_dir, resample_size, max_len_test)

# print(max_len_train)
# print(max_len_valid)
# print(max_len_test)

num_classes = 10
lr = 0.01
input_shape = (resample_size, 1)
batch_size = 100
num_epochs = 80

tensorboard = TensorBoard(log_dir="logs/")

train_labels_one_hot = tf.keras.utils.to_categorical(train_labels, num_classes)
valid_labels_one_hot = tf.keras.utils.to_categorical(valid_labels, num_classes)
test_labels_one_hot = tf.keras.utils.to_categorical(test_labels, num_classes)

# train_labels_one_hot = train_labels_one_hot.reshape(len(train_labels), 1, num_classes)
# valid_labels_one_hot = valid_labels_one_hot.reshape(len(valid_labels), 1, num_classes)
# test_labels_one_hot = test_labels_one_hot.reshape(len(test_labels), 1, num_classes)

m = 3

with tf.name_scope("Layers"):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv1D(filters=128, kernel_size=m, strides=m, padding='same', input_shape=input_shape))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    # modules start
    model.add(tf.keras.layers.Conv1D(filters=128, kernel_size=m, strides=1, padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPool1D(pool_size=m, strides=m))

    model.add(tf.keras.layers.Conv1D(filters=128, kernel_size=m, strides=1, padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPool1D(pool_size=m, strides=m))
    model.add(tf.keras.layers.Conv1D(filters=256, kernel_size=m, strides=1, padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPool1D(pool_size=m, strides=m))
    model.add(tf.keras.layers.Conv1D(filters=256, kernel_size=m, strides=1, padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPool1D(pool_size=m, strides=m))
    model.add(tf.keras.layers.Conv1D(filters=256, kernel_size=m, strides=1, padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPool1D(pool_size=m, strides=m))
    model.add(tf.keras.layers.Conv1D(filters=256, kernel_size=m, strides=1, padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPool1D(pool_size=m, strides=m))
    model.add(tf.keras.layers.Conv1D(filters=256, kernel_size=m, strides=1, padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPool1D(pool_size=m, strides=m))
    model.add(tf.keras.layers.Conv1D(filters=512, kernel_size=m, strides=1, padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPool1D(pool_size=m, strides=m))
    # modules end
    model.add(tf.keras.layers.Conv1D(filters=512, kernel_size=1, strides=1, padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(num_classes, activation='sigmoid'))

sgd = tf.keras.optimizers.SGD(lr=lr, momentum=0.9, nesterov=True)
model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=sgd,
              metrics=['accuracy']
              )

model.summary()

history = model.fit(x=train_data, y=train_labels_one_hot, batch_size=batch_size, epochs=num_epochs,
                    validation_data=(valid_data, valid_labels_one_hot), callbacks=[tensorboard])

test_loss, test_acc = model.evaluate(x=test_data, y=test_labels_one_hot)
