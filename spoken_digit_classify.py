import tensorflow as tf
import numpy as np
from scipy.io.wavfile import read
from scipy.signal import resample
import os

training_path_dir = 'recordings/training/'
validation_path_dir = 'recordings/validation/'
test_path_dir = 'recordings/test/'

training_file_list = os.listdir(training_path_dir)
validation_file_list = os.listdir(validation_path_dir)
test_file_list = os.listdir(test_path_dir)

training_file_list.sort()
validation_file_list .sort()
test_file_list.sort()

training_sets = []
validation_sets = []
test_sets = []

training_labels = []
validation_labels = []
test_labels = []

resample_size = 6561

for wav_file in training_file_list:
    _, input_data = read(training_path_dir + wav_file)
    input_data = resample(input_data, resample_size, axis=0)
    training_sets.append(input_data)
    training_labels.append(wav_file[0])

for wav_file in validation_file_list:
    _, input_data = read(validation_path_dir + wav_file)
    input_data = resample(input_data, resample_size, axis=0)
    validation_sets.append(input_data)
    validation_labels.append(wav_file[0])

for wav_file in test_file_list:
    _, input_data = read(test_path_dir + wav_file)
    input_data = resample(input_data, resample_size, axis=0)
    test_sets.append(input_data)
    test_labels.append(wav_file[0])

training_sets = np.array(training_sets, dtype=float)
training_labels = np.array(training_labels, dtype=int)

validation_sets = np.array(validation_sets, dtype=float)
validation_labels = np.array(validation_labels, dtype=int)

test_sets = np.array(test_sets, dtype=float)
test_labels = np.array(test_labels, dtype=int)

training_sets = training_sets.reshape(1200, resample_size, 1)
validation_sets = validation_sets.reshape(400, resample_size, 1)
test_sets = test_sets.reshape(400, resample_size, 1)

num_classes = 10
lr = 0.001
input_shape = (resample_size, 1)
batch_size = 100
num_epochs = 10

training_labels_one_hot = tf.keras.utils.to_categorical(training_labels, num_classes)
validation_labels_one_hot = tf.keras.utils.to_categorical(validation_labels, num_classes)
test_labels_one_hot = tf.keras.utils.to_categorical(test_labels, num_classes)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv1D(filters=128, kernel_size=3, strides=3, padding='same', activation='relu', input_shape=input_shape))
model.add(tf.keras.layers.Conv1D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPool1D(pool_size=3, strides=3))
model.add(tf.keras.layers.Conv1D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPool1D(pool_size=3, strides=3))
model.add(tf.keras.layers.Conv1D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPool1D(pool_size=3, strides=3))
model.add(tf.keras.layers.Conv1D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPool1D(pool_size=3, strides=3))
model.add(tf.keras.layers.Conv1D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPool1D(pool_size=3, strides=3))
model.add(tf.keras.layers.Conv1D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPool1D(pool_size=3, strides=3))
model.add(tf.keras.layers.Conv1D(filters=512, kernel_size=3, strides=1, padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPool1D(pool_size=3, strides=3))
model.add(tf.keras.layers.Conv1D(filters=512, kernel_size=1, strides=1, padding='same', activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(num_classes, activation='sigmoid'))

# model.compile(loss=tf.keras.losses.categorical_crossentropy,
model.compile(loss=tf.keras.losses.mean_absolute_error,
              optimizer=tf.keras.optimizers.Adam(lr=lr),
              metrics=['accuracy']
              )

model.summary()

model.fit(x=training_sets, y=training_labels, batch_size=batch_size, epochs=num_epochs,
          validation_data=(validation_sets, validation_labels))
