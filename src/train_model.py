from __future__ import print_function
import numpy as np
import pandas as pd
from PIL import Image
from keras.utils import np_utils
from keras import backend as K
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import SGD
from IPython.display import display
import os
import glob
import numpy as np
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
# from keras.preprocessing.image import array_to_img
# img_pil = array_to_img(img_array)
# img_pil.show()


# %% dataprep
train_files = list(sorted(glob.glob('./data/THU-DorsalFinger/FDT2_Train/*/*.bmp')))
test_files = list(sorted(glob.glob('./data/THU-DorsalFinger/FDT2_Test/*/*.bmp')))

assert len(train_files) == 610

train_hash = hash(''.join([f[-7:] for f in train_files]))
test_hash = hash(''.join([f[-7:] for f in test_files]))
assert test_hash == train_hash

x_train = []
y_train = []
x_test = []
y_test = []

print('loading data')
for file_train,file_test in zip(train_files,test_files):
    subject_id = int(file_test.split('/')[-2])-1

    img_train = load_img(file_train,color_mode = "grayscale")
    arr_train = img_to_array(img_train)

    img_test = load_img(file_test,color_mode = "grayscale")
    arr_test = img_to_array(img_test)

    x_train.append(arr_train)
    y_train.append(subject_id)
    x_test.append(arr_test)
    y_test.append(subject_id)

X_train = np.array(x_train)
y_train = np.array(y_train)
X_test = np.array(x_test)
y_test = np.array(y_test)
# %%
batch_size = 1
nb_classes = len(x_test)
nb_epoch = 1
# input image dimensions
img_rows, img_cols, _ = x_train[0].shape
# number of convolutional filters to use
nb_filters = 7#32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)
# Checking if the backend is Theano or Tensorflow
print('transorming data for appropriate backend')
if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
# %%
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
print('converting categories to binary class matrices')
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

print('building model')
model = Sequential()
model.add(Conv2D(nb_filters, kernel_size[0], kernel_size[1],
                border_mode='valid',
                input_shape=input_shape))
model.add(Activation('relu'))
model.add(Conv2D(nb_filters, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

print('compling model')
model.compile(loss='categorical_crossentropy',
        optimizer='sgd',
        metrics=['accuracy'])

print('training model')
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(X_test, Y_test))
print('evaluating model')
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
