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

#TODO separate activation layers?
#BatchNormalization layer after each conv layer
mp_stride = (2,2)
mp_size = (2,2)
input_shape
X_train[0].shape

# %%
model = Sequential()

if True:
    #C1
    model.add(Conv2D(96, (7,7),padding='valid',activation='relu',input_shape=input_shape))
    # model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=mp_size,strides=mp_stride))
    #C2
    model.add(Conv2D(128, (5,5),padding='valid',activation='relu'))
    # model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=mp_size,strides=mp_stride))
    #C3
    model.add(Conv2D(128, (3,3),padding='valid',activation='relu'))
    # model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=mp_size,strides=mp_stride))
    #C4
    model.add(Conv2D(128, (3,3),padding='valid',activation='relu'))
    # model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=mp_size,strides=mp_stride))

    model.add(Flatten())

    #FC1
    model.add(Dense(1024, input_shape=(10*3*128,),activation='relu')) #TODO inputshape
    # model.add(Dropout(0.5))

    #FC2
    model.add(Dense(512 ,activation='relu')) #TODO inputshape

    #FC- OUTPUT, classification to classes, softmax
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
else:
    #Instantiate an empty model
    model = Sequential()

    # 1st Convolutional Layer
    model.add(Conv2D(filters=96, input_shape=input_shape, kernel_size=(11,11), strides=(4,4), padding='valid'))
    model.add(Activation('relu'))
    # Max Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    # 2nd Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))
    # Max Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    # 3rd Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))
    # 4th Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))
    # 5th Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))
    # Max Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    # Passing it to a Fully Connected layer
    model.add(Flatten())
    # 1st Fully Connected Layer
    model.add(Dense(4096, input_shape=(224*224*3,)))
    model.add(Activation('relu'))
    # Add Dropout to prevent overfitting
    model.add(Dropout(0.4))
    # 2nd Fully Connected Layer
    model.add(Dense(4096))
    model.add(Activation('relu'))
    # Add Dropout
    model.add(Dropout(0.4))
    # 3rd Fully Connected Layer
    model.add(Dense(1000))
    model.add(Activation('relu'))
    # Add Dropout
    model.add(Dropout(0.4))
    # Output Layer
    model.add(Dense(17))
    model.add(Activation('softmax'))


# Compile the model
# model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=[“accuracy”])
print('model summary:')
model.summary()
# %%


print('compling model')

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss=keras.losses.categorical_crossentropy,
        # optimizer = 'adam',
        optimizer = 'sgd',
        metrics=['accuracy'])

print('training model')
# model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(X_test, Y_test))

# %%
from skimage import exposure

def AHE(img):
    img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.03)
    return img_adapteq
datagen = ImageDataGenerator(rotation_range=30, horizontal_flip=0.5, preprocessing_function=AHE)

print('Using real-time data augmentation.')
# This will do preprocessing and realtime data augmentation:
datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    zca_epsilon=1e-06,  # epsilon for ZCA whitening
    rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
    # randomly shift images horizontally (fraction of total width)
    width_shift_range=0.1,
    # randomly shift images vertically (fraction of total height)
    height_shift_range=0.1,
    shear_range=0.,  # set range for random shear
    zoom_range=0.,  # set range for random zoom
    channel_shift_range=0.,  # set range for random channel shifts
    # set mode for filling points outside the input boundaries
    fill_mode='nearest',
    cval=0.,  # value used for fill_mode = "constant"
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False,  # randomly flip images
    # set rescaling factor (applied before any other transformation)
    rescale=None,
    # set function that will be applied on each input
    preprocessing_function=None,
    # image data format, either "channels_first" or "channels_last"
    data_format=None,
    # fraction of images reserved for validation (strictly between 0 and 1)
    validation_split=0.0)

datagen.fit(X_train)
# %%


# model.fit(X_train, Y_train, batch_size=20, epochs=50, verbose=1, validation_data=(X_test, Y_test))
model.fit_generator(
    datagen.flow(X_train, Y_train, batch_size=20),
    steps_per_epoch=X_train.shape[0]/10,
    epochs=10, verbose=1, validation_data=(X_test, Y_test))

print('evaluating model')
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
os.system( "say beep" )
