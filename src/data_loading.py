from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from IPython.display import display
from PIL import Image
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

train_hash = hash(''.join([f[-7:] for f in train_files]))
test_hash = hash(''.join([f[-7:] for f in test_files]))
assert test_hash == train_hash

x_train = []
y_train = []
x_test = []
y_test = []

for file_train,file_test in zip(train_files,test_files):
    # print(file_train)
    # print(file_test)
    subject_id = int(file_test.split('/')[-2])

    img_train = load_img(file_train,color_mode = "grayscale")
    arr_train = img_to_array(img_train)

    img_test = load_img(file_test,color_mode = "grayscale")
    arr_test = img_to_array(img_test)

    x_train.append(arr_train)
    y_train.append(subject_id)
    x_test.append(arr_test)
    y_test.append(subject_id)

# %% ---- ABOVE BEEN COPIED

set([i.shape for i in x_train])

batch_size = 32
num_classes = 10
epochs = 100
data_augmentation = True
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_trained_model.h5'

(x_train1, y_train1), (x_test1, y_test1) = cifar10.load_data()
type(x_train1)
x_train1.shape
x_train[0].shape
#model.add(Conv2D(32, (3, 3), padding='same',input_shape=x_train.shape[1:]))



type(x_train)
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
