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

# %% dataprep
def get_pairs():
    train_files = list(sorted(glob.glob('./data/THU-DorsalFinger/FDT2_Train/*/*.bmp')))
    test_files = list(sorted(glob.glob('./data/THU-DorsalFinger/FDT2_Test/*/*.bmp')))

    train_hash = hash(''.join([f[-7:] for f in train_files]))
    test_hash = hash(''.join([f[-7:] for f in test_files]))
    assert test_hash == train_hash

    pairs = []

    for file_train,file_test in zip(train_files,test_files):
        subject_id = int(file_test.split('/')[-2])
        img_train = load_img(file_train,color_mode = "grayscale")
        arr_train = img_to_array(img_train)
        img_test = load_img(file_test,color_mode = "grayscale")
        arr_test = img_to_array(img_test)
        pair.append((arr_test,arr_train))
    return pairs
