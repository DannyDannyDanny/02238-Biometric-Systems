from __future__ import print_function
import random as r
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
        arr_train = np.squeeze((img_to_array(img_train)))
        img_test = load_img(file_test,color_mode = "grayscale")
        arr_test = np.squeeze((img_to_array(img_test)))
        pairs.append([arr_test,arr_train])
    return pairs

def get_x_y():
    all_pairs = np.array(get_pairs())
    half_len = len(all_pairs)//2
    ymatch = all_pairs[:half_len]
    rem = all_pairs[half_len:]
    # generating class shuffled pairs
    no_matches = False
    while not no_matches:
        ran1 = [i for i in range(half_len)]
        ran2 = [i for i in range(half_len)]
        r.shuffle(ran1)
        r.shuffle(ran2)
        no_matches = sum([a==b for (a,b) in zip(ran1,ran2)]) == 0
        if no_matches:
            print('Non-matching list generated')
        else:
            print('No matches:',no_matches)

    nmatch = np.array([[rem[a][0],rem[a][1]] for (a,b) in zip(ran1,ran2)])
    ymatch = np.array(ymatch)
    nmatch.shape
    ymatch.shape

    # add class
    ymatch = [[1,i] for i in ymatch]
    nmatch = [[0,i] for i in nmatch]
    # merge and shuffle
    class_pairs = ymatch + nmatch
    r.shuffle(class_pairs)

    len(class_pairs)
    class_pairs[0][0]
    data_y = np.array([y for y,x in class_pairs])
    data_x = np.array([x for y,x in class_pairs])
    return data_x,data_y
