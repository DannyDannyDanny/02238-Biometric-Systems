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
from skimage import exposure


# %% dataprep
def get_pairs(squeeze=False):
    train_files = list(sorted(glob.glob('./data/THU-DorsalFinger/FDT2_Train/*/*.bmp')))
    test_files = list(sorted(glob.glob('./data/THU-DorsalFinger/FDT2_Test/*/*.bmp')))

    train_hash = hash(''.join([f[-7:] for f in train_files]))
    test_hash = hash(''.join([f[-7:] for f in test_files]))
    assert test_hash == train_hash

    pairs = []

    for file_train,file_test in zip(train_files,test_files):
        subject_id = int(file_test.split('/')[-2])

        img_train = load_img(file_train,color_mode = "grayscale")
        arr_train = exposure.equalize_hist(img_to_array(img_train))
        if squeeze: arr_train = np.squeeze(arr_train)

        img_test = load_img(file_test,color_mode = "grayscale")
        arr_test = exposure.equalize_hist(img_to_array(img_test))
        if squeeze: arr_test = np.squeeze(arr_test)

        pairs.append([arr_test,arr_train])
    return pairs

# %%

def get_x_y(ttr=0.2,squeeze=False,v=0):
    all_pairs = np.array(get_pairs(squeeze=squeeze))

    # assert all images unique
    string_imgs = []
    for pair in all_pairs:
        string_imgs.append(pair[0].tostring())
        string_imgs.append(pair[1].tostring())
    assert len(string_imgs) == len(set(string_imgs))
    del string_imgs

    # split data for matching pairs and non-matching pairs
    half_len = len(all_pairs)//2
    # (already) matching pairs
    ymatch = all_pairs[:half_len]
    # remaining (matching) pairs
    rem = all_pairs[half_len:]
    # generating class shuffled pairs
    no_matches = False
    while not no_matches:
        ran1 = [i for i in range(half_len)]
        ran2 = [i for i in range(half_len)]
        r.shuffle(ran1)
        r.shuffle(ran2)
        no_matches = sum([a==b for (a,b) in zip(ran1,ran2)]) == 0
        if v >= 1:
            if no_matches:
                print('Non-matching list generated')
            else:
                print('No matches:',no_matches)

    nmatch = np.array([[rem[a][0],rem[b][1]] for (a,b) in zip(ran1,ran2)])
    ymatch = np.array(ymatch)

    # add class
    ymatch = [[1,i] for i in ymatch]
    nmatch = [[0,i] for i in nmatch]
    # merge and shuffle
    ttr = 0.2
    tts = int(len(ymatch)*ttr)
    te_pairs = ymatch[tts:] + nmatch[tts:]
    tr_pairs = ymatch[:tts] + nmatch[:tts]
    r.shuffle(te_pairs)
    r.shuffle(tr_pairs)

    te_x = np.array([x for y,x in te_pairs])
    tr_x = np.array([x for y,x in tr_pairs])
    te_y = np.array([y for y,x in te_pairs])
    tr_y = np.array([y for y,x in tr_pairs])

    return te_x, tr_x, te_y, tr_y
    # return data_x,data_y

# %%
te_x, tr_x, te_y, tr_y = get_x_y(squeeze=True)

ste_x, str_x, ste_y, str_y = get_x_y(squeeze=False)
# %%
te_x.shape
te_x[0][0].shape
# %%
ste_x.shape
ste_x[0][0].shape


# %%
all_pairs = get_pairs()
img_origin_lookup_dict = {}
for i,(a,b) in enumerate(all_pairs):
    img_origin_lookup_dict[b.tostring()] = (i,'b')
    img_origin_lookup_dict[a.tostring()] = (i,'a')

# # %%
# match = 0
# nonmatch = 0
# errors = 0
# # %%
# for pair in pairs:
#     a = img_origin_lookup_dict[pair[0].tostring()]
#     b = img_origin_lookup_dict[pair[1].tostring()]
#     if a[0] == b[0]:
#         match += 1
#     else:
#         nonmatch += 1
#
#     if a[1] == b[1]:
#         errors += 1
#
#
# match
# nonmatch
# errors
