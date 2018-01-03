# Wayne Nixalo - 2018-Jan-02 23:15 / 2018-Jan-03 12:59

### IMPORTS
import keras
import keras.preprocessing.image

import cv2
import os
import numpy as np
import time

import tensorflow as tf
from PIL import Image

from keras_retinanet.utils.image import preprocess_image, resize_image
from keras_retinanet.models.resnet import custom_objects

from utils.common import c_shift
from utils.common import detect
from utils.common import crop

### INIT TF & MODEL
def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True # will this eat all GRAM?
    return tf.Session(config=config)

keras.backend.tensorflow_backend.set_session(get_session())

model = keras.models.load_model('data/retinanet-model/resnet50_coco_best_v1.2.2.h5',
                                custom_objects=custom_objects)

# tpath = 'data/train/'
# tempath = 'data/tmp/'
# rejectpath = tempath + 'reject/'
# folders = os.listdir(tpath)
# folders.sort()  # subfolders are numerically ordered
#
# for folder in folders:
#     # get all filenames
#     fnames = os.listdir(tpath + folder)
#     fnames.sort()
#
#     # run detection on each file
#     for fname in fnames:
#         image = Image.open(tpath+folder+'/'+fname)
#         image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
#         b = detect(image=image, model=model, mode='ss', fname=fname)
#
#         if type(b)==list:
#             cropped = crop(image, b)
#             cv2.imwrite()
#
# # TODO: crop and save image
#     # if bounding box save to tmp/
#     cropped = crop(image)
#
#     # if rejected save for manual





# image = Image.open(image)
# image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
#
# detect(image)


# TEST:
x = detect(image=cv2.cvtColor(np.array(Image.open('data/train/006440-006548/006530.jpg')),
                              cv2.COLOR_RGB2BGR), model=model, fname='006530.jpg')

if type(x) == np.ndarray:
    print(f'Chosen Bounding Box: {x}')


















#
