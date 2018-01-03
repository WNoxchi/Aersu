# Wayne Nixalo - 2018-Jan-02 23:15

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

### INIT TF & MODEL
def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True # will this eat all GRAM?
    return tf.Session(config=config)

keras.backend.tensorflow_backend.set_session(get_session())

model = keras.models.load_model('data/retinanet-model/resnet50_coco_best_v1.2.2.h5',
                                custom_objects=custom_objects)

tpath = 'data/train/'
folders = os.listdir(tpath)

# TODO: loop through folders

# TODO: load image file

# image = Image.open(image)
# image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

# detect(image)


# TEST:
x = detect(image=cv2.cvtColor(np.array(Image.open('data/train/006440-006548/006530.jpg')),
                              cv2.COLOR_RGB2BGR), model=model)

if type(x) == np.ndarray:
    print(f'Chosen Bounding Box: {x}')


















#
