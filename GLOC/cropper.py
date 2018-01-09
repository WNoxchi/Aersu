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
from utils.common import save_crop_overlay

### INIT TF & MODEL
def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True # will this eat all GRAM?
    return tf.Session(config=config)

def main():
    """
    Saves images cropped by bounding boxes from common.detect() to a temporary
    folder. If no bounding-box is returned, a copy of the image is saved to a
    reject folder for manual processing.
    """

    keras.backend.tensorflow_backend.set_session(get_session())

    model = keras.models.load_model('data/retinanet-model/resnet50_coco_best_v1.2.2.h5',
                                    custom_objects=custom_objects)

    tpath = 'data/train/'
    tempath = 'data/tmp/'
    rejectpath = tempath + 'reject/'
    folders = os.listdir(tpath)
    folders.sort()  # subfolders are numerically ordered

    for folder in folders:
        # get all filenames
        fnames = os.listdir(tpath + folder)
        fnames.sort()

        # run detection on each file
        for fname in fnames:
            fpath = tpath+folder+'/'+fname
            # image = Image.open(fpath)
            image = cv2.imread(fpath)
            image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
            b = detect(image=image, model=model, mode='ss', fname=fname, quiet=False)

            # crop & save to tmp/ if bounding box
            if type(b)==np.ndarray:
                # add folder if not there
                if folder not in os.listdir(tempath):
                    os.mkdir(tempath + folder)

                cropped = crop(image, b)
                save_crop_overlay(tempath+folder+'/'+'crop_'+fname, cropped)
                # cv2.imwrite(tempath+folder+'/'+'crop_'+fname, cropped)
                # print(tempath+folder+'/'+'crop_'+fname)

            # otherwise save original to reject/ for manual labelling
            elif type(b)==int:
                # Exit Signal
                if b == -1:
                    return
                # add folder if not there
                if folder not in os.listdir(rejectpath):
                    os.mkdir(rejectpath + folder)
                cv2.imwrite(rejectpath+folder+'/'+fname, image)

    return

if __name__ =='__main__':
    main()



# image = Image.open(image)
# image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
#
# detect(image)


# # TEST:
# x = detect(image=cv2.cvtColor(np.array(Image.open('data/train/006440-006548/006530.jpg')),
#                               cv2.COLOR_RGB2BGR), model=model, fname='006530.jpg')
#
# if type(x) == np.ndarray:
#     print(f'Chosen Bounding Box: {x}')


















#