# Wayne Nixalo - 2018-Jan-02 23:15 / 2018-Jan-03 12:59

### IMPORTS
import keras
import keras.preprocessing.image

import cv2
import os
import numpy as np
import time
from pandas import DataFrame

import tensorflow as tf

from keras_retinanet.utils.image import preprocess_image, resize_image
from keras_retinanet.models.resnet import custom_objects

from utils.common import c_shift
from utils.common import detect
from utils.common import crop
from utils.common import overlay_image
from utils.common import bbx_to_DataFrame

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

    interstage_ids = []
    interstage_bbx = []

    tpath = 'data/train/'
    tempath = 'data/tmp/'
    rejectpath = tempath + 'reject/'
    folders = os.listdir(tpath)
    folders.sort()  # subfolders are numerically ordered

    # create destination folders if needed
    for path in tempath, rejectpath:
        if not os.path.exists(path):
            os.mkdir(path)

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
                crop_img = crop(image, b)
                bg_img  = image.copy()
                bg_img[:] = 0
                xof = (bg_img.shape[1] - crop_img.shape[1])//2
                yof = (bg_img.shape[0] - crop_img.shape[0])//2
                overlay = overlay_image(bg_img, crop_img, x_offset=xof, y_offset=yof)
                cv2.imwrite(tempath+folder+'/'+fname, overlay)
                # b = [int(i) for i in b]

            # otherwise save original to reject/ for manual labelling
            elif type(b)==int:
                # Exit Signal
                if b == -1:
                    break
                # add folder if not there
                if folder not in os.listdir(rejectpath):
                    os.mkdir(rejectpath + folder)
                cv2.imwrite(rejectpath+folder+'/'+fname, image)
                b = np.array([0,0,0,0])


            # record label: [file-id, bounding_box]
            if not (type(b)==int and b == -1):
                interstage_ids.append(folder+'/'+fname)
                interstage_bbx.append(b)

        if type(b)== int and b == -1:
            break

        # NOTE: how is this going to screw up RetinaNet? It's predicting
        #       a bunch of classes, each with their own bounding boxes...
        #       Will I also have to change the common.detect function to account
        #       for there being only 1 output bounding box and NO 'classes' ?

    # write interstage labels CSV file
    # print(interstage_ids)
    # print(interstage_bbx)
    # print(type(interstage_bbx))
    # print(type(interstage_bbx[0]), interstage_bbx[0])

    df = bbx_to_DataFrame(interstage_ids, interstage_bbx)
    df.to_csv('data/interstage_labels.csv', index=False)

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
