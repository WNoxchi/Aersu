# Wayne Nixalo - 2018-Jan-02 23:15 / 2018-Jan-03 12:59 / 2018-Jan-14 23:45
#                2018-Jan-15 10:44

### IMPORTS
import keras
import keras.preprocessing.image

import cv2
import os
import numpy as np
import time
from pandas import DataFrame
import pandas as pd

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
    # rejectpath = tempath + 'reject/'
    folders = os.listdir(tpath)
    folders.sort()  # subfolders are numerically ordered
    if '.DS_Store' in folders:
        folders.remove('.DS_Store')

    # create destination folder if needed
    if not os.path.exists(tempath):
        os.mkdir(tempath)
        clean_start = True
        last_fname = -1
    else:
        # find starting point if quit before
        #NOTE: requires deletion of CSVs if tmp/ data deleted! otherwise will skip
        clean_start = False
        interstage_csvs = [csv_fname for csv_fname in os.listdir('data/') if 'interstage_labels-' in csv_fname]
        interstage_csvs.sort()
        last_csv = pd.read_csv('data/' + max(interstage_csvs))
        # find last recorded filename
        last_fpath = last_csv['id'].iloc[-1]
        last_folder, last_fname = last_fpath.split('/')
        # remove all folders before last
        for idx,folder in enumerate(folders):
            if folder < last_folder:
                folders.remove(folder)

    for folder in folders:
        # get all filenames
        fnames = os.listdir(tpath + folder)
        fnames.sort()
        # remove all fnames before last in 1st folder if not starting fresh
        removals = []
        if not clean_start and folder == last_folder:
            for idx,fname in enumerate(fnames):
                if fname <= last_fname:
                    removals.append(fname)
        for rem in removals:
            fnames.remove(rem)

        # run detection on each file
        for fname in fnames:
            print(f'Displaying: {fname}')

            fpath = tpath+folder+'/'+fname
            # image = Image.open(fpath)
            image = cv2.imread(fpath)
            image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
            b = detect(image=image, model=model, mode='ss', fname=fname, quiet=False)

            # crop & save to tmp/ if bounding box
            if type(b)==np.ndarray or type(b)==list:
                # add folder if not there
                if folder not in os.listdir(tempath):
                    os.mkdir(tempath + folder)
                crop_img = crop(image, b)
                crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                bg_img  = image.copy()
                bg_img[:] = 0
                xof = (bg_img.shape[1] - crop_img.shape[1])//2
                yof = (bg_img.shape[0] - crop_img.shape[0])//2
                overlay = overlay_image(bg_img, crop_img, x_offset=xof, y_offset=yof)
                cv2.imwrite(tempath+folder+'/'+fname, overlay)

                # record label: [file-id, bounding_box]
                interstage_ids.append(folder+'/'+fname)
                interstage_bbx.append(b)   # list or ndarray is fine for transposing below

            elif type(b)==int and b == -1:
                # Exit Signal
                break

            else:
                # This should NEVER happen
                print('Woah. Something went wrong.\n`Bounding Box`: {b}')

            # only update end fname if all went smoothly
            new_last_fname = fname

        if type(b)== int and b == -1:
            # cascading Exit Signal
            break

        # NOTE: how is this going to screw up RetinaNet? It's predicting
        #       a bunch of classes, each with their own bounding boxes...
        #       Will I also have to change the common.detect function to account
        #       for there being only 1 output bounding box and NO 'classes' ?

    # write interstage labels CSV file
    if len(interstage_ids) > 0:
        start = int(last_fname.split('.')[0])+1 if not clean_start else 0
        end   = int(new_last_fname.split('.')[0])
        df = bbx_to_DataFrame(interstage_ids, interstage_bbx)
        df.to_csv(f'data/interstage_labels-{start:0=6d}-{end:0=6d}.csv', index=False)

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
