# Common Utilities
# Wayne H Nixalo - GLoC v2 Refactor
# 2018-Apr-09 12:46

import cv2
import numpy as np
import time
import pandas as pd
import os
from fastai.dataset import ImageClassifierData

def c_shift(c=[255,0,0], n=1, shifts=1, val=255, quiet=True):
    """
    Creates equal-sized shifts in color, where size is val*3/n, starting with
    pure-Red and ending with pure-Blue (assuming RGB).
    """
    f = val*3/n # 255*3/5 = 153
    shifts = min(n, shifts)
    c = list(c)

    for i in range(shifts):
        if not quiet:  print(f'print: {c}')
        if i == n - 1: continue

        # find first nonzero
        idx = next(i for i,x in enumerate(c) if x > 0)

        move = 0

        if c[idx] >= f:
            move += f
            c[idx] -= f
        else:
            move += c[idx]
            c[idx] = 0
        if move == f:
            c[idx+1] += f
        else:
            c[idx+1] += move
            move = f -  move
            if idx+2 <= len(c)-1:
                c[idx+2] += move
                c[idx+1] -= move

    # round all to ints
    c = [round(x) for x in c]

    return tuple(c) # changed to tuple for OpenCV

# Wayne H Nixalo -- 2018-Jan-03 13:00 - 14:00
# image cropper
def crop(image, bbox):
    """
    bbox:[x1,y1,x2,y2]
    roi: [y1:y2,x1:x2]
    image: (ndarray)
    """
    p1,p2 = (bbox[0],bbox[1]), (bbox[2],bbox[3])
    crop = image.copy()
    return crop[p1[1]:p2[1], p1[0]:p2[0]]

# 2018-Jan-07 21:46 | 2018-Apr-09 12:52
# overlay: http://stackoverflow.com/a/14102014/627517
# via: https://sourcegraph.com/github.com/lexfridman/boring-detector@master/-/blob/boring_common.py#L60
def overlay_image(λ_img, s_img, x_offset=0, y_offset=0):
    """
    Overlays 'small' foreground image `s_img` atop 'large' background
        image `λ_img` adjusted by x_- & y_offset.

    Returns: overlayed image (ndarray)
    """
    assert y_offset + s_img.shape[0] <= λ_img.shape[0]
    assert x_offset + s_img.shape[1] <= λ_img.shape[1]

    λ_img = λ_img.copy()

    # if an alpha-channel exists in fg-image: remove it
    if s_img.shape[-1] == 4:
        for c in range(0, 3):
            λ_img[y_offset:y_offset+s_img.shape[0],
                  x_offset:x_offset+s_img.shape[1], c] = (
                                s_img[:,:,c] * (s_img[:,:,3]/255.) +
                                λ_img[y_offset:y_offset+s_img.shape[0],
                                      x_offset:x_offset+s_img.shape[1], c] *
                                (1. - s_img[:,:,3]/255.))
    else:
        # no alpha-channel:
        λ_img[y_offset:y_offset+s_img.shape[0],
              x_offset:x_offset+s_img.shape[1]] = s_img

    return λ_img

def load_dummy(fpath=None):
    """
    Returns a single image & label from the dataset, for use as a dummy
    training set to initalize the Fast.ai array dataloader.
    """
    assert type(fpath) == str
    train_dat = cv2.imread(fpath)
    train_dat = cv2.cvtColor(train_dat, cv2.COLOR_BGR2RGB)
    return np.array([train_dat]), np.array([1])

# function to update FastAI dataloader with screengrab
def load_test_image(PATH, image=None, train_dat=None, valid_dat=None, classes=[0,1], tfms=None):
    """
    Returns a FastAI dataloader with a single image as its test set.
    """
    test_dat = np.array([image]) if type(image) == np.ndarray else None
    return ImageClassifierData.from_arrays(PATH, train_dat, valid_dat, bs=1,
                                           tfms=tfms, classes=classes, test=test_dat)

# Wayne Nixalo - 2018-Jan-19 15:19
def csv_stitcher(path='data/', csv_name=''):
    """
    Concatenates all numbered CSVs together - in order - into one CSV file.
    """
    # gather list of CSVs
    csvs = [fname for fname in os.listdir(path) if csv_name in fname and '.csv' in fname]
    csvs.sort()
    # init out DataFrame with 1st CSV
    out_df = pd.read_csv(path + csvs[0])
    # append all other CSVS to end
    for i in range(1, len(csvs)):
        out_df = out_df.append(pd.read_csv(path + csvs[i]))
    return out_df

#













#
