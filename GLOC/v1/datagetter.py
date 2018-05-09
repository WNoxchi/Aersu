# 2017-Dec-12 15:39 Wayne H Nixalo
# G-Lock Data getter
################################################################################
import numpy as np
import cv2
from utils.getkey import getKey
from utils.getscreen import getScreen
from time import time
from time import sleep
from pandas import DataFrame
from os.path import exists as os_path_exists
from os import mkdir as os_mkdir
from glob import glob


def save_screen_box(bbox = (8,160,682,544), idx=0, path='data/'):
    # MACOS Retina doubles resolution. May need halve on Linux/Win
    h = (bbox[3] - bbox[1]) * 2 # 768
    w = (bbox[2] - bbox[0]) * 2 # 1348

    # x & y transforms
    tfx = 400 / w
    tfy = 400 / h

    # capture, resize, save screen
    img = getScreen(bbox=bbox)
    img = np.asarray(img)
    img = cv2.resize(img, None, fx=tfx, fy=tfy)
    cv2.imwrite(path+f'{idx:0=6d}.jpg', img)

def find_idx(dpath=None):
    if not dpath:
        dpath = 'data/'
    if not os_path_exists(dpath):
        os_mkdir(dpath)
        # starting from zero index
        return 0

    # path/prefix length
    pl = len(dpath)
    # suffix/extension length
    sl = len('.jpg')

    # get next index
    idx = 1 + int(max(glob(dpath+'*.jpg'))[pl:-sl]) if glob(dpath+'*.jpg') else 0

    return idx

def main():
    # find starting index and find/create data folder
    dpath = 'data/'
    idx = find_idx()
    idx_last = max(-1, idx - 1)

    # record data
    print("[SPACE] for G-Lock, [Any Other Key] otherwise, [RETURN] to quit.")
    for i in list(range(4))[::-1]:
        print(f'Beginning Capture in: {i+1}')
        sleep(1)

    key = None
    output = []
    ids = []

    while key != '\n':
        key = getKey()
        output.append(key)
        save_screen_box(idx=idx, path=dpath)
        ids.append(idx)
        idx += 1

    # convert output to a one-hot array
    output = [[0,1][i==' '] for i in output]

    out_df = DataFrame({'id':ids, 'lock':output})
    out_df.to_csv(dpath+f'labels_{idx_last+1}-{idx-1}.csv', index=False)

    print(f'{idx-1 - idx_last} data points recorded to: labels_{idx_last+1}-{idx-1}.csv')

if __name__ == "__main__":
    main()



################################################################################
# BENCHMARKS

# MSS screengrab saved by OpenCV (100 runs avg):
# (JPG):    0.03196930885314941 s       (400x400):  0.005743522644042969 s
# (PNG):    0.05808886766433716 s       (400x400):  0.011203761100769044 s

# MSS screengrab (1000 runs avg):
#           0.018143182039260863 s      (400x400):

# MSS img to NumPy array conversion:
#           4.024839401245117e-05 s     (400x400):

# OpenCV resize 768x1348 -> 400x400:
#           0.0016115713119506836 s
#           0.0048240418434143065 s  [interpolation=INTER_AREA]

# So my priority is to minimize the size of what I'm saving. Hell most of the
# models I use are trained in the 244 square-pixel range anyway.

# I should be getting good FPS saving 400x400 boxes, but I have to resize them
# with OpenCV I think.
