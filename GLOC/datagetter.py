# G-Lock Data getter
# Wayne H Nixalo
# 2017-Dec-12 15:39 | 2018-Apr-09 13:15

# saves images to a single data directory in format: 0000XX.jpg, 0000XY.jpg, etc
# saves CSV label file to data directory in format: labels_X-Z.csv
################################################################################
import numpy as np
import cv2
from utils.getkey import getKey
from utils.getscreen import getScreen
from time import sleep
from pandas import DataFrame
from os.path import exists as os_path_exists
from os import mkdir as os_mkdir
from glob import glob

# default image square size NOTE: havent decided on making/not imgs square yet
sz = 400

def save_screen_box(bbox = (8,160,682,544), idx=0, path='data/', ftype='.jpg'):
    """
        Saves a screenshot at coordinates {bbox} to {path} with name {idx}
        of type {ftype}. Name is zero-padded to 6 digits.
    """
    # MACOS Retina doubles resolution. May need halve on Linux/Win
    # h = (bbox[3] - bbox[1]) * 2 # 768
    # w = (bbox[2] - bbox[0]) * 2 # 1348

    # x & y transforms
    # tfx = sz / w
    # tfy = sz / h

    # capture, resize, save screen
    img = getScreen(bbox=bbox)
    img = np.asarray(img)
    # img = cv2.resize(img, None, fx=tfx, fy=tfy) # resize via transform
    cv2.resize(img, (sz,sz)) # resize via desired output size
    cv2.imwrite(path+f'{idx:0=6d}{ftype}', img)

def find_idx(dpath=None, ftype='.jpg'):
    """
       Scans data directory for last file number and returns next index.
       Creates data directory and returns index 0 if none exists.
    """
    if not dpath: dpath = 'data/'
    if not os_path_exists(dpath):
        os_mkdir(dpath)
        # if new directory start new index
        return 0
    # path/prefix length
    pl = len(dpath)
    # suffix/extension length
    sl = len(ftype)
    # get next index
    idx = 1 + int(max(glob(dpath+'*.jpg'))[pl:-sl]) if glob(dpath+'*'+ftype) else 0

    return idx

def main():
    # find starting index and find/create data folder
    idx = find_idx()
    # idx_last = max(0, idx - 1)
    idx_last = idx - 1
    dpath='data/'

    # record data
    print("Starting with index: {idx:0=6d}")
    print("[SPACE] for G-LoC, [Any Other Key] otherwise, [RETURN] to quit.")
    for i in list(range(4))[::-1]:
        print(f'Beginning Capture in: {i+1}')
        sleep(1)

    key = None
    output = []
    ids = []

    while key != '\n':
        key = getKey()
        if key == '\n':
            break
        output.append(key)
        save_screen_box(idx=idx, path=dpath)
        ids.append(idx)
        idx += 1

    # write CSV if anything recorded
    if len(output) > 0:
        # convert output to a one-hot array
        output = [[0,1][i==' '] for i in output]

        # create CSV of file IDs and labels
        out_df = DataFrame({'id':ids, 'lock':output})
        out_df.to_csv(dpath+f'labels_{idx_last+1}-{idx-1}.csv', index=False)

        print(f'{idx-1 - idx_last} data points recorded to: labels_{idx_last+1}-{idx-1}.csv')
    else:
        print("Nothing recorded. Quit signal entered. Quiting.")

if __name__ == "__main__":
    main()
