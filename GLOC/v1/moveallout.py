# Wayne Nixalo - 2017-Dec-21 23:45
# Moves all files from subfolders to root data folder

from glob import glob
from os import listdir, rename

PATH = 'data/'

# get subfolder list
subfolders = glob(PATH + '*/')

# get names of all files in subfolders


# move all files out of their subfolders
for i,s in enumerate(subfolders):
    fpaths = listdir(s)
    # pull out the filename and move it up
    for fp in fpaths:                           # each fp looks like this: '002553.jpg'
        rename(s + fp, PATH + fp)
        # print(f'{s + fp} --> {PATH + fp}')    # example: data/006440-006548/006506.jpg --> data/006506.jpg
