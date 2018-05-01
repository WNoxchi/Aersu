# Takes data generated by datagetter.py and automatically formats into proper
# subfolders and CSV file.
################################################################################
# Wayne H Nixalo
# 2018-Apr-09 13:03-15:04|16:57-18:04
################################################################################
from pandas import DataFrame, read_csv
from glob import glob
from os import path, makedirs, rename
# import pandas as pd
import numpy as np
import os

def subfolderize(PATH='data/', ftype='.jpg', verbose=1):
    """
        Create subfolders and move images; based on CSV filenames.
    """
    subfolders = []
    # get numbered label csv filenames
    labels = glob(PATH + 'labels_*.csv')

    # create subdirectories by number-range
    for label in labels:
        nr = label.split('_')[1].split('.')[0]
        start,end = f'{nr.split("-")[0]}', f'{nr.split("-")[1]}'
        start,end = f'{int(start):0=6d}',  f'{int(end):0=6d}'
        dirname = start + '-' + end + '/'
        if not path.exists(PATH + dirname):
            makedirs(PATH + dirname)
            if verbose: print(f'New subdir: {PATH+dirname}')
        elif verbose:
            print(f'Subdir exists: {PATH+dirname}')
        # add dirname to list
        subfolders.append(dirname)

    # build sorted list of all filenames (pathless) NOTE:https://stackoverflow.com/questions/8858008/how-to-move-a-file-in-python
    fnames = glob(PATH + '*' + ftype)
    for i,f in enumerate(fnames):
        fnames[i] = f.split('/')[-1] # example: 'data/005016.jpg' --> '005016.jpg'
    fnames.sort()
    subfolders.sort() # this seems to work

    # move to subfolders
    idx = 0
    for i,f in enumerate(fnames):
        while f.split('.')[0] > subfolders[idx].split('-')[1].split('/')[0]:
            idx += 1
        rename(PATH + f, PATH + subfolders[idx] + f)

def combine_labels_script(csv_file_prefix='labels', verbose=1):
    """
        Combine CSV files into 1 master CSV file
    """
    # list all .csv's & sort indices by starting image index
    g = glob(f'data/{csv_file_prefix}_*.csv')
    idxs_srtd = sorted([[i,e.split('_')[1].split('-')[0]] for i,e in enumerate(g)], key = lambda x: int(x[1]))
    # sort csv list by index
    flist = [g[i[0]] for i in idxs_srtd]
    # initialize DataFrame as first csv
    df = read_csv(flist[0])
    # append all other csv's to DataFrame and save
    for i in range(1, len(flist)):
        df = df.append(read_csv(flist[i]))
    df.to_csv(f'data/{csv_file_prefix}.csv', index=False)
    if verbose: print(f'Wrote: data/{csv_file_prefix}.csv')

    # move original CSV files to {csv_file_prefix} folder
    makedirs(f'data/{csv_file_prefix}', exist_ok=True)
    for csvf in flist:
        rename(f'{csvf}', f'data/{csv_file_prefix}/{csvf.split("/")[-1]}')


def name_corrector():
    """
        Correct Master CSV IDs (“0-21” —> “000000-000021”)
    """
    # load file
    fname = 'data/labels.csv'
    with open(fname) as f:
        csvf = f.readlines()

    # convert all ids, skip header
    for i in range(1, len(csvf)):
        csvf[i] = f'{int(csvf[i].split(",")[0]):0=6d}' + csvf[i][len(csvf[i].split(',')[0]):]

    # prepare arrays for DataFrame
    ids = [i.split(',')[0] for i in csvf[1:]]
    out = [int(i.split(',')[1]) for i in csvf[1:]]

    # save csv file
    df = DataFrame(out)
    df.columns = ['gloc']
    df.insert(0, 'id', ids)
    df.to_csv(fname, index=False)

def directorize_fnames():
    """
        Update IDs in Master CSV file to match subfolder directories.
        Example: XXXXXX —> STARTIDX-ENDIDX/XXXXXX
    """
    # get subdirectory names
    # if train/ directory exists, use that, else data/
    dirpath = 'data/train/' if path.exists('data/train/') else 'data/'
    dirnames = glob(dirpath + '*/')
    for i,d in enumerate(dirnames):
        dirnames[i] = d.split('/')[-2]                      # example: 'data/train/000000-003707/' --> '000000-003707'
    dirnames.sort()

    # load DataFrame
    df = read_csv('data/labels.csv')
    key = df.columns[0]
    idx = 0
    newcol = []

    # convert file ids to subdirectorized-format
    for i,e in enumerate(df[key]):
        if int(e) > int(dirnames[idx].split('-')[1]):       # example: ['data', 'train', '042599-044710', ''] --> 044710
            idx += 1
        newcol.append(dirnames[idx] + '/' + f'{e:0=6d}')
    newdf = DataFrame(newcol, columns=['id'])
    newdf.insert(1, 'gloc', df['gloc'])

    # save new DataFrame
    newdf.to_csv('data/labels.csv', index=False)


def main():
    print(f'Creating subfolders and moving data files.')
    subfolderize()
    print(f'Merging label files into master labels CSV file.')
    combine_labels_script()
    print(f'Correcting Master CSV label IDs.')
    name_corrector()
    print(f'Directorizing Master CSV label IDs.')
    directorize_fnames()

if __name__ == "__main__":
    main()
