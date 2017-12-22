# Wayne Nixalo - 2017-Dec-21 22:26
# Move all files to their respective subfolders as defined by the numbered-csv
# label files.

# NOTE: assuming all files start in data/ folder

from glob import glob
from os import path, makedirs, rename

PATH = 'data/'

subfolders = []

# get numbered label csv filenames
labels = glob(PATH + 'labels_*.csv')

# create subdirectories by number-range
for label in labels:
    # pull out number range
    nr = label.split('_')[1].split('.')[0]
    # make subdirectory
    start,end = f'{nr.split("-")[0]}', f'{nr.split("-")[1]}'
    start,end = f'{int(start):0=6d}',  f'{int(end):0=6d}'
    dirname = start + '-' + end + '/'
    if not path.exists(PATH + dirname):
        makedirs(PATH + dirname)
        print(f'New subdir: {PATH+dirname}')
    else:
        print(f'Subdir exists: {PATH+dirname}')
    # add dirname to list
    subfolders.append(dirname)

### move all files to respective subfolders
# NOTE:https://stackoverflow.com/questions/8858008/how-to-move-a-file-in-python
fnames = glob(PATH + '*.jpg')

# build sorted list of all filenames (pathless)
for i,f in enumerate(fnames):
    fnames[i] = f.split('/')[-1]              # example: 'data/005016.jpg' --> '005016.jpg'
fnames.sort()

# sort subfolders -- this seems to work
subfolders.sort()

# move to subfolders
idx = 0
for i,f in enumerate(fnames):
    while f.split('.')[0] > subfolders[idx].split('-')[1].split('/')[0]:
        idx += 1
    rename(PATH + f, PATH + subfolders[idx] + f)
    # print(f"INDEX:{idx}, FILE:{f}, SUBFOLDER:{subfolders[idx]}")

# idx = -1
# print(fnames[-1].split('.')[0], subfolders[idx].split('-')[1].split('/')[0])
# print(fnames[-1].split('.')[0] > subfolders[idx].split('-')[1].split('/')[0])
