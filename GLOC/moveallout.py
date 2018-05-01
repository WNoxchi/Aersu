# Wayne Nixalo - 2017-Dec-21 23:45 | 2018-Apr-09 17:11
# Moves all files from subfolders to root data folder
# and delete empty subfolders

from glob import glob
from os import listdir, rename, rmdir

def main():
    PATH = 'data/'

    # get subfolder list
    subfolders = glob(PATH + '*/')

    # Safety Warning and Confirmation
    print(f'WARNING: this will delete ALL subfolders in the {PATH} directory \n'
          f'after moving their contents to {PATH}\n'
          f'Subfolders that would be deleted:')
    for s in subfolders:
        print(f'     {s.split("/")[1]}/')
    print(f'     --------')
    inp = ''
    while not inp.isalpha() or inp[0].lower() not in 'yn':
        inp = input(f'Confirm you want to do this: (y)es / (n)o:')
    inp = inp[0].lower()
    if inp == 'n':
        print('Exiting.')
        return

    # move all files out of their subfolders & del empty subfolders
    for i,s in enumerate(subfolders):
        fpaths = listdir(s)
        # pull out the filename and move it up
        for fp in fpaths:                           # each fp looks like this: '002553.jpg'
            rename(s + fp, PATH + fp)
            # print(f'{s + fp} --> {PATH + fp}')    # example: data/006440-006548/006506.jpg --> data/006506.jpg
        # delete empty subfolder
        rmdir(s)

if __name__ == "__main__":
    main()
