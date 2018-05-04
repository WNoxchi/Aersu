import shutil
import os
import numpy as np
import time

def count_files(path, verbose=False):
    count = 0
    for folder in path.iterdir():
        count += len(list(folder.glob('*')))
        if verbose: print(f'{folder}: {len(list(folder.glob("*")))}')
    return count

def clear_cpu_dir(cpu_path, verbose=False):
    """
        Deletes the temprorary CPU directory. Exits quietly if directory doesn't exist.
    """
    if not os.path.exists(cpu_path): return
    if verbose: print('deleting: ', cpu_path)
    shutil.rmtree(cpu_path)
    if verbose: print('done')

def get_leaf(path):
    """returns the last element of a path"""
    return str(path).split('/')[-1]

def get_random_sample(df, p):
    """Returns a uniform random sample of dataset indices."""
    if p > 1: return df.sample(n=int(p))
    else: return df.sample(frac=p)

## NOTE: wait.. if I'm using CSVs.. why would I even need to play w/ folders?
##       I should rewrite this to also specify IDs so it can also create
##       matching bounding-box datasets.
def create_cpu_dataset(path, p=1000, subfolders='train', seed=0):
    """
        Creates a temporary sub-dataset for cpu-machine work.

        path : (pathlib.Path) root directory of dataset
        p    : (float, int) proportion for subset. If p <= 1: treated
                            as percetnage. If p > 1: treated as absolute
                            count and converetd to percentage.
        subfolders : (list(str), str) data subdirectories to copy.

        NOTE: currently the `shutil.copyfile` calls take quite
              a bit of time.
    """
    cpu_path = path/'cpu'
    if subfolders == None: subfolders = ['train','valid','test']
    if type(subfolders) == str: subfolders = [subfolders]
    np.random.seed(seed=seed)

    # delete & recreate cpu_path directory
    clear_cpu_dir(cpu_path, verbose=False)
    os.makedirs(cpu_path)

    for subfolder in subfolders:
        # if p absolute: calculate percentage
        if p > 1:
            count = count_files(path/subfolder)
            t = p
            p = min(1.0, p/count)
            count *= p
        else:
            count = p * count_files(path/subfolder)
        # copy files to cpu directory
        os.makedirs(cpu_path/subfolder)
        for clas in os.listdir(path/subfolder):
            if clas == '.DS_Store': continue # MacOS annoyance
            os.makedirs(cpu_path/subfolder/clas)
            flist = list((path/subfolder/clas).iterdir())
            n_copy = int(np.round(len(flist) * p))
            flist = np.random.choice(flist, n_copy, replace=False)
            for f in flist:
                fname = get_leaf(f)
                shutil.copyfile(f, cpu_path/subfolder/clas/fname)
                count -= 1
        # cap off total copied
        while count > 0:
            for clas in os.listdir(path/subfolder):
                if count == 0: break
                flist = list((path/subfolder/clas).iterdir())
                f = np.random.choice(flist)
                while get_leaf(f) in os.listdir(cpu_path/subfolder/clas):
                    f = np.random.choice(flist)
                fname = get_leaf(f)
                shutil.copyfile(f, cpu_path/subfolder/clas/fname)
                count -= 1

def create_cpu_labels(cpu_path, cpu_label_path='cpu_labels', subdir='train'):
    """
        Creates labels from cpu data subset by looking at its `subdir` directory.
            (default: 'train')
    """
##### ensure cpu label filepath correct
    if '.csv' not in cpu_label_path: cpu_label_path += '.csv'
    assert cpu_label_path[-4:] == '.csv'
##### get idxs of cpu dataset files (int(id) = idx in GLoC dataset)
    keep_ids = []
    cpu_data_path = cpu_path /  subdir
    subfolders = os.listdir(cpu_data_path)
    for undes in ['.DS_Store', cpu_label_path.split('/')[-1]]: # remove old label file & .DS_Store
        if undes in subfolders: subfolders.remove(undes)
    subfolders.sort()
    for subfolder in subfolders:
        fnames = os.listdir(cpu_data_path/subfolder)
        fnames.sort()
        for fname in fnames:
            keep_ids.append(subfolder + '/' + fname.split('.')[0])
##### get ids of cpu dataset files
    keep_idxs = [int(i.split('/')[-1]) for i in keep_ids]
    df_cpu = df.loc[keep_idxs]
##### write csv & return df
    df_cpu.to_csv(cpu_path / cpu_label_path, index=False)
    return df_cpu
