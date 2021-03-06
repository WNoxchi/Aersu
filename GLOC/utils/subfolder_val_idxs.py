# 2018-May-01 14:35

################################################################################
# Originally subfolder_val_loader.py in v1:
    # Wayne Nixalo - 2017-Dec-17 23:07 - 2017-Dec-18 01:49
    # NOTE: this function makes some very specific assumptions I've not documented yet!
################################################################################

import numpy as np
from pandas import read_csv

def set_val_idxs(label_csv, val_pct=0.2, thresh=0.05, seed=None):
    """
        Returns indices of validation set split by subfolder.
        Assumes 1st column of label_csv is id. Assumes id is in format:
            subfolder/start_idx-end_idx.sfx
        label_csv: path to labels csv file.
        val_pct  : validation percentage
        thresh   : threshold
        seed     : PRNG seed
    """
    # set random seed if specfd
    if seed != None: np.random.seed(seed)

    # safety check
    assert val_pct < 1, "Validation Percent must be below 100: %f" % val_pct

    # get labels csv
    df = read_csv(label_csv)
    labels = df[df.columns[0]].as_matrix()

    # build subfolders array
    subfolders = np.unique([λ.split('/')[0] for λ in labels])

    # get total number of data
    total = len(labels)

    # get array of total data each subfolder
    counts = []
    for folder in subfolders:
        a,b = list(map(lambda x: int(x), folder.split('-')))
        counts.append(b-a+1)

    # upper and lower bounds
    target_max = total * (val_pct+thresh)
    target_min = total * (val_pct-thresh)
    val_total  = 0
    idx        = -1

    # try permutations until above min threshold
    # NOTE: assertion above aimed at making this safe. Also max tries.
    tries = 0
    while val_total < target_min:
        tries += 1
        if tries >= 50: raise Error('Unable to fit validation set by subfolder into specifed bounds. Specify a larger threshold, different validation percentage, or ensure dataset\'s subfolders can provide the split you want.')

        # shuffle indices
        valfolder_idxs = np.random.permutation(len(subfolders)-1)

        while val_total < target_max:
            idx += 1
            val_total += counts[valfolder_idxs[idx]]
        # just over threshold: lose last idx
        if val_total > target_max:
            val_total -= counts[valfolder_idxs[idx]]
            idx -= 1

    # validation subfolders & indices
    valfolder_idxs = valfolder_idxs[:idx+1]
    valfolders = subfolders[valfolder_idxs]

    # build subtraction index to match validation-index format of FastAI dataloader
    subtract_idx = np.zeros(len(subfolders))
    for idx, folder in enumerate(subfolders):
        if idx == 0:
            # start with beginning
            subtract_idx[idx] = int(subfolders[idx].split('-')[0])
        else:
            # add the previous + the difference
            subtract_idx[idx] = subtract_idx[idx-1] + \
                                    int(subfolders[idx].split('-')[0]) - \
                                    int(subfolders[idx-1].split('-')[1]) - 1

    # build final validation index list
    val_idxs = []
    for idx, vdx in enumerate(valfolder_idxs):
        start, end = list(map(lambda x: int(x), valfolders[idx].split('-')))
        start -= subtract_idx[vdx]
        end   -= subtract_idx[vdx]
        val_idxs.extend(np.arange(start, end+1).astype(np.int32))   # otherwise will be Floats

    return val_idxs
