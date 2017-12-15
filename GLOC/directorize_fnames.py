# Wayne Nixalo - 2017-Dec-14 14:57 | 21:37
# This script converts all label names from XXXXXX to STARTIDX-ENDIDX/XXXXXX
#
# This is part of an attempt to troubleshoot a memory-usage error in Jupyter
# by moving the dataset into subfolders. This script makes this compatible
# with FastAI's `.from_csv()` dataloader. Hopefully.

import pandas as pd
from glob import glob

# get subdirectory names
dirnames = glob('data/train/*/')
for i,d in enumerate(dirnames):
    dirnames[i] = d.split('/')[-2]                      # example: 'data/train/000000-003707/' --> '000000-003707'
dirnames.sort()

# load DataFrame
df = pd.read_csv('data/labels.csv')
key = df.columns[0]
idx = 0
newcol = []

# convert file ids to subdirectorized-format
for i,e in enumerate(df[key]):
    if int(e) > int(dirnames[idx].split('-')[1]):       # example: ['data', 'train', '042599-044710', ''] --> 044710
        idx += 1
    newcol.append(dirnames[idx] + '/' + f'{e:0=6d}')

newdf = pd.DataFrame(newcol, columns=['id'])
newdf.insert(1, 'gloc', df['gloc'])

# save new DataFrame
newdf.to_csv('data/labels.csv', index=False)
