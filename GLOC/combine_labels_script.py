# WNixalo 2017-Dec-13 14:33
# NOTE: won't work right if unwanted .csv's exist in directory

import pandas as pd
import numpy as np
from glob import glob

# list all .csv's
g = glob('data/labels/*.csv')
# sort indices by starting image index
temp = sorted([[i,e.split('_')[1].split('-')[0]] for i,e in enumerate(g)], key = lambda x: int(x[1]))
# sort csv list by index
flist = [g[i[0]] for i in temp]
# initialize DataFrame as first csv
df = pd.read_csv(flist[0])
# append all other csv's to DataFrame
for i in range(1, len(flist)):
    df = df.append(pd.read_csv(flist[i]))
# save the new csv file
df.to_csv('data/labels.csv', index=False)
