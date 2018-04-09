# WNixalo - 2017-Dec-13 16:35
# script to correct csv filename ids to their correct names.

# 0     -> 000000
# 15    -> 000015

from pandas import DataFrame

# load file
fname = 'data/labels.csv'
with open(fname) as f:
    csvf = f.readlines()

# convert all ids
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


################################################################################
# NOTE: https://stackoverflow.com/questions/15653688/preserving-column-order-in-python-pandas-dataframe
# Holy Shit. That's quite a Bug.. okay.  I wanted to do it on one line, but I
# can just insert the ids column. Pandas orders the rows alphabetically by
# column. That explains the issue I encountered writing the datagetter yesterday.

# this should work in Pandas 0.11.1:
# # save csv file
# df = DataFrame({'id':ids, 'gloc':out})
# df.to_csv(fname, index=False)
