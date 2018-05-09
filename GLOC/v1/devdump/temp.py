# 2018-Jan-15 00:55
import os
import pandas as pd
interstage_csvs = [cf for cf in os.listdir('data/') if 'interstage_labels-' in cf]
interstage_csvs.sort()
last_csv = pd.read_csv('data/' + max(interstage_csvs))
last_fpath = last_csv['id'][-1]

last_csv['id'].iloc[-1]



# 2018-Jan-14 22:25
# more playing with arrays

array = [[i+j*10 for i in range(4)] for j in range(10)]
array
# [[0, 1, 2, 3],
#  [10, 11, 12, 13],
#  [20, 21, 22, 23],
#  [30, 31, 32, 33],
#  [40, 41, 42, 43],
#  [50, 51, 52, 53],
#  [60, 61, 62, 63],
#  [70, 71, 72, 73],
#  [80, 81, 82, 83],
#  [90, 91, 92, 93]]

array[5] = np.array(array[5])
array
array[0]

# Transpose
arrayT = [[r[cdx] for r in array] for cdx in range(len(array[0]))]
arrayT
# [[0, 10, 20, 30, 40, 50, 60, 70, 80, 90],
#  [1, 11, 21, 31, 41, 51, 61, 71, 81, 91],
#  [2, 12, 22, 32, 42, 52, 62, 72, 82, 92],
#  [3, 13, 23, 33, 43, 53, 63, 73, 83, 93]]

import numpy as np
np.array(arrayT)[1,:]
# array([ 1, 11, 21, 31, 41, 51, 61, 71, 81, 91])













# 2018-Jan-14 21:59
# learning about Pandas and datatypes. It looks like you can't save a column
# with array elements and expect to read them back as ints or floats: they come
# back as strings.

import pandas as pd

df = pd.read_csv('data/interstage_labels.csv')

df['bounding box']

x = df['bounding box'][0]

x

type(x)

df['bounding box'][0].values

x = [[i for i in range(5)]]
df = pd.DataFrame({'x':x})
df.values
df


df = pd.read_csv('data/interstage_labels.csv')['bounding box']

np.asarray(df[0])
np.array(df[0])


# x = [[i for i in range(5)], 0]
x = [i for i in range(5)]
x.append([1,2,3])
df = pd.DataFrame({'x':x})
df.to_csv('tempcsv.csv', index=False)

df = pd.read_csv('tempcsv.csv')

for i in df['x']:
    print(i, type(i))


pd.DataFrame({('a', 'b'): {('A', 'B'): 1, ('A', 'C'): 2},
               ('a', 'a'): {('A', 'C'): 3, ('A', 'B'): 4},
               ('a', 'c'): {('A', 'B'): 5, ('A', 'C'): 6},
               ('b', 'a'): {('A', 'C'): 7, ('A', 'B'): 8},
               ('b', 'b'): {('A', 'D'): 9, ('A', 'B'): 10}}).values


x1,y1,x2,y2 =

x1,y1,x2,y2 = [[i+j*10 for i in range(10)] for j in range(4)]
x1
y1

# df = pd.DataFrame({'x1':x, 'y1':y, 'x2':w, 'y2':z})


arr = np.array([[i+j*10 for i in range(10)] for j in range(4)])
arr

arr[1,:]


df = pd.DataFrame(data=x1,columns=['x1'])
df.insert(1,'y1',y1)
df.insert(2,'x2',x2)
df.insert(3,'y2',y2)
df
df.to_csv('tempcsv.csv', index=False)
df = pd.read_csv('tempcsv.csv')
df.values
df['x1'][0].dtype
int(df['x1'][0])


np.int64 == np.dtype('int64')
int == np.dtype('int64')




a = np.arange(10).reshape(2,5)
b = np.arange(10, 20).reshape(2,5)
df = pd.DataFrame({'foo':[42,51], 'arr':[a,b]})

df.to_csv('tempcsv.csv',index=False)
df=pd.read_csv('tempcsv.csv')

df.values


df['foo']
df['arr']


df.columns



Out[10]:
                                            arr  foo
0            [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]   42
1  [[10, 11, 12, 13, 14], [15, 16, 17, 18, 19]]   51










df = pd.read_csv('data/labels.csv')

df['gloc'][0].dtype


import numpy as np
print(np.array([i for i in range(5)]))




# # screen grab utility
# from utils.getscreen import getScreen
#
# # FastAI imports
# from fastai_osx.model import resnet34
# from fastai_osx.conv_learner import *
#
# # Load dummy training set to init dataloader
# dummy_fpath = 'data/train/000000-000412/000000.jpg'
# train_dat = cv2.imread(dummy_fpath)
# train_dat = cv2.cvtColor(train_dat, cv2.COLOR_BGR2RGB)
# train_dat = np.array([train_dat]), np.array([0])
# valid_dat  = train_dat
# classes   = [0,1]
#
# # Darwin Retina system:
# bbox = (8,160,682,544)
# h = (bbox[3] - bbox[1]) * 2 # 768
# w = (bbox[2] - bbox[0]) * 2 # 1348
# tfx = 400 / w
# tfy = 400 / h
#
# img = getScreen(bbox=bbox)
# img = np.asarray(img)
# img = cv2.resize(img, None, fx=tfx, fy=tfy)
# img2= img[:,:,:3]
#
#
# # in_img = cv2.resize(np.asarray(getScreen(bbox=bbox)), None, fx=tfx, fy=tfy)
#
# cv2.imshow('1', img)
# cv2.imshow('2', img2)
# cv2.waitKey(1)
# time.sleep(5)
# cv2.destroyAllWindows()
#
# print(img.shape)

# print(type(in_img))


































#
# import cv2
# import os
# import matplotlib.pyplot as plt
# os.getcwd()
# if os.getcwd() != '/Users/WayNoxchi/Aersu/GLOC':
#     os.chdir(os.getcwd() + '/GLOC')
# os.getcwd()
#
# import time
#
#
# image = cv2.imread('blah.jpg')
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
# # overlay: http://stackoverflow.com/a/14102014/627517
# # via: https://sourcegraph.com/github.com/lexfridman/boring-detector@master/-/blob/boring_common.py#L60
# def overlay_image(λ_img, s_img, x_offset=0, y_offset=0):
#     assert y_offset + s_img.shape[0] <= λ_img.shape[0]
#     assert x_offset + s_img.shape[1] <= λ_img.shape[1]
#
#     # print(y_offset, x_offset)
#
#     λ_img = λ_img.copy()
#     # print(type(λ_img))
#
#     # if an alpha-channel exists in fg-image: remove it
#     if s_img.shape[-1] == 4:
#         for c in range(0, 3):
#             λ_img[y_offset:y_offset+s_img.shape[0],
#                   x_offset:x_offset+s_img.shape[1], c] = (
#                                 s_img[:,:,c] * (s_img[:,:,3]/255.) +
#                                 λ_img[y_offset:y_offset+s_img.shape[0],
#                                       x_offset:x_offset+s_img.shape[1], c] *
#                                 (1. - s_img[:,:,3]/255.))
#     else:
#         # no alpha-channel:
#         λ_img[y_offset:y_offset+s_img.shape[0],
#               x_offset:x_offset+s_img.shape[1]] = s_img
#
#     return λ_img
#
# image.shape[2] == 3
#
#
# plt.imshow(image[:,:,3])
#
# plt.imshow(image[:,:,3])
#
#
# aerzh_img = image.copy()
# aerzh_img[:] = 0
# plt.imshow(aerzh_img)
#
# image.shape
# image[:100,:,:].shape
#
# %timeit -o time.sleep(0.2)
#
#
# in_img = image[20:120,:,:]
# overlayed_img = overlay_image(aerzh_img, in_img,
#                     x_offset = (aerzh_img.shape[1] - in_img.shape[1])//2,
#                     y_offset = (aerzh_img.shape[0] - in_img.shape[0])//2)
# plt.imshow(overlayed_img)
#
#
# # okay so the images are HEIGHT,WIDTH (ROW,COLOMN) order. Right, duh. NumPy arrays.
# plt.imshow(image[:100,:,:])
#
#
# plt.imshow(image)
#
#
# # TESTING FULL OVERLAY FUNCTION .. THING
#
# # NOTE:
# # bounding box from neural net:   [x1,y1,x2,y2]
# # OpenCV region of interest:      [y1:y2,x1:x2]
#
#
# fname = 'ricedude.jpg'
# plt.imshow(cv2.imread('ricedude.jpg'))
#
# img = cv2.imread('ricedude.jpg') # GrayScale img; no BGR->RGB convrsn nec. (I think)
#
# # pulled from utils.common.py
# def crop(image, bbox):
#     """
#     bbox:[x1,y1,x2,y2]
#     roi: [y1:y2,x1:x2]
#     image: (ndarray)
#     """
#     p1,p2 = (bbox[0],bbox[1]), (bbox[2],bbox[3])
#     crop = image.copy()
#     return crop[p1[1]:p2[1], p1[0]:p2[0]]
#
# bbox = [200,200,450,400]
#
# img_crop = crop(img, bbox)
#
# x_off = (img.shape[1] - img_crop.shape[1])//2
# y_off = (img.shape[0] - img_crop.shape[0])//2
# overlay_img = overlay_image(img, img_crop, x_offset=x_off, y_offset=y_off)
#
# plt.imshow(overlay_img)
#
# import numpy as np
#
# img = cv2.imread('toolate.jpg')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img = cv2.resize(img, None, fx=0.5,fy= 0.5)
# plt.imshow(img)
#
# overlay_img = overlay_image(img, img_crop)
# plt.imshow(overlay_img)
#
#
#
#
#
#
#
# %matplotlib inline
# %reload_ext autoreload
# %autoreload 2
#
# import os
# os.getcwd()
# os.chdir(os.getcwd() + '/GLOC/')
# os.getcwd()
#
#
# from fastai_osx.model import resnet34
# from fastai_osx.conv_learner import *
#
#
# PATH = 'data/'
#
# def get_image_ndarray(path='data/train/'):
#     """Returns a random image as an ndarray in an ndarray from the GLOC Dataset"""
#     # get random image
#     if '.DS_Store' in os.listdir(path):
#         os.remove(path + '.DS_Store')
#     folders = os.listdir(path)
#     folder  = np.random.choice(folders)
#     fname   = np.random.choice(os.listdir(path+folder))
#     fpath   = path+folder+'/'+fname; fpath
#
#     # load image as ndarray
#     image = cv2.imread(fpath) # dtype = 'uint8'
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # BGR -> RGB conversion
#
#     # reshape image to PyTorch Tensor Order: (Channel,H,W)
#     # image = np.rollaxis(image, 2, 0)
#
#     # return image as ndarray of ndarrays, and image filepath
#     return np.array([image]), fpath
#
#
# # get random test image and dummy train/val image & label
# img_ndarray, fpath = get_image_ndarray()
#
# train_dat = img_ndarray, np.array([1])
# val_dat = train_dat
#
# test_dat, fpath = get_image_ndarray()
#
# classes = [0,1]
#
#
# # initialize dataloader and learner
# sz = 400
# tfms = tfms_from_model(resnet34, sz, aug_tfms=transforms_side_on, max_zoom=1.2)
# data = ImageClassifierData.from_arrays(PATH, train_dat, val_dat, bs=1,
#                                        tfms=tfms, classes=classes, test=test_dat)
# learner = ConvLearner.pretrained(resnet34, data)
# learner.load('RN34_400_WD_λ0-529_00')
#
#
# # run neural net on image
# logpred,_ = learner.TTA(is_test=True)
# pred = np.mean(np.exp(logpred), 0)
#
#
# # get actual label from CSV
# label_df = pd.read_csv(PATH + 'labels.csv')
# folder   = fpath.split('/')[-2]
# fname    = fpath.split('/')[-1]
# answer   = labels_df.loc[labels_df['id']==folder+'/'+fname.split('.')[0]]['gloc'].values[0]
#
#
# # display results
# print(fname)
# print(pred)
# print(answer)
# plt.imshow(img_ndarray[0])
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# from fastai_osx.model import resnet34
# from fastai_osx.conv_learner import * # imports learner -> imports dataset & imports
#
#
# # PyTorch Tensor shape: (N, Channels, Rows, Cols) # or (,,X,Y)?
# train_dat = (np.array([]), np.array([]))
#
# sz = 400
# tfms = tfms_from_model(resnet34, sz)
# # ImageClassifierData.from_arrays(path, trn, val, bs=64, tfms=(None, None),
# #                                   classes=None, num_workers=4, test=None)
# data = ImageClassifierData.from_arrays(PATH, train_dat, val_dat, bs=1,
#                                        tfms=tfms, classes=classes, test=test_dat)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# import fastai_osx.dataset
#
#
#
# data = ImageClassifierData.from_csv(PATH, 'train', labels_csv, bs=bs, tfms=tfms,
#                                     val_idxs=val_idxs, suffix='.jpg', num_workers=8,
#                                     test_name=test_name)
# # learner = ConvLearner.pretrained(resnet34, data)
#
#
# # ImageClassifierData:
# @classmethod
# def from_csv(cls, path, folder, csv_fname, bs=64, tfms=(None,None),
#              val_idxs=None, suffix='', test_name=None, continuous=False,
#              skip_header=True, num_workers=8):
#     """ Read in images and their labels given a CSV file. """
#
#
#
#
#
#
#
#
#
# help(fastai_osx.dataset.ImageClassifierData)
#
#
# help(fastai_osx.dataset.ImageData)
