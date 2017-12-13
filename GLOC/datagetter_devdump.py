# 2017-Dec-12 15:39 Wayne H Nixalo
# G-Lock Data getter
################################################################################
import numpy as np
import cv2
from getkey import getKey
from getscreen import getScreen
from time import time

def key_to_output(key):
    """
    Converts a keypress to a positive One-Hot Encoded label, zero otherwise.
    """
    return getKey()


# MACOS Retina doubles resolution. May need halve on Linux/Win
bbox = (8,160,682,544)
h = (bbox[3] - bbox[1]) * 2 # 768
w = (bbox[2] - bbox[0]) * 2 # 1348

# x & y transforms
tfx = 400 / w
tfy = 400 / h

# t = 0.
# tz = time()
n = 1
for i in range(n):

    img = getScreen(bbox=bbox)

    img = np.asarray(img)

    # t0 = time()
    # img = cv2.resize(img, None, fx=tfx, fy=tfy, interpolation=cv2.INTER_AREA)
    img = cv2.resize(img, None, fx=tfx, fy=tfy)
    # t += time()-t0

    img = cv2.GaussianBlur(img, (5,5), 0)


    cv2.imwrite('blah2.jpg', img)


    # t0 = time()
    # cv2.imwrite('blah.png', img)
    # t += time()-t0

# print(f'MSS screengrab saved by OpenCV: {t/n} s')
# print(f'MSS screengrab: {t/n} s')
# print(f'NumPy array conversion: {t/n} s')
# print(f'Elapsed Time: {time()-tz} s')

# scipy.misc.imresize(arr, size, interp='bilinear', mode=None)[source]
################################################################################
# BENCHMARKS

# MSS screengrab saved by OpenCV (100 runs avg):
# (JPG):    0.03196930885314941 s       (400x400):  0.005743522644042969 s
# (PNG):    0.05808886766433716 s       (400x400):  0.011203761100769044 s

# MSS screengrab (1000 runs avg):
#           0.018143182039260863 s      (400x400):

# MSS img to NumPy array conversion:
#           4.024839401245117e-05 s     (400x400):

# OpenCV resize 768x1348 -> 400x400:
#           0.0016115713119506836 s
#           0.0048240418434143065 s  [interpolation=INTER_AREA]

# So my priority is to minimize the size of what I'm saving. Hell most of the
# models I use are trained in the 244 square-pixel range anyway.

# I should be getting good FPS saving 400x400 boxes, but I have to resize them
# with OpenCV I think.
