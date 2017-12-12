# import numpy as np
# import cv2
from PIL import ImageGrab
from PIL import Image

from time import time

bbox = (8,160,690,544)
bbox = tuple([i*2 for i in bbox])

t0 = time()
ImageGrab.grab(bbox=bbox).save('blah.png')
print(f'PNG: {time()-t0} s')

t0 = time()
# ImageGrab.grab(bbox=bbox).convert("RGB").save('blah.jpg')
ImageGrab.grab()
print(f'JPG: {time()-t0} s')
