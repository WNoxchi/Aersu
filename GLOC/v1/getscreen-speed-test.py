# import numpy as np
# import cv2
from PIL import ImageGrab
from PIL import Image

from time import time

bbox = (8,160,690,544)
bbox = tuple([i*2 for i in bbox])

t0 = time()
ImageGrab.grab(bbox=bbox).save('blah.png')
print(f'PNG(PIL): {time()-t0} s')

t0 = time()
ImageGrab.grab(bbox=bbox).convert("RGB").save('blah.jpg')
print(f'JPG(PIL): {time()-t0} s')


from mss.darwin import MSS as mss
import mss.tools

t0 = time()
with mss.mss() as sct:
    # Use the 1st monitor
    monitor = sct.monitors[1]

    # im = sct.grab((8,2*160,682,384))
    bbox = (8,160,682,544)
    im = sct.grab(bbox)

    # mss.tools.to_png(im.rgb, im.size, 'blah.png')
    print(f'PNG(MSS): {time()-t0} s')
