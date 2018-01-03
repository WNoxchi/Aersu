# Wayne Nixalo - 2018-Jan-03 02:01

# This script demonstrates how to display an image with OpenCV, and close it
# after entering input to the terminal.
#
# My guess is, OpenCV needs time to open the image. Without `time.sleep`,
# OpenCV was only openning an empty window. So, the execution of the code
# didn't leave enough time for OpenCV to open its image.
#
# I tried to find a lower bound for waiting-time. I learned 2 cool things:
#   1. OpenCV opens the image in a small horz-rectangular window that starts
#       all black.
#   2. The image is then resized to its correct dimensions
# I found at 1e-13 (100 femtoseconds) I could sometimes catch the image before
# resizing. At 1e-120 (fucking small.... 76 orders of magnitude smaller than
# Plank Time....) I could often catch the black small window.
# There's no way a modern computer is doing anything at even 1e-20
# seconds. Just tested that, yeah. There's a minimum beyond-which time.sleep
# just rounds to zero. Trying with time.sleep(0) gives the same behavior. Cool.

import cv2
import time
import sys

image = cv2.imread('blah.jpg', cv2.IMREAD_COLOR)
cv2.imshow('blah', image)
cv2.waitKey(1)
time.sleep(1e-3)
inp = input("type something")
if inp:
    cv2.destroyAllWindows()

# references:
# https://stackoverflow.com/questions/27117705/how-to-update-imshow-window-for-python-opencv-cv2
# https://stackoverflow.com/questions/14494101/using-other-keys-for-the-waitkey-function-of-opencv
