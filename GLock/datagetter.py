# 2017-Dec-12 15:39 Wayne H Nixalo
# G-Lock Data getter
################################################################################

from getkey import getKey
import cv2

def key_to_output(key):
    """
    Converts a keypress to a positive One-Hot Encoded label, zero otherwise.
    """
    return getKey()

def 
