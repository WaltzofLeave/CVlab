from configuration import IMAGEPATH
from configuration import SAVEDPATH
import numpy as np
import cv2

def get(gray=True):
    if gray:
        return cv2.imread(IMAGEPATH + r'\lxm.png',cv2.IMREAD_GRAYSCALE)
    else:
        return cv2.imread(IMAGEPATH + r'\lxm.png')
