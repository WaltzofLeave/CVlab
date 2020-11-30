from Script.util.conv import medianFilter22
import cv2
import numpy as np
from Script.util.show import show
from Script.util.conv import conv
from configuration import SAVEDPATH
import matplotlib.pyplot as plt


def median_filter(img):
    img = medianFilter22(img,(3,3))
    show(img)
    return img

def mean_filter(img):
    core = 1/9 * np.ones((3,3))
    img = conv(img, core)
    show(img)
    return img

def mission2():
    img1 = cv2.imread(SAVEDPATH + "\guass_noise.jpg", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(SAVEDPATH + "\salt_and_pepper_noise.jpg", cv2.IMREAD_GRAYSCALE)
    show(img1)
    show(img2)
    median_filter(img1)
    mean_filter(img2)
    mean_filter(img1)
    mean_filter(img2)

mission2()