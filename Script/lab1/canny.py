import numpy as np
from Script.util.show import show
from Script.util.conv import conv
from Script.util.conv import touintpicture
from Script.lab1.sobel import sobel
from Script.lab1.getpicture import get
from configuration import IMAGEPATH
import cv2

def gaussfilter(img):
    filter = np.array([[0.0751,0.1238,0.0751],[0.1238,0.2042,0.1238],[0.0751,0.1238,0.0751]])
    img = conv(img,filter,scale=True)
    return img
def nms(img,imgx,imgy):
    degree = np.arctan(img)
    for i in range(1,img.shape[0]-1):
        for j in range(1,img.shape[1]-1):
            if degree[i][j] <= - np.pi * 3 /8  or degree[i][j] > np.pi * 3 / 8:
                if img[i][j] < img[i][j+1] or img[i][j] < img[i][j-1]:
                    img[i][j] = 0
            elif -np.pi * 3 / 8 < degree[i][j] <= -np.pi / 8:
                if img[i][j] < img[i + 1][j - 1] or img[i][j] < img[i - 1][j + 1]:
                    img[i][j] = 0
            elif -np.pi / 8  < degree[i][j] <= np.pi/8:
                if img[i][j] < img[i + 1][j] or img[i][j] < img[i - 1][j]:
                    img[i][j] = 0
            elif np.pi / 8 < degree[i][j] <= np.pi * 3 / 8:
                if img[i][j] < img[i + 1][j + 1] or img[i][j] < img[i - 1][j - 1]:
                    img[i][j] = 0
            else:
                print("There is an ERROR in degree decision!")
    return img
def dtd(img,high,low):
    for i in range(1,img.shape[0]-1):
        for j in range(1,img.shape[1]-1):
            if img[i][j] < low:
                img[i][j] = 0
            elif img[i][j] > high:
                img[i][j] = 255
            elif low < img[i][j] < high:
                if any([img[k][t] > high for k in range(i-1,i+2) for t in range(j-1,j+2)]):
                    img[i][j] = 255
                else:
                    img[i][j] = 0
    return img
def canny(img,high=60,low=30):
    img = gaussfilter(img)
    img , imgx, imgy = sobel(img,xyreturn=True)
    img = nms(img,imgx,imgy)
    img = dtd(img,high,low)
    return img

img = cv2.imread(IMAGEPATH + r'\cyh.jpg',cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img,(500,500))
img = canny(img)
show(img)

