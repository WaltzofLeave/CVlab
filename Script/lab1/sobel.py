import numpy as np
from Script.util.conv import conv
from Script.util.conv import touintpicture
from Script.lab1.getpicture import get
from Script.util.show import show
def sobel(img,xyreturn=False):
    corex = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    corey = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    imgx = conv(img,corex)
    imgx1 = np.abs(imgx)
    imgy = conv(img,corey)
    imgy1 = np.abs(imgy)
    img = imgx1 + imgy1
    img = touintpicture(img)
    if xyreturn:
        return img,imgx,imgy
    return img

def test_sobel():
    img = get()
    img = sobel(img)
    show(img)

#test_sobel()