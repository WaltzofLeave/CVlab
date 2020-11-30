import cv2
import numpy as np
from Script.lab1.getpicture import get
from Script.util.show import show
class morphology:
    @classmethod
    def binaryzation(cls,img,threshold=127):
        for i in range(0,img.shape[0]):
            for j in range(0,img.shape[1]):
                if img[i][j] < threshold:
                    img[i][j] = 0
                else:
                    img[i][j] = 255
        return img

    @classmethod
    def erode(cls,img):
        ans = np.array(img)
        for i in range(2,img.shape[0]-2):
            for j in range(2,img.shape[1]-2):
                if img[i][j] == 0:
                    ans[i-2][j] = 0
                    ans[i-1][j] = 0
                    ans[i][j] = 0
                    ans[i+1][j] = 0
                    ans[i+2][j] = 0
                    ans[i][j-2] = 0
                    ans[i][j-1] = 0
                    ans[i][j+1] = 0
                    ans[i][j+2] = 0
        return ans

    @classmethod
    def dilate(cls,img):
        ans = np.zeros(img.shape)
        for i in range(2,img.shape[0]-2):
            for j in range(2,img.shape[1]-2):
                if img[i][j] == 255:
                    ans[i - 2][j] = 255
                    ans[i - 1][j] = 255
                    ans[i][j] = 255
                    ans[i + 1][j] = 255
                    ans[i + 2][j] = 255
                    ans[i][j - 2] = 255
                    ans[i][j - 1] = 255
                    ans[i][j + 1] = 255
                    ans[i][j + 2] = 255
        return ans
    @classmethod
    def open(cls,img):
        img = cls.erode(img)
        img = cls.dilate(img)
        return img

    @classmethod
    def close(cls,img):
        img = cls.dilate(img)
        img = cls.erode(img)
        return img


img = get()
img = morphology.binaryzation(img)

# for i in range(0,10):
#     img = morphology.open(img)
#     img = morphology.close(img)

er = morphology.erode(img)
di = morphology.dilate(img)
ans = di - er
ans = np.abs(ans)
show(ans)

