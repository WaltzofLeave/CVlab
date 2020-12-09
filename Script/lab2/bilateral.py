import numpy as np
import cv2
from Script.util.conv import samepadding
from Script.lab1.getpicture import get
from Script.util.show import show
from Script.lab1.noise import gauss_noise
def bilateral(img:np.ndarray,sigmad:float=200,sigmar:float=200):
    img = np.array(img,dtype=np.float)
    g = np.zeros(img.shape,dtype=np.float)
    img = samepadding(img,np.eye(5))
    for i0 in range(0,img.shape[0]-4):
        for j0 in range(0,img.shape[1]-4):
            i = i0 + 2
            j = j0 + 2
            sumup = 0
            sumdown = 0
            for k in range(i0,i0+5):
                for l in range(j0,j0+5):
                    d = np.exp(-(((i-k)**2+(j-l)**2)/(2 * sigmad ** 2)))
                    r = np.exp(-((img[i][j]-img[k][l])**2)/(2 * sigmar ** 2))
                    w = d * r
                    sumdown += w
                    sumup += w * img[k][l]
            g[i0][j0] = sumup/sumdown
    g = np.array(g,dtype='u1')
    return g

img = get()
show(img)
img = gauss_noise(img,SNR=0.95)
img1 = bilateral(img)
show(img1)
