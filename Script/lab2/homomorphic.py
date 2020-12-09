import cv2
import numpy as np
from Script.util.show import show
from Script.lab1.getpicture import get
from Script.lab1.noise import gauss_noise
def BHPF(img:np.ndarray,D0=360,n=14):
    filter = np.zeros(img.shape)
    for i in range(0,filter.shape[0]):
        for j in range(0,filter.shape[1]):
            filter[i][j] = np.sqrt((i-filter.shape[0]//2)**2 + (j-filter.shape[1]//2)**2)
    img1 = img + 0.000000001
    return img * (1/(1+(D0/filter)**(2*n)))

def homomorphic(img:np.ndarray):
    img = img + 0.000000001
    print(img)
    img = np.log(img)
    print(img)
    img = np.fft.fft(img)
    img = BHPF(img)
    img = np.fft.ifft(img)
    img = np.exp(img)
    print(img)
    img = np.array(img, dtype='f')
    print(img)
    show(img)
    return img

img = get()
img = gauss_noise(img)
show(img)
img = img / 255
homomorphic(img)
