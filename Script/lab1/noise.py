import numpy as np
import cv2
from configuration import IMAGEPATH
from configuration import SAVEDPATH
from Script.util.show import show


picture = cv2.imread(IMAGEPATH+'\lena.jpg')
show(picture)

def salt_and_pepper_noise(picture:np.ndarray,SNR:float=0.75,verbose:bool=True)->np.ndarray:
    orig_size = picture.shape
    arr = picture.flatten()
    length = arr.shape[0]
    choice = np.random.randint(0, length/3, int((length/3)*(1-SNR)))
    for i in choice:
        x = np.random.random()
        if x < 0.5:
            arr[i * 3] = 0
            arr[i * 3 + 1] = 0
            arr[i * 3 + 2] = 0
        else:
            arr[i * 3] = 255
            arr[i * 3 + 1] = 255
            arr[i * 3 + 2] = 255
    picture = arr.reshape(orig_size)
    if verbose:
        show(picture)
    cv2.imwrite(SAVEDPATH + r'\salt_and_pepper_noise.jpg',picture)
    return picture
def gauss_noise(picture:np.ndarray,SNR:float = 0.75,verbose:bool=True)->np.ndarray:
    orig_size = picture.shape
    arr = picture.flatten()
    length = arr.shape[0]
    choice = np.random.randint(0, length / 3, int((length / 3) * (1 - SNR)))
    for i in choice:
        x = np.random.random()
        arr[i*3] = np.random.normal(arr[i*3],20)
        arr[i*3+1] = np.random.normal(arr[i*3+1], 20)
        arr[i*3+2] = np.random.normal(arr[i*3+2], 20)
    picture = arr.reshape(orig_size)
    if verbose:
        show(picture)
    cv2.imwrite(SAVEDPATH + r'\guass_noise.jpg',picture)
    return picture
#sppicture = salt_and_pepper_noise(picture)
#gspicture = gauss_noise(picture)