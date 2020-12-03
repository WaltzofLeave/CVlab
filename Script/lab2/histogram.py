import numpy as np
from Script.lab1.getpicture import get
import cv2
class mapping:
    dic = None
    def __init__(self,l,isdict=False):
        if isdict:
            self.dict = l
            return
        self.dic = {}
        for i in range(0,len(l)):
            self.dic[i] = l[i]

    def to(self,a):
        return self.dic[a]
    def back(self,a):
        for key in self.dic:
            if self.dic[key] == a:
                return key
        return -1

def gettable(level:int)->list:
    l = []
    for i in range(0, level):
        l.append(i * (256 // level))
    return l

def leveled(img:np.ndarray,table:list)->np.ndarray:
    leveledimg = np.zeros(img.shape)
    for i in range(0,img.shape[0]):
        for j in range(0,img.shape[1]):
            for k in range(0,len(table)):
                leveledimg[i][j] = len(table)-1
                if img[i][j] < table[k]:
                    leveledimg[i][j] = k-1
                    break
    return np.array(leveledimg,dtype='u1')
def restored(leveledimg:np.ndarray,tabel:list)->np.ndarray:
    img = np.zeros(leveledimg.shape)
    for i in range(0, leveledimg.shape[0]):
        for j in range(0,leveledimg.shape[1]):
            img[i][j] = tabel[leveledimg[i][j]]
    return np.array(img,dtype='u1')



def histogram_equalization(img: np.ndarray,level:int)->np.ndarray:
    table = gettable(level)
    img = leveled(img,table)
    n = np.zeros((level,))
    p = np.zeros((level,))
    c = np.zeros((level,))
    g = np.zeros((level,))
    relation = {}
    for i in range(0,img.shape[0]):
        for j in range(0,img.shape[1]):
            n[img[i][j]] += 1

    imglen = img.shape[0] * img.shape[1]
    for i in range(0,level):
        p[i] = n[i]/imglen

    c[0] = p[0]
    for i in range(1,level):
        c[i] = c[i-1] + p[i]

    g = np.floor((level-1)*c+0.5)

    for i in range(0,img.shape[0]):
        for j in range(0,img.shape[1]):
            img[i][j] = g[img[i][j]]

    return restored(img,table)


def histogram_specification(img:np.ndarray, level: int, spec: list)->np.ndarray:
    table = gettable(level)
    img = leveled(img, table)
    n = np.zeros((level,))
    p = np.zeros((level,))
    c = np.zeros((level,))
    g = np.zeros((level,))
    relation = {}
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            n[img[i][j]] += 1

    imglen = img.shape[0] * img.shape[1]
    for i in range(0, level):
        p[i] = n[i] / imglen

    c[0] = p[0]
    for i in range(1, level):
        c[i] = c[i - 1] + p[i]

    g = np.floor((level - 1) * c + 0.5)

    p1 = np.zeros((level,))
    c1 = np.zeros((level,))
    p1 = spec
    print(p1)
    print(c1)
    c1[0] = p1[0]
    for i in range(1, level):
        c1[i] = c1[i - 1] + p1[i]
    g1 = np.floor((level - 1) * c1 + 0.5)
    print("g:\n",g)
    print("g1:\n",g1)
    print(img)
    for i in range(0,img.shape[0]):
        for j in range(0,img.shape[1]):
            w = len(spec) - 1
            flag = 0
            for k in range(1,len(spec)):
                if g[img[i][j]] < g1[k]:
                    img[i][j] = k-1
                    flag = 1
                    break
            if flag == 0:
                img[i][j] = w

    print(img)
    return restored(img,table)
img = get()
img = histogram_specification(img,8,[0,0,0,0.15,0.20,0.30,0.20,0.15])
cv2.imshow('ok',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
