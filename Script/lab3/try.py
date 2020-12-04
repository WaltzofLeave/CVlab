import numpy as np
import cv2
from Script.lab3.getpicture import get
from Script.util.show import show
from Script.lab1.sobel import sobel
car = get(False)

bluef = car[...,0]
bluef = np.array(bluef,dtype='f4')

anotherf = car[...,1]
anotherf = np.array(anotherf,dtype='f4')

ansf = bluef - anotherf

ansf = (np.abs(ansf) + ansf )//2
print(ansf)
ansf = np.array(ansf,dtype='u1')
# cv2.imshow('test',ansf)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
show(ansf)
#ret,ansf = cv2.threshold(ansf,80,255,cv2.THRESH_BINARY)
ansfx = cv2.Sobel(ansf,-1,1,0)
ansfy = cv2.Sobel(ansf,-1,0,1)
ansf = np.array(ansfx + ansfy,dtype='u1')
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
for i in range(0,10):
    ansf = cv2.dilate(ansf,kernel)
for i in range(0,10):
    ansf = cv2.erode(ansf,kernel)

ret,ansf = cv2.threshold(ansf,60,255,cv2.THRESH_BINARY)
show(ansf)
# ansfx = cv2.Sobel(ansf, -1, 1, 0)
# ansfy = cv2.Sobel(ansf, -1, 0, 1)
# ansf = np.array(ansfx + ansfy,dtype='u1')
ansf = cv2.morphologyEx(ansf,cv2.MORPH_GRADIENT,kernel)

show(ansf)
