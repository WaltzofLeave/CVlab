import numpy as np
import cv2
from Script.lab3.getpicture import get
from Script.util.show import show
from Script.lab1.sobel import sobel
car = get(False)
def getlisenceplate(car:np.ndarray):
    bluef = car[...,0]
    bluef = np.array(bluef,dtype='f4')

    anotherf = car[...,1]
    anotherf = np.array(anotherf,dtype='f4')

    anotherf1 = car[...,2]
    anotherf1 = np.array(anotherf1,dtype='f4')

    ansf1 = bluef - anotherf
    ansf2 = bluef - anotherf1

    ansf = (ansf1 + ansf2) // 2

    ansf = (np.abs(ansf) + ansf )//2
    print(ansf)
    ansf = np.array(ansf,dtype='u1')
    # cv2.imshow('test',ansf)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #show(ansf)
    #ret,ansf = cv2.threshold(ansf,80,255,cv2.THRESH_BINARY)
    ansfx = cv2.Sobel(ansf,-1,1,0)
    ansfy = cv2.Sobel(ansf,-1,0,1)
    ansf = np.array(ansfx + ansfy,dtype='u1')
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
    for i in range(0,9):
        ansf = cv2.dilate(ansf,kernel)
    for i in range(0,9):
        ansf = cv2.erode(ansf,kernel)

    for i in range(0,5):
        ansf = cv2.erode(ansf,kernel)
    for i in range(0,5):
        ansf = cv2.dilate(ansf,kernel)

    ret,ansf = cv2.threshold(ansf,60,255,cv2.THRESH_BINARY)
    #show(ansf)
    # ansfx = cv2.Sobel(ansf, -1, 1, 0)
    # ansfy = cv2.Sobel(ansf, -1, 0, 1)
    # ansf = np.array(ansfx + ansfy,dtype='u1')
    ansf = cv2.morphologyEx(ansf,cv2.MORPH_GRADIENT,kernel)
    #ansf = cv2.dilate(ansf, kernel)
    #show(ansf)
    # lines = cv2.HoughLines(ansf, 0.8, np.pi/180, 40)
    # board = np.zeros((ansf.shape[0],ansf.shape[1],3))
    # for line in lines:
    #     rho, theta = line[0]
    #     a = np.cos(theta)
    #     b = np.sin(theta)
    #     x0 = a * rho
    #     y0 = b * rho
    #     x1 = int(x0 + 1000 * (-b))
    #     y1 = int(y0 + 1000 * (a))
    #     x2 = int(x0 - 1000 * (-b))
    #     y2 = int(y0 - 1000 * (a))
    #     cv2.line(board, (x1, y1), (x2, y2), (0, 0, 255))
    # show(board)
    ret,ansf = cv2.threshold(ansf,254,255,cv2.THRESH_BINARY)
    #show(ansf)
    xmin = ansf.shape[1]-1
    xmax = 0
    ymin = ansf.shape[0]-1
    ymax = 0
    for i in range(0,ansf.shape[0]):
        for j in range(0,ansf.shape[1]):
            if ansf[i][j] >= 254:
                if xmin > j:
                    xmin = j
                if ymin > i:
                    ymin = i
                if xmax < j:
                    xmax = j
                if ymax < i:
                    ymax = i

    cv2.line(car, (xmin, ymin), (xmax, ymin), (0, 0, 255),3)
    cv2.line(car, (xmin, ymax), (xmax, ymax), (0, 0, 255),3)
    cv2.line(car, (xmin, ymin), (xmin, ymax), (0, 0, 255),3)
    cv2.line(car, (xmax, ymin), (xmax, ymax), (0, 0, 255),3)
    show(car)
    return car

car = getlisenceplate(car)