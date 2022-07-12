#joining images: hiển thị nhiều hình ảnh cùng lúc

import cv2
import numpy as np


"""
#ngăn xếp ngang horizontal stack, xếp ảnh theo hàng ngang
imgHor = np.hstack((img, img))

#xếp ảnh theo hàng dọc
imgVer = np.vstack((img, img))
#---> không thể thay đổi size của ảnh --> tràn -->sd numpy
cv2.imshow("Horizontal", imgHor)
cv2.imshow("Vertical", imgVer)
"""


def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver


img = cv2.imread("Resources/anh1.png")
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#tạo ngăn xếp  các ảnh, 0.2: tỉ lệ ảnh
imgStack = stackImages(0.2, ([img,imgGray, img, img], [img, img, img, imgGray])) # (ngang, doc)

cv2.imshow("ImageStack", imgStack)

cv2.waitKey(0)