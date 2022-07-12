# basic processing of image

import cv2
import numpy as np

img=cv2.imread("Resources/anh1.png")
kernel = np.ones((5,5), np.uint8)

imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # chuyển sang màu xám
imgBlur = cv2.GaussianBlur(imgGray, (7,7), 0) # làm mờ ảnh, kích thước hạt nhân là lẻ
imgCanny = cv2.Canny(img, 100, 100)# phát hiện các cạnh, góc trong ảnh
imgDialation = cv2.dilate(imgCanny, kernel ,iterations=1) # độ dày của đường nét --> số lần lặp lại đường nét
imgEroded = cv2.erode(imgDialation, kernel, iterations=1) # làm giảm độ dày của đường nét

cv2.imshow("Gray image", imgGray)
cv2.imshow("Blur image", imgBlur)
cv2.imshow("Canny image", imgCanny)
cv2.imshow("Dialation image", imgDialation)
cv2.imshow("Eroded image", imgEroded)

cv2.waitKey(0)