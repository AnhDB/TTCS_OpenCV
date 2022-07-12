#Color detection
#Phát hiện màu sắc
import cv2

path = "Resources/lambo.png"
img = cv2.imread(path)


# chuyển sang màu HSV
imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

cv2.imshow("Original", img)
cv2.imshow("HSV", imgHSV)

cv2.waitKey(0)