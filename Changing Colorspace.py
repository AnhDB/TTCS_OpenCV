# # Theo dõi đối tượng
# import cv2
# import numpy as np
# cap = cv2.VideoCapture(0)
# while(1):
#     # Lấy từng khung hình
#     _, frame = cap.read()
#     # Convert BGR to HSV
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#     # Xác định phạm vi màu xanh dương trong HSV
#     lower_blue = np.array([110,50,50])
#     upper_blue = np.array([130,255,255])
#     # Giới hạn ảnh hsv để chỉ có màu xanh dương
#     mask = cv2.inRange(hsv, lower_blue, upper_blue)
#     # Bitwise-AND mask and original image
#     res = cv2.bitwise_and(frame,frame, mask= mask)
#     cv2.imshow('frame',frame)
#     cv2.imshow('mask',mask)
#     cv2.imshow('res',res)
#     k = cv2.waitKey(5) & 0xFF
#     if k == 27:
#         break
# cv2.destroyAllWindows()

import cv2
import numpy as np
green = np.uint8([[[0,255,0 ]]])
hsv_green = cv2.cvtColor(green,cv2.COLOR_BGR2HSV)
print (hsv_green)