# Read images, videos, webcam
# import cv2

#
# img = cv2.imread("Resources/anh.png")
# cv2.imshow('image',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


#đọc video

# cap = cv2.VideoCapture("Resources/test_video.mp4")
#
# while True:
#     success, img=cap.read()
#     cv2.imshow("Video",img)
#     if cv2.waitKey(1) & 0xFF == ord('q'): #bấm q thì dừng
#         break


#đọc video từ webcam
# cap = cv2.VideoCapture(0) # ghi id của webcam nếu có nhiều , ghi 0 nếu có 1 webcam
# cap.set(3,320) # độ rộng của khung hình, id=3
# cap.set(4, 480) # chiều dài, id=4, kích thước: 320x480
# cap.set(10, 100) # độ sáng của cam
# while True:
#     success, img = cap.read()
#     cv2.imshow("Video", img)
#     if cv2.waitKey(1) & 0xFF== ord('q'):
#         break
 #using matplotlib
# import numpy as np
# import cv2
# from matplotlib import pyplot as plt
# img = cv2.imread('Resources/anh.png',0)
# plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
# plt.xticks([]), plt.yticks([]) # ẩn trục x,y
# plt.show()

# import numpy as np
# import cv2
# cap = cv2.VideoCapture(0)
# while(True):
#     # Chụp từng khung
#     ret, frame = cap.read()
#     # Our operations on the frame come here
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     # Hiển thị kết quả
#     cv2.imshow('frame',gray)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# Khi đã hoàn thành, giải phóng webcam, sau đó đóng tất cả cửa sổ imshow ().
# cap.release()
# cv2.destroyAllWindows()