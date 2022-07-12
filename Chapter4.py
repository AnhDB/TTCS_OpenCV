#Shapes and texts

import cv2
import numpy as np

#tạo 1 ma trận toàn số 0, số 0 tương đương màu đen  --> tạo 1 ảnh màu đen
img = np.zeros((512, 512, 3), np.uint8)  # kích thước 512x512, độ tương phản = 3

# img[:] = 255, 0, 0  # đổi sang màu xanh

# tạo đường vẽ có điểm bắt đầu (0,0), điểm cuối (300, 300), màu xanh lá cây có mã (0, 255, 0), độ dày = 3
cv2.line(img, (0,0), (300, 300), (0, 255, 0), 3)

# tạo đường vẽ bằng đường chéo của hình ảnh, màu xanh lá cây có mã (0, 255, 0), độ dày = 3
cv2.line(img, (0,0),(img.shape[1], img.shape[0]), (0, 255, 0), 3) #shape[1]: chiều rộng, shape[0]: chiều cao

#tạo hình chữ nhật
cv2.rectangle(img, (0,0), (250, 350), (0,0,255), 2) #250: chiều rộng, 350: chiều cao, màu đỏ

# dùng  Filled  để tô hình chữ nhật
cv2.rectangle(img, (0,0), (250, 350), (0,0,255), cv2.FILLED)


#tạo hình tròn (400, 50): tâm, 30: bán kính, (255,255,0) : màu xanh lam, độ dày=5
cv2.circle(img, (400, 50), 30, (255,255,0),5)

#viết chữ lên hình ảnh, txt="OPENCV", vị trí bắt đầu = (300, 100), phông chữ HERSHEY, tỉ lệ chữ=1, màu = (0,150,0), độ dày=3
cv2.putText(img, "OPENCV", (300, 150), cv2.FONT_HERSHEY_COMPLEX, 1, (0,150,0), 3)

cv2.imshow("Image", img)

cv2.waitKey(0)