import numpy as np
import cv2
# Create a black image
img = np.zeros((512,512,3), np.uint8)
# Draw a diagonal blue line with thickness of 5 px
# vẽ đường thẳng
img = cv2.line(img,(0,0),(511,511),(255,0,0),5)
# vẽ một hình chữ nhật màu xanh ở góc trên cùng bên phải của hình ảnh.
img = cv2.rectangle(img,(384,0),(510,128),(0,255,0),3)
# vẽ một vòng tròn màu đỏ bên trong hình chữ nhật được vẽ trên.
img = cv2.circle(img,(447,63), 63, (0,0,255), -1)
# vẽ một nửa elip ở trung tâm hình ảnh.
img = cv2.ellipse(img,(256,256),(100,50),0,0,180,255,-1)
# vẽ một đa giác nhỏ với 4 đỉnh màu vàng
pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
pts = pts.reshape((-1,1,2))
img = cv2.polylines(img,[pts],True,(0,255,255))
#viết chữ openCV màu trắng
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,'OpenCV',(10,500), font, 4,(255,255,255),2,cv2.LINE_AA)
cv2.imshow('img', img)
cv2.waitKey(0)