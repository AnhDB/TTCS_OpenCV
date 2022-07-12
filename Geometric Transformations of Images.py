# Thay đổi kích thước hình ảnh
# import cv2
# import numpy as np
# img = cv2.imread('Resources/anh1.png')
# res = cv2.resize(img,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
# # cv2.imshow('before', img)
# # cv2.imshow('after', res)
# # cv2.waitKey(0)
# #OR
# height, width = img.shape[:2]
# res = cv2.resize(img,(0.5*width, 0.5*height), interpolation = cv2.INTER_CUBIC)
# cv2.imshow('before', img)
# cv2.imshow('after', res)
# cv2.waitKey(0)

# dịch ảnh
# import cv2
# import numpy as np
# img = cv2.imread('Resources/anh1.png')
# img = cv2.resize(img,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC) # đổi kích thước ảnh
# rows,cols = img.shape[0], img.shape[1]
# M = np.float32([[1,0,100],[0,1,50]])
# dst = cv2.warpAffine(img,M,(cols,rows))
# cv2.imshow('original', img)
# cv2.imshow('img',dst)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#xoay ảnh
# import cv2
# img = cv2.imread('Resources/anh1.png')
# img = cv2.resize(img,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC) # đổi kích thước ảnh
# rows,cols = img.shape[0], img.shape[1]
# M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
# dst = cv2.warpAffine(img,M,(cols,rows))
# cv2.imshow('original', img)
# cv2.imshow('img', dst)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Chuyển đổi Affin
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# img = cv2.imread('Resources/ball.png')
# rows,cols,ch = img.shape
# pts1 = np.float32([[50,50],[200,50],[50,200]])
# pts2 = np.float32([[10,100],[200,50],[100,250]])
# M = cv2.getAffineTransform(pts1,pts2)
# dst = cv2.warpAffine(img,M,(cols,rows))
# plt.subplot(121) # hiển thị bên trái
# plt.imshow(img),plt.title('Input')
# plt.subplot(122) # hiển thị bên phải
# plt.imshow(dst),plt.title('Output')
# plt.show()

# Chuyển đổi góc nhìn
import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('Resources/anh.png')
rows,cols,ch = img.shape
pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
pts2 = np.float32([[0,0],[200,0],[0,200],[200,200]])
M = cv2.getPerspectiveTransform(pts1,pts2)
dst = cv2.warpPerspective(img,M,(200,200))
plt.subplot(121),plt.imshow(img),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')
plt.show()