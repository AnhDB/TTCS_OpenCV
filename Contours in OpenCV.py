# Getting Started
# import matplotlib.pyplot as plt
# import cv2
# img = cv2.imread("Resources/anh.png")
# imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# ret, thresh = cv2.threshold(imgray, 127, 255, 0)
# contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# # Vẽ tất cả các đường viền
# img1 = cv2.drawContours(img, contours, -1, (0,255,0), 3)
# # Vẽ đường bao riêng lẻ
# img2 = cv2.drawContours(img, contours, 3, (0,255,0), 3)
# # Phương pháp tối ưu
# cnt = contours[4]
# img3 = cv2.drawContours(img, [cnt], 0, (0,255,0), 3)
# # show image
# plt.subplot(1,3,1),plt.imshow(img1)
# plt.title('All the contours'), plt.xticks([]), plt.yticks([])
# plt.subplot(1,3,2),plt.imshow(img2)
# plt.title('An individual contour'), plt.xticks([]), plt.yticks([])
# plt.subplot(1,3,3),plt.imshow(img3)
# plt.title('Useful-img2'), plt.xticks([]), plt.yticks([])
# plt.show()
# cv2.waitKey(0)

# Contour Features
# import cv2
# import numpy as np
# img = cv2.imread('Resources/anh.png',0)
# ret,thresh = cv2.threshold(img,127,255,0)
# contours,hierarchy = cv2.findContours(thresh, 1, 2)
# cnt = contours[0]
# M = cv2.moments(cnt)
# # area =cv2.contourArea(cnt) # contour Area
# # perimeter = cv2.arcLength(cnt,True) # Chu vi đường tròn
# # print(M)
# # print(area)
# # print(perimeter)
# k=cv2.isContourConvex(cnt) # kiểm tra độ lồi
# print(k)

#  bounding rectangle
# import cv2
# import numpy as np
# img = cv2.imread('Resources/set.png', 0)
# ret,thresh = cv2.threshold(img,127,255,0)
# contours,hierarchy = cv2.findContours(thresh, 1, 2)
# cnt = contours[0]
# x,y,w,h =cv2.boundingRect(cnt)
# img=cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 2)
# rect = cv2.minAreaRect(cnt)
# box = cv2.boxPoints(rect)
# box = np.int0(box)
# img=cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
# cv2.imshow('out', img)
# cv2.waitKey(0)

# # minimim Enclosing Circle
# import cv2
# import numpy as np
# img = cv2.imread('Resources/set.png', 0)
# ret,thresh = cv2.threshold(img,127,255,0)
# contours,hierarchy = cv2.findContours(thresh, 1, 2)
# cnt = contours[0]
# (x,y),radius = cv2.minEnclosingCircle(cnt)
# center = (int(x),int(y))
# radius = int(radius)
# img = cv2.circle(img,center,radius,(0,255,0),2)
# cv2.imshow('out', img)
# cv2.waitKey(0)

# # elip
# import cv2
# import numpy as np
# img = cv2.imread('Resources/set.png', 0)
# ret,thresh = cv2.threshold(img,127,255,0)
# contours,hierarchy = cv2.findContours(thresh, 1, 2)
# cnt = contours[0]
# ellipse = cv2.fitEllipse(cnt)
# img = cv2.ellipse(img,ellipse,(0,255,0),2)
# cv2.imshow('out', img)
# cv2.waitKey(0)

# # fitting a line
# import cv2
# import numpy as np
# img = cv2.imread('Resources/set.png', 0)
# ret,thresh = cv2.threshold(img,127,255,0)
# contours,hierarchy = cv2.findContours(thresh, 1, 2)
# cnt = contours[0]
# rows,cols = img.shape[:2]
# [vx,vy,x,y] = cv2.fitLine(cnt, cv2.DIST_L2,0,0.01,0.01)
# lefty = int((-x*vy/vx) + y)
# righty = int(((cols-x)*vy/vx)+y)
# img = cv2.line(img,(cols-1,righty),(0,lefty),(0,255,0),2)
# cv2.imshow('out', img)
# cv2.waitKey(0)


# Tỉ lệ khung hình
# import cv2
# import numpy as np
# img = cv2.imread('Resources/set.png', 0)
# ret,thresh = cv2.threshold(img,127,255,0)
# contours,hierarchy = cv2.findContours(thresh, 1, 2)
# cnt = contours[0]
# x,y,w,h = cv2.boundingRect(cnt)
# aspect_ratio = float(w)/h
# print(aspect_ratio)

# Mức độ
# import cv2
# import numpy as np
# img = cv2.imread('Resources/set.png', 0)
# ret,thresh = cv2.threshold(img,127,255,0)
# contours,hierarchy = cv2.findContours(thresh, 1, 2)
# cnt = contours[0]
# area = cv2.contourArea(cnt)
# x,y,w,h = cv2.boundingRect(cnt)
# rect_area = w*h
# extent = float(area)/rect_area
# print(extent)


# # Độ rắn
# import cv2
# import numpy as np
# img = cv2.imread('Resources/set.png', 0)
# ret,thresh = cv2.threshold(img,127,255,0)
# contours,hierarchy = cv2.findContours(thresh, 1, 2)
# cnt = contours[0]
# area = cv2.contourArea(cnt)
# hull = cv2.convexHull(cnt)
# hull_area = cv2.contourArea(hull)
# solidity = float(area)/hull_area
# print(solidity)

# # Đường kính tương đương
# import cv2
# import numpy as np
# img = cv2.imread('Resources/set.png', 0)
# ret,thresh = cv2.threshold(img,127,255,0)
# contours,hierarchy = cv2.findContours(thresh, 1, 2)
# cnt = contours[0]
# area = cv2.contourArea(cnt)
# equi_diameter = np.sqrt(4*area/np.pi)
# print(equi_diameter)

# # Đường kính tương đương
# import cv2
# import numpy as np
# img = cv2.imread('Resources/set.png', 0)
# ret,thresh = cv2.threshold(img,127,255,0)
# contours,hierarchy = cv2.findContours(thresh, 1, 2)
# cnt = contours[0]
# (x,y),(MA,ma),angle = cv2.fitEllipse(cnt)
# print(angle)

 # Mask and Pixel Points
# import cv2
# import numpy as np
# img = cv2.imread('Resources/set.png', 0)
# ret,thresh = cv2.threshold(img,127,255,0)
# contours,hierarchy = cv2.findContours(thresh, 1, 2)
# cnt = contours[0]
# mask = np.zeros(img.shape,np.uint8)
# cv2.drawContours(mask,[cnt],0,255,-1)
# pixelpoints = np.transpose(np.nonzero(mask))
# #pixelpoints = cv2.findNonZero(mask)
# print(pixelpoints)

 # Maximum Value, Minimum Value and their locations
# import cv2
# import numpy as np
# img = cv2.imread('Resources/set.png', 0)
# ret,thresh = cv2.threshold(img,127,255,0)
# contours,hierarchy = cv2.findContours(thresh, 1, 2)
# cnt = contours[0]
# mask = np.zeros(img.shape,np.uint8)
# min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(img,mask = mask)
# print(f"Min_val: {min_val}")
# print(f"Max_val: {max_val}")
# print(f"Min_loc: {min_loc}")
# print(f"Max_loc: {max_loc}")

# # 8.	Màu trung bình và cường độ trung bình
# import cv2
# import numpy as np
# img = cv2.imread('Resources/set.png', 0)
# ret,thresh = cv2.threshold(img,127,255,0)
# contours,hierarchy = cv2.findContours(thresh, 1, 2)
# cnt = contours[0]
# mask = np.zeros(img.shape,np.uint8)
# mean_val = cv2.mean(img,mask = mask)
# print(f"Mean_val: {mean_val}")


# Extreme Points
# import cv2
# import numpy as np
# img = cv2.imread('Resources/set.png', 0)
# ret,thresh = cv2.threshold(img,127,255,0)
# contours,hierarchy = cv2.findContours(thresh, 1, 2)
# cnt = contours[0]
# leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
# rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
# topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
# bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])
# print(f"Leftmost: {leftmost}")
# print(f"Rightmost: {rightmost}")
# print(f"Topmost: {topmost}")
# print(f"Bottommost: {bottommost}")

# # Tính toán biểu đồ OpenCV
# import cv2
# import numpy as np
# img = cv2.imread('Resources/anh1.png', 0)
# hist = cv2.calcHist([img], [0], None, [256], [0, 256])
# print(hist)

# # Tính toán biểu đồ OpenCV
# import cv2
# import numpy as np
# img = cv2.imread('Resources/anh1.png', 0)
# hist, bins = np.histogram(img.ravel(), 256, [0, 256])
# print(hist)

# Vẽ biểu đồ Matplotlib
# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
# img = cv2.imread('Resources/anh1.png',0)
# cv2.imshow('img', img)
# plt.hist(img.ravel(),256,[0,256])
# plt.show()

# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
# img = cv2.imread('Resources/anh1.png')
# color = ('b','g','r')
# for i,col in enumerate(color):
#     histr = cv2.calcHist([img],[i],None,[256],[0,256])
#     plt.plot(histr,color = col)
#     plt.xlim([0,256])
# plt.show()

# # mặt nạ
# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
# img = cv2.imread('Resources/anh1.png')
# # tạo mặt nạ
# mask = np.zeros(img.shape[:2], np.uint8)
# mask[200:400, 200:500] = 255
# masked_img = cv2.bitwise_and(img,img,mask = mask)
# # Tính toán biểu đồ với mặt nạ và không có mặt nạ
# # Kiểm tra đối số thứ ba cho mặt nạ
# hist_full = cv2.calcHist([img],[0],None,[256],[0,256])
# hist_mask = cv2.calcHist([img],[0],mask,[256],[0,256])
# plt.subplot(221), plt.imshow(img, 'gray')
# plt.subplot(222), plt.imshow(mask,'gray')
# plt.subplot(223), plt.imshow(masked_img, 'gray')
# plt.subplot(224), plt.plot(hist_full), plt.plot(hist_mask)
# plt.xlim([0,256])
# plt.show()

# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
# img = cv2.imread('Resources/wiki.jpg',0)
# hist,bins = np.histogram(img.flatten(),256,[0,256])
# cdf = hist.cumsum()
# cdf_normalized = cdf * hist.max()/ cdf.max()
# plt.plot(cdf_normalized, color = 'b')
# plt.hist(img.flatten(),256,[0,256], color = 'r')
# plt.xlim([0,256])
# plt.legend(('cdf','histogram'), loc = 'upper left')
# cv2.imshow('Original', img)
# cv2.waitKey(0)
# plt.show()

#
# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
# img = cv2.imread('Resources/wiki.jpg',0)
# equ = cv2.equalizeHist(img)
# res = np.hstack((img,equ)) #stacking images side-by-side
# cv2.imshow('Res', res)
# cv2.waitKey(0)


# 2D Histogram
# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
# img = cv2.imread('Resources/anh1.png')
# hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
# hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
# # cv2.imshow('out', hist)
# # cv2.waitKey(0)
# plt.imshow(hist, interpolation='nearest')
# plt.show()

# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
# img = cv2.imread('Resources/anh1.png')
# hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
# hist, xbins, ybins = np.histogram2d(h.ravel(),s.ravel(),[180,256],[[0,180],[0,256]])

# BackProjection
import cv2
import numpy as np
roi = cv2.imread('Resources/back.png')
hsv = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
target = cv2.imread('Resources/messi.jpg')
hsvt = cv2.cvtColor(target,cv2.COLOR_BGR2HSV)
# Tính toán biểu đồ đối tượng
roihist = cv2.calcHist([hsv],[0, 1], None, [180, 256], [0, 180, 0, 256] )
#Bình thường tần xuất và áp dụng backprojection
cv2.normalize(roihist,roihist,0,255,cv2.NORM_MINMAX)
dst = cv2.calcBackProject([hsvt],[0,1],roihist,[0,180,0,256],1)
# Now convolute with circular disc
disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
cv2.filter2D(dst,-1,disc,dst)
# threshold and binary AND
ret,thresh = cv2.threshold(dst,50,255,0)
thresh = cv2.merge((thresh,thresh,thresh))
res = cv2.bitwise_and(target,thresh)
res = np.hstack((target,thresh,res))
cv2.imshow('res.jpg',res)
cv2.waitKey(0)