# import cv2
# img = cv2.imread("Resources/anh1.png")
# cv2.imshow('out0', img)
# img = cv2.pyrDown(img)
# cv2.imshow('out1', img)
# img = cv2.pyrDown(img)
# cv2.imshow('out2', img)
# img = cv2.pyrDown(img)
# cv2.imshow('out3', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# import cv2
# img = cv2.imread("Resources/logo1.png")
# cv2.imshow('out0', img)
# img = cv2.pyrUp(img)
# cv2.imshow('out1', img)
# img = cv2.pyrUp(img)
# cv2.imshow('out2', img)
# # img = cv2.pyrUp(img)
# # cv2.imshow('out3', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# laplacian
# import cv2
# img = cv2.imread("Resources/logo1.png")
# img = cv2.Canny(img,100,200)
# cv2.imshow('out0', img)
# img = cv2.pyrUp(img)
# cv2.imshow('out1', img)
# img = cv2.pyrUp(img)
# cv2.imshow('out2', img)
# # img = cv2.pyrUp(img)
# # cv2.imshow('out3', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Trộn 2 hình ảnh sd kim tự tháp
import cv2
import numpy as np
A = cv2.imread('Resources/orange.jpg')
B = cv2.imread('Resources/apple.jpg')

# Tạo kim tự tháp gauss cho A
G = A.copy()
gpA = [G]
for i in range(6):
    G = cv2.pyrDown(G)
    gpA.append(G)

# Tạo kim tự tháp gauss cho B
G = B.copy()
gpB = [G]
for i in range(6):
    G = cv2.pyrDown(G)
    gpB.append(G)

# Tạo kim tự tháp Laplacian cho A
lpA = [gpA[5]]
for i in range(5,0,-1):
    GE = cv2.pyrUp(gpA[i])
    L = cv2.subtract(gpA[i-1],GE)
    lpA.append(L)

# Tạo kim tự tháp Laplacian cho B
lpB = [gpB[5]]
for i in range(5,0,-1):
    GE = cv2.pyrUp(gpB[i])
    L = cv2.subtract(gpB[i-1],GE)
    lpB.append(L)

# Bây giờ thêm nửa trái và phải của hình ảnh ở mỗi cấp
LS = []
for la,lb in zip(lpA,lpB):
    rows,cols,dpt = la.shape
    ls = np.hstack((la[:,0:cols//2], lb[:,cols//2:]))
    LS.append(ls)

# Bây giờ lại tái tạo
ls_ = LS[0]
for i in range(1,6):
    ls_ = cv2.pyrUp(ls_)
    ls_ = cv2.add(ls_, LS[i])

# Hình ảnh kết nối trực tiếp với nhau
real = np.hstack((A[:,:cols//2],B[:,cols//2:]))
cv2.imshow('Pyramid_blending',ls_)
cv2.imshow('Direct_blending',real)
cv2.waitKey(0)
cv2.destroyAllWindows()