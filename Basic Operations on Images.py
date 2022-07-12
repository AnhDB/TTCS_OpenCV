import cv2

img = cv2.imread("Resources/anh.png")
# loại dữ liệu ảnh
# print(img.dtype)
# tổng số pixel trong ảnh
# print(img.size)
# px = img[100,100]
# print(px)
# # accessing only blue pixel
# blue = img[100,100,0]
# # print (blue)
#
# img[100,100] = [255,255,255]
# print(img[100,100])

# # accessing RED value
# print(img.item(10,10,2))
# # modifying RED value
# img.itemset((10,10,2),100)
# print(img.item(10,10,2))
# hình dạng ảnh
# print (img.shape)

#sao chép quả bóng và copy sang vị trí khác
# img = cv2.imread("Resources/ball.png")
# ball = img[580:780, 140:345]
# img[200:400, 120:325] = ball
# cv2.imshow('out',img)
# cv2.waitKey()

#tách ảnh + gộp ảnh
# b,g,r = cv2.split(img)
# img = cv2.merge((b,g,r))
#
# b = img[:,:,0]
#
# img[:,:,2] = 0

# #tạo biên của ảnh
import cv2
import numpy as np
from matplotlib import pyplot as plt
BLUE = [255,0,0]
img1 = cv2.imread('Resources/opencv_logo.jpg')
replicate = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_REPLICATE)
reflect = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_REFLECT)
reflect101 = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_REFLECT_101)
wrap = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_WRAP)
constant= cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_CONSTANT,value=BLUE)
plt.subplot(231),plt.imshow(img1,'gray'),plt.title('ORIGINAL')
plt.subplot(232),plt.imshow(replicate,'gray'),plt.title('REPLICATE')
plt.subplot(233),plt.imshow(reflect,'gray'),plt.title('REFLECT')
plt.subplot(234),plt.imshow(reflect101,'gray'),plt.title('REFLECT_101')
plt.subplot(235),plt.imshow(wrap,'gray'),plt.title('WRAP')
plt.subplot(236),plt.imshow(constant,'gray'),plt.title('CONSTANT')
plt.show()
# import cv2
# import numpy as np
# x = np.uint8([250])
# y = np.uint8([10])
# print (cv2.add(x,y)) # 250+10 = 260 => 255
#
# print (x+y)  # 250+10 = 260 % 256 = 4

# Trộn hình ảnh
# import cv2
# img1 = cv2.imread('Resources/logo.png')
# img2 = cv2.imread('Resources/opencv_logo1.jpg')
# dst = cv2.addWeighted(img1,0.8,img2,0.2,0)
# cv2.imshow('img1', img1)
# cv2.imshow('img2', img2)
# cv2.imshow('dst',dst)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # bitwise Operations
# import cv2
# # Load two images
# img1 = cv2.imread('Resources/anh.png')
# img2 = cv2.imread('Resources/opencv_logo.jpg')
# # Đặt logo ở góc trên cùng bên trái --> tạo ROI
# rows,cols,channels = img2.shape
# roi = img1[0:rows, 0:cols]
# # Bây giờ tạo một mặt nạ logo và tạo mặt nạ nghịch đảo
# img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
# ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
# mask_inv = cv2.bitwise_not(mask)
# # màu đen của logo trong roi
# img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
# # Chỉ chọn vùng logo từ hình ảnh biểu trưng.
# img2_fg = cv2.bitwise_and(img2,img2,mask = mask)
# # Đặt biểu tượng roi và sửa đổi hình ảnh chính
# dst = cv2.add(img1_bg,img2_fg)
# img1[0:rows, 0:cols ] = dst
# cv2.imshow('res',img1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# tính hiệu suất
# import cv2
# img1 = cv2.imread('Resources/anh.png')
# e1 = cv2.getTickCount()
# for i in range(5,49,2):
#     img1 = cv2.medianBlur(img1,i)
# e2 = cv2.getTickCount()
# t = (e2 - e1)/cv2.getTickFrequency()
# print(t)