#Resizing and cropping

import cv2

img = cv2.imread("Resources/lambo.png")
print(img.shape) # kích thước của ảnh (900, 1200, 3) --> 900: chiều cao, 1200: chiều rộng, 3: số BRG

# thay đổi kích thước của ảnh
imgResize = cv2.resize(img, (300, 200)) # chiều cao:200, chiều rộng:300


# cắt hình ảnh
imgCrop = img[0:500, 200: 900] #lấy dưới dạng ma trận

cv2.imshow("Lambo", img)
cv2.imshow("Resize", imgResize)
cv2.imshow("Cropped", imgCrop)
cv2.waitKey(0)