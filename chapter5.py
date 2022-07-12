#warp prespective
# Cắt và làm phẳng ảnh
import  cv2
import numpy as np

img=cv2.imread("Resources/cards.png")

width, height = 135, 220 # chiều rộng, cao của output
pts1 = np.float32([[100, 16], [192, 17], [192, 151], [97, 143]]) # 4 góc của ảnh cần cắt
pts2 = np.float32([[0,0], [width, 0], [width, height], [0, height]]) # vị trí cuả 4 góc ứng với pts1

matrix  =cv2.getPerspectiveTransform(pts1, pts2)

imgOut=cv2.warpPerspective(img, matrix, (width, height))

cv2.imshow("Image", img)
cv2.imshow("Output", imgOut)

cv2.waitKey(0)