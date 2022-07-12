# Erosion
#
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# img = cv2.imread('Resources/j.png')
# kernel = np.ones((5,5),np.uint8)
# # erosion = cv2.erode(img,kernel,iterations = 1)
# dilation = cv2.dilate(img,kernel,iterations = 1)
# plt.subplot(121),plt.imshow(img),plt.title('Input')
# plt.subplot(122),plt.imshow(dilation),plt.title('Output')
# plt.show()

# Opening
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# img = cv2.imread('Resources/j.png')
# kernel = np.ones((9,9),np.uint8)
# # opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
# # closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
# # gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
# # tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
# blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
# plt.subplot(121),plt.imshow(img),plt.title('Input')
# plt.subplot(122),plt.imshow(blackhat),plt.title('Output')
# plt.show()

# Rectangular Kernel
# import cv2
# print(cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)))

# Elliptical Kernel
import cv2
# print(cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))

# Cross-shaped Kernel
import cv2
print(cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5)))