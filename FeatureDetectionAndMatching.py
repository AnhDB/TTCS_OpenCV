# #Hari
# # Importing the libraries
# import cv2
# import numpy as np
#
# # Reading the image and converting the image to B/W
# image = cv2.imread('Resources/sach.jpg')
# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# gray_image = np.float32(gray_image)
#
# # Applying the function
# dst = cv2.cornerHarris(gray_image, blockSize=2, ksize=3, k=0.04)
#
# # Giãn ra để đánh dấu các góc
# dst = cv2.dilate(dst, None)
# image[dst > 0.01 * dst.max()] = [0, 255, 0]
#
# cv2.imshow('haris_corner', image)
# cv2.waitKey()


# # Shi - Tomasi
# # Importing the libraries
# import cv2
# import numpy as np
#
# # Reading the image and converting into B?W
# image = cv2.imread("Resources/sach.jpg")
# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
# # Applying the function
# corners = cv2.goodFeaturesToTrack(
#     gray_image, maxCorners=50, qualityLevel=0.02, minDistance=20)
# corners = np.float32(corners)
#
# for item in corners:
#     x, y = item[0]
#     x = int(x)
#     y = int(y)
#     cv2.circle(image, (x, y), 6, (0, 255, 0), -1)
#
# # Showing the image
# cv2.imshow('good_features', image)
# cv2.waitKey()

# # SIFT
# # Importing the libraries
# import cv2
# # Reading the image and converting into B/W
# image = cv2.imread('Resources/sach.jpg')
# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# # Applying the function
# sift = cv2.xfeatures2d.SIFT_create()
# kp, des = sift.detectAndCompute(gray_image, None)
# # Applying the function
# kp_image = cv2.drawKeypoints(image, kp, None, color=(
#     0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# cv2.imshow('SIFT', kp_image)
# cv2.waitKey()

# # FAST
# # Importing the libraries
# import cv2
# # Reading the image and converting into B/W
# image = cv2.imread('Resources/sach.jpg')
# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# # Applying the function
# fast = cv2.FastFeatureDetector_create()
# fast.setNonmaxSuppression(False)
# # Drawing the keypoints
# kp = fast.detect(gray_image, None)
# kp_image = cv2.drawKeypoints(image, kp, None, color=(0, 255, 0))
# cv2.imshow('FAST', kp_image)
# cv2.waitKey()

# ORB
# Importing the libraries
import cv2
# Reading the image and converting into B/W
image = cv2.imread('Resources/sach.jpg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Applying the function
orb = cv2.ORB_create(nfeatures=2000)
kp, des = orb.detectAndCompute(gray_image, None)
# Drawing the keypoints
kp_image = cv2.drawKeypoints(image, kp, None, color=(0, 255, 0), flags=0)
cv2.imshow('ORB', kp_image)
cv2.waitKey()