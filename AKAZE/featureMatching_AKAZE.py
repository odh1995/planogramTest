import cv2
import numpy as np
from config import *

img1 = cv2.imread(PATH_ACTUAL)
img2 = cv2.imread(PATH_PLANOGRAM)

img1 = cv2.resize(img1, (1920, 1440))
img2 = cv2.resize(img2, (1920, 1440))

#ORB Detector
akaze = cv2.AKAZE_create()
kp_1, des1 = akaze.detectAndCompute(img1, None)
kp_2, des2 = akaze.detectAndCompute(img2, None)

#Brute Force
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x:x.distance)

print("Amount of matches: ", matches)

matching_result = cv2.drawMatches(img1, kp_1, img2, kp_2, matches[:50], None) #top 50 matches

# cv2.imshow("img1",img1)
# cv2.imshow("img2",img2)
cv2.imshow("matching result", cv2.resize(matching_result, None, fx=0.4, fy=0.4))
cv2.imwrite("../output/AKAZE_TOP_50_MATCHES.png", matching_result)
cv2.waitKey(0)