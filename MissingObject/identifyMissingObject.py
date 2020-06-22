import cv2
import numpy as np

PATH_COFFEE = "../Resources/Coffee.png"
PATH_MISSING_COFFEE = "../Resources/missingCoffee.jpg"

img1 = cv2.imread(PATH_COFFEE,)
img2 = cv2.imread(PATH_MISSING_COFFEE)

img1 = cv2.resize(img1, (1920, 1440))
img2 = cv2.resize(img2, (1920, 1440))

#Identify the differences
difference = cv2.subtract(img2, img1)
overlap = cv2.addWeighted(difference, 0.7, img2, 0.3, 10)

cv2.imshow("overlap", cv2.resize(overlap, None, fx=0.4, fy=0.4))
cv2.imshow("difference", cv2.resize(difference, None, fx=0.4, fy=0.4))
cv2.imshow("missing", cv2.resize(img2, None, fx=0.4, fy=0.4))

cv2.waitKey(0)