import cv2
import numpy as np
from config import *

IMG_SIZE = 1000

image1 = cv2.imread(PATH_ACTUAL)
image2 = cv2.imread(PATH_PLANOGRAM)

image1 = np.array(image1)
image2 = np.array(image2)

image1 = cv2.resize(image1,(1920, 1440))
image2 = cv2.resize(image2,(1920, 1440))

#1. Check if images are equal
if image1.shape == image2.shape:
    print("They both have same size and channels")
    difference = cv2.subtract(image1, image2)
    b, g, r = cv2.split(difference)

    # cv2.imshow("difference", r)

    if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
        print("The images are completely equal")
    else:
        print("Images are not equal")

#2. Check the similarity
surf = cv2.xfeatures2d.SURF_create()
kp_1, desc_1 = surf.detectAndCompute(image1, None)
kp_2, desc_2 = surf.detectAndCompute(image2, None)

print("Keypoints 1st image: " + str(len(kp_1)))
print("Keypoints 2nd image: " + str(len(kp_2)))

index_params = dict(algorithm=0, trees=5)
search_params = dict()
flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(desc_1, desc_2, k=2)

good_points = []
for m, n in matches:
    if m.distance < 0.5*n.distance:
        good_points.append(m)


print("Good Matches: " + str(len(good_points)))

if(len(kp_1) > len(kp_2)):
    similarities = (len(good_points) / len(kp_2)) * 100
else:
    similarities = (len(good_points) / len(kp_1)) * 100

print("Similarity level: {}%".format(str(similarities)))

result = cv2.drawMatches(image1, kp_1, image2, kp_2, good_points, None)
cv2.imshow("result", cv2.resize(result, None, fx=0.4, fy=0.4))
cv2.imwrite("../output/SURF_0.5_DISTANCE_MATCHES.png", result)

# print(b)
# print(g)
# print(r)

# cv2.imshow("Original", original)
# cv2.imshow("Duplicate", duplicate)

cv2.waitKey(0)


