import cv2
import numpy as np

PATH_ACTUAL = "../Resources/actualPic.jpg"
PATH_PLANOGRAM = "../Resources/planogram.png"

PATH_COFFEE = "../Resources/Coffee.png"
PATH_UPSIDE_DOWN = "../Resources/upsideDown.jpg"

actualPic = cv2.imread(PATH_ACTUAL)
planogram = cv2.imread(PATH_PLANOGRAM)

actualPic = cv2.resize(actualPic, (1280, 720))
planogram = cv2.resize(planogram, (1280, 720))

actualPicGray = np.array(cv2.cvtColor(actualPic, cv2.COLOR_BGR2GRAY))
planogramGray = np.array(cv2.cvtColor(planogram, cv2.COLOR_BGR2GRAY))



# print(actualPic.shape)
# print(planogram.shape)

#Initiate ORB
orb = cv2.ORB_create(1000)

kp1, des1 = orb.detectAndCompute(actualPic, None)
kp2, des2 = orb.detectAndCompute(planogram, None)

matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)

#match the descriptor
matches = matcher.match(des1, des2, None)

matches = sorted(matches, key=lambda x:x.distance)

points1 = np.zeros((len(matches), 2), dtype=np.float32)
points2 = np.zeros((len(matches), 2), dtype=np.float32)

print(len(matches))

#before applying homography
for i, match in enumerate(matches):
    points1[i, :] = kp1[match.queryIdx].pt
    points2[i, :] = kp2[match.queryIdx].pt

h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

#use homography
height, width, channel = planogram.shape

im1Reg = cv2.warpPerspective(actualPic, h, (width, height))

img3 = cv2.drawMatches(actualPicGray, kp1,planogramGray, kp2, matches[:100], np.array([]))

cv2.imshow("Registered Image", cv2.resize(im1Reg, None, fx=0.4, fy=0.4))
cv2.imshow("Keypoint Images", cv2.resize(img3, None, fx=0.4, fy=0.4))
cv2.waitKey(0)




# cv2.imshow("actual", actualPic)
# cv2.imshow("planogram", planogram)
cv2.waitKey(0)

