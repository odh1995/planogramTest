import cv2
import numpy as np

#Load YOLO
net = cv2.dnn.readNet("../Resources/yolo/yolov3.weights", "../Resources/yolo/yolov3.cfg")
classes = []
with open("../Resources/yolo/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layers_names = net.getLayerNames()
outputlayers = [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

#Loading Image
img = cv2.imread("../Resources/yolo/Drawing.png")
img = cv2.resize(img, None, fx=0.4, fy=0.4)
height, width, channel = img.shape

#Detecting object
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True)
net.setInput(blob)
outs =net.forward(outputlayers)

#showing informations on the screen
class_ids = []
confidences = []
boxes = []

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            #object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            #rectangle coordinates
            x = int(center_x - w/2)
            y = int(center_y - h/2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

number_object_detected = len(boxes)
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
font = cv2.FONT_HERSHEY_PLAIN

x_array = []
y_array = []
planogramLabel = []

for i in range(len(boxes)):
    x, y, w, h = boxes[i]
    x_array.append(x)
    y_array.append(y+h)
    label = classes[class_ids[i]]
    color = colors[i]
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    cv2.putText(img, label, (x, y + 30), font, 3, color, 3)
    # print(label)


# print(boxes)

x_array.sort()
y_array.sort()

y_ref = []
prev = 0
numberOfLayer = 0

if not y_ref:
    y1 = y_array[0]
    y_ref.append(y1)
    # give a threshold of 20%
    upperThres = y1 * 1.2
    lowerThres = y1 * 0.8
    numberOfLayer += 1

# loop down the layers
for i in range(len(y_array)):
    for r in range(len(y_ref)):
        if (y_array[i]< upperThres and y_array[i] > lowerThres):
            pass
        else:
            upperThres = y_array[i] * 1.2
            lowerThres = y_array[i] * 0.8
            numberOfLayer += 1
            y_ref.append(y_array[i])
            break

for x1 in range(len(boxes)):
    for x2 in range(len(boxes)):
        if x_array[x1] == boxes[x2][0]:
            planogramLabel.append(classes[class_ids[x2]])


# print(x_array)
print(y_array)
print("Shelve Layer", numberOfLayer)
print(y_ref)
# print(planogramLabel)


cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()