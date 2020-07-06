import cv2
import numpy as np

# Load YOLO
net = cv2.dnn.readNet("../Resources/yolo/yolov3.weights", "../Resources/yolo/yolov3.cfg")
classes = []
with open("../Resources/yolo/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layers_names = net.getLayerNames()
outputlayers = [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Loading Image
img = cv2.imread("../Resources/yolo/yoloLayeredImage.png")
img = cv2.resize(img, None, fx=0.4, fy=0.4)
height, width, channel = img.shape

# Detecting object
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True)
net.setInput(blob)
outs = net.forward(outputlayers)

# showing informations on the screen
class_ids = []
confidences = []
boxes = []

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            # object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

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
    y_array.append(y)
    label = classes[class_ids[i]]
    color = colors[i]
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    cv2.putText(img, label, (x, y + 30), font, 3, color, 3)
    # print(label)

print("boxes: ", boxes)
print("No. of boxes: ", len(boxes))

x_array.sort()
y_array.sort()

y_ref = []
prev = 0
numberOfLayer = 0

x_buffer = []
final_x = []
roundI = 0
organised = []
unorganisedLabel = []
organisedLabel = []
finalLabel = []


# sort the horizontal value from left to right
def LRSort():
    x_buffer.sort()
    for i in range(len(x_buffer)):
        for r in range(len(boxes)):
            if x_buffer[i] == boxes[r][0]:
                organisedLabel.append(classes[class_ids[r]])


# For the first time, take reference value
if not y_ref:
    y1 = y_array[0]
    y_ref.append(y1)
    # give a threshold of 20%
    upperThres = (y1 + 200) * 1.3
    lowerThres = (y1 + 200) * 0.7
    numberOfLayer += 1

# loop down the layers
for i in range(len(y_array)):
    for r in range(len(y_ref)):
        if ((y_array[i] + 200) < upperThres and (y_array[i] + 200) > lowerThres):
            # loop through the horizontal item
            for t in range(len(boxes)):
                if boxes[t][1] == y_array[i]:
                    x_buffer.append(boxes[t][0])
                    unorganisedLabel.append(classes[class_ids[t]])
                    break
        # subsequent layer
        else:
            LRSort()
            finalLabel.append(organisedLabel)
            upperThres = (y_array[i] + 200) * 1.3
            lowerThres = (y_array[i] + 200) * 0.7
            numberOfLayer += 1
            y_ref.append(y_array[i])
            break

print("X: ", x_array)
print("Y: ", y_array)
print("Y Reference: ", y_ref)
print("Shelve Layer", numberOfLayer)
print("Items: ", planogramLabel)
print("X Buffer ", x_buffer)
print("UnorganisedLabel: ", unorganisedLabel)
print("Final Label", finalLabel)

# print(planogramLabel)

print("=======================================================")
# This is an improved algorithm maybe?
# First don't assume the layers?
sorted_y_bounding_boxes = sorted(boxes, key=lambda box_y: box_y[1])
layers_bounding_boxes = []
current_bounding_box = {"min_x": 0, "min_y": 0, "max_x": 0, "max_y": 0}

for box in sorted_y_bounding_boxes:
    x, y, width, height = box

    if current_bounding_box["min_x"] == 0 and current_bounding_box["min_y"] == 0:
        # This is first registration
        current_bounding_box["min_x"] = x
        current_bounding_box["max_x"] = x + width
        current_bounding_box["min_y"] = y
        current_bounding_box["max_y"] = y + height
    else:
        # If they are in the same row, then most likely the y value will collide between the range.
        if y < current_bounding_box["max_y"]:
            current_bounding_box["min_x"] = min(current_bounding_box["min_x"], x)
            current_bounding_box["max_x"] = max(current_bounding_box["max_x"], x + width)
            current_bounding_box["min_y"] = min(current_bounding_box["min_y"], y)
            current_bounding_box["max_y"] = max(current_bounding_box["max_y"], y + height)
        else:
            cv2.rectangle(img, (current_bounding_box["min_x"], current_bounding_box["min_y"]), (current_bounding_box["max_x"], current_bounding_box["max_y"]), (255, 0, 0), 2)
            layers_bounding_boxes.append(current_bounding_box)
            current_bounding_box = {"min_x": x, "min_y": y + width, "max_x": x + width, "max_y": y + height}

cv2.rectangle(img, (current_bounding_box["min_x"], current_bounding_box["min_y"]), (current_bounding_box["max_x"], current_bounding_box["max_y"]), (255, 0, 0), 2)
layers_bounding_boxes.append(current_bounding_box)
print("Number of Layer:", len(layers_bounding_boxes))
print("Layer Bounding Boxes:", layers_bounding_boxes)

shelf_row = [[] for _ in range(len(layers_bounding_boxes))]

for i in range(len(boxes)):
    x, y, width, height = boxes[i]
    object_label = classes[class_ids[i]]

    for layer in range(len(layers_bounding_boxes)):
        current_bounding_box = layers_bounding_boxes[layer]

        if current_bounding_box["min_x"] <= x <= current_bounding_box["max_x"] and \
                current_bounding_box["min_y"] <= y <= current_bounding_box["max_y"]:
            # This is in this layer
            shelf_row[layer].append(object_label)
            break

shelf_row = [sorted(row, key=lambda row_key: row_key[0]) for row in shelf_row]
print("Shelf:", shelf_row)



cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
