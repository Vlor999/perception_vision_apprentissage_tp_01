import cv2
import numpy as np
import sys

# Load ONNX model
net = cv2.dnn.readNetFromONNX("yolov3.onnx")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Load class names
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

wanted_classes = ["motorbike", "airplane", "person"]  # filter

# Load image
img = cv2.imread("image.jpg")
if img is None:
    sys.exit()
height, width = img.shape[:2]

# Prepare blob
blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)

# Forward pass
outs = net.forward()[0]  # remove batch dim

# Parse detections
boxes, confidences, class_ids = [], [], []
for detection in outs:
    scores = detection[5:]
    class_id = int(np.argmax(scores))
    confidence = float(scores[class_id])
    print(classes[class_id])
    if confidence > 0.5 and classes[class_id] in wanted_classes:
        cx, cy, w, h = detection[0:4]
        x = int((cx - w / 2) * width)
        y = int((cy - h / 2) * height)
        w = int(w * width)
        h = int(h * height)
        boxes.append([x, y, w, h])
        confidences.append(confidence)
        class_ids.append(class_id)

# Non-max suppression
indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# Draw boxes
for i in indices.flatten():
    x, y, w, h = boxes[i]
    label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

cv2.imshow("Detections", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
