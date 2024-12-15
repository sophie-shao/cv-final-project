import cv2
import numpy as np

# Load YOLO model
def load_yolo_model():
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")  # Replace with actual YOLO model weights and config
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, output_layers

# Load class labels (e.g., cat, person, food)
def load_classes():
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    return classes

# Detect objects with YOLO
def detect_objects(image, net, output_layers, classes):
    height, width, channels = image.shape
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for out in outputs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    return boxes, confidences, class_ids, indices

# Apply the mask and compute the average depth for detected objects
def get_subject_depth(image, boxes, depth_map):
    depths = []
    for (x, y, w, h) in boxes:
        # Extract the region corresponding to the detected object
        object_region = depth_map[y:y + h, x:x + w]
        # Compute the average depth of the object
        avg_depth = np.mean(object_region)
        depths.append(avg_depth)
    return np.mean(depths)  # Return the average depth of the subject
