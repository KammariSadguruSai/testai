import cv2
import numpy as np

def detect_objects(image_path):
    # Load YOLOv3 model and configuration
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    classes = open("coco.names").read().strip().split("\n")

    # Load image
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)

    # Perform object detection
    detections = net.forward(output_layers)
    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Draw bounding box and label
                center_x, center_y, w, h = (
                    int(obj[0] * width),
                    int(obj[1] * height),
                    int(obj[2] * width),
                    int(obj[3] * height),
                )
                x, y = int(center_x - w / 2), int(center_y - h / 2)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    image,
                    classes[class_id],
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )
    return image
