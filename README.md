# Object-detection-using-web-camera

<H3>Name: SANJEEV RAJ.S</H3>
<H3>Register no: 212223220096</H3>
<H3>Date: 30.9.2025 </H3>

# AIM:
To perform real-time object detection using a trained YOLO v4 model through your laptop camera.

# PROGRAM:
```python
import cv2
import numpy as np
import time
from IPython.display import display, clear_output
import matplotlib.pyplot as plt

# Load YOLOv4 network
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")

# Load COCO class labels
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Output layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Colors for each class
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Set up webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open webcam")
    exit()

prev_time = 0
last_frame = None  # store last frame

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        frame = cv2.resize(frame, (640, 480))
        height, width, _ = frame.shape

        # Prepare image for YOLOv4
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        outputs = net.forward(output_layers)

        boxes, confidences, class_ids = [], [], []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = max(0, int(center_x - w / 2))
                    y = max(0, int(center_y - h / 2))
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = confidences[i]
                color = colors[class_ids[i]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # FPS calculation
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Store last frame
        last_frame = frame.copy()

        # Convert BGR to RGB for display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display frame inline
        clear_output(wait=True)
        plt.imshow(frame_rgb)
        plt.axis('off')
        display(plt.gcf())

        # Press "Stop" or interrupt kernel to stop
        time.sleep(0.03)

except KeyboardInterrupt:
    print("Stopped by user")

finally:
    cap.release()
    plt.close()

    # Show last frame
    if last_frame is not None:
        last_rgb = cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB)
        plt.imshow(last_rgb)
        plt.axis('off')
        plt.title("Last Captured Frame")
        plt.show()


```

# OUTPUT:

<img width="621" height="495" alt="image" src="https://github.com/user-attachments/assets/2dde7531-3e62-498d-9cd5-50feaf3d9f1a" />

 
