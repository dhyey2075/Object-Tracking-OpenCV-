import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import*

# cap = cv2.VideoCapture(1)
cap = cv2.VideoCapture("./images/cars.mp4")
cap.set(3, 1280)
cap.set(4, 720)
mask = cv2.imread("./images/mask.png")
model = YOLO("./Yolo-Weights/yolov8n")
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]
tracker = Sort(max_age=25, min_hits=3, iou_threshold=0.3)
limits = [400, 297, 673, 297]
cnt = 0
while True:
    _, img = cap.read()
    imgRegion = cv2.bitwise_and(mask, img)

    results = model(imgRegion, stream=True)

    detections = np.empty((0, 5))
    for r in results:
        boxes = r.boxes
        for box in boxes:
            #Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            w, h = x2 - x1, y2 - y1

            #drawing fancy reactangle using cvzone
            #x1 and y1 are coordinated while w and h is the width and height of the reactangle

            #Confidence
            conf = math.ceil(box.conf[0]*100)/100

            #Class
            cls = box.cls[0]
            currentClass = classNames[int(cls)]
            if currentClass == "car":
                cvzone.cornerRect(img, (x1, y1, w, h), l=9)
                # cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 2)
                # cvzone.putTextRect(img, f'{classNames[int(cls)]} {conf}', (max(0, x1), max(35, y1)), scale=0.6, thickness=1, offset=3)
                currArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currArray))


    resultsTracker = tracker.update(detections)
    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cvzone.putTextRect(img, f'{id}', (max(0, x1), max(35, y1)), scale=2, thickness=3,
                           offset=3)
        cx, cy = x1+h//2, y1+h//2
        # cv2.circle(img, (cx, cy), 5, (0, 0, 255), cv2.FILLED)
        if limits[0] < cx <limits[2] and limits[1]-7 < cy < limits[1]+7:
            cnt += 1
        cv2.putText(img, f'Cars Passed: {cnt}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 0, 255), 2)

        print(result)

    cv2.imshow('img', img)
    # cv2.imshow("Region", imgRegion)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()

