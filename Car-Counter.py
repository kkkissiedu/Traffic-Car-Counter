from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *
import imageio
import numpy as np

#_--------------For Webcam-----------------
#cap = cv2.VideoCapture(0)     #Object representing video capture device, 0 for internal webcam, 1 for external webcam
#Setting the resolution of the webcam
#cap.set(3, 1280)  #Set width
#cap.set(4, 720)  #Set height

#_--------------For Video File-------------
cap = cv2.VideoCapture('cars.mp4')

if not cap.isOpened():
    print("Error: Could not open video file 'cars.mp4'.")
    exit()

model = YOLO("yolov8l.pt")     #Loading pre-trained YOLOv8 'nano' model


classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dot", "pizza", "donut", "cake", "chair", "sofa", "potted plant", "bed",
              "dining table", "toilet", "tv monitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair dryer", "toothbrush"
              ]


mask = cv2.imread("mask.png")

frames = []
fps = int(cap.get(cv2.CAP_PROP_FPS))
print(f"Video FPS: {fps}")

#Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

limits = [400, 297, 673, 297]
totalCount = []


while True:     #Infinite loop to continuously capture frames from the webcam
    success, img = cap.read()  #success will be 1 if frame was captured successfully, 0 otherwise, and the loop will end. Img will be a numpy array of the captured frame
    
    if not success:
        break

    imgRegion = cv2.bitwise_and(img, mask)
    imgGraphics = cv2.imread("graphics.png", cv2.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(img, imgGraphics, (0, 0))

    results = model(imgRegion, stream=True) #Perform inference on the captured frame, stream=True allows for real-time processing
    
    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes     #Return a list of detected bounding boxes

        for box in boxes:
            
            #Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)     #Convert coordinates from tensors to integers
            # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            w, h = x2 - x1, y2 - y1
            
            #Confidence
            conf = math.ceil((box.conf[0]*100))/100
            
            #Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass in ["car", "truck", "bus", "motorbike"] and conf > 0.3:
                # cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0,x1), max(35, y1)), scale = 0.6, thickness = 1, offset =3)
                # cvzone.cornerRect(img, (x1, y1, w, h), l = 9, rt = 5)
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections)
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)

    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        print(result)

        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(img, f'{int(id)}', (max(0, x1), max(35, y1)), scale=2, thickness=3, offset=10)

        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:
            if totalCount.count(id) == 0:
                totalCount.append(id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)
                
    # cvzone.putTextRect(img, f'Count:{len(totalCount)}', (50, 50))
    cv2.putText(img, str(len(totalCount)), (255, 100), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 255), 8)

    height, width, _ = img.shape
    new_width = 640
    new_height = int((new_width / width) * height)
    img_resized = cv2.resize(img, (new_width, new_height))

    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    frames.append(img_rgb)

imageio.mimsave('output_cars.gif', frames, fps=15)
print("GIF saved successfully as 'output_cars.gif'")

cap.release()