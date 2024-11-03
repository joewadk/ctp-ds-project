import cv2
from cvzone.HandTrackingModule import HandDetector
from dotenv import load_dotenv
from roboflow import Roboflow
from ultralytics import YOLO
import os
import mediapipe as mp
load_dotenv() #environment variables, make a .env file with your ROBOFLOW_API_KEY

import torch

#roboflow stuff (it just works)
apikey = os.getenv("ROBOFLOW_API_KEY")
rf = Roboflow(api_key=apikey)


dataset_path = "data/ASL-Detection.v1i.yolov8"
#model = YOLO("runs/detect/train/weights/best.pt") #calling my model i made via CLI  #100 epochs <- super overfitting
#model = YOLO("runs/detect/train4/weights/best.pt") #10 epochs <- underfitting probably
#model = YOLO("runs/detect/train3/weights/best.pt") #30 epochs <- overfitting still
#model = YOLO("runs/detect/train6/weights/best.pt")
#model = YOLO("runs/detect/train7/weights/best.pt") #test with only 1 epoch, clearly underfitting
#model = YOLO("runs/detect/train11/weights/best.pt") #new model
device = torch.device('cpu')
model = YOLO("runs/detect/train24/weights/best.pt").to(device)
#camera stuff

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Failed to open camera")
    exit()

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image")
        continue

    # Convert the BGR image to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process the image and detect hands
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get bounding box coordinates
            x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * img.shape[1])
            y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * img.shape[0])
            x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * img.shape[1])
            y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * img.shape[0])

            # Ensure the bounding box is within the image dimensions
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(img.shape[1], x_max)
            y_max = min(img.shape[0], y_max)

            # Crop the hand region
            hand_img = img[y_min:y_max, x_min:x_max]

            # Resize the cropped hand image to the required dimensions
            hand_img_resized = cv2.resize(hand_img, (640, 640))

            # Convert the resized hand image to a tensor and add batch dimension
            hand_img_tensor = torch.from_numpy(hand_img_resized).permute(2, 0, 1).unsqueeze(0).float().to(device)

            # Use YOLO model for detection on the cropped hand region with lower confidence threshold
            yolo_results = model(hand_img_tensor, conf=0.25)  # Set confidence threshold to 0.25

            # Debug: Print YOLO results
            print(yolo_results)

            # Draw detection results on the original image
            for result in yolo_results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf)  # Convert tensor to float
                    cls = int(box.cls)  # Get the class index
                    label = model.names[cls]  # Get the class label
                    cv2.rectangle(img, (x_min + x1, y_min + y1), (x_min + x2, y_min + y2), (0, 255, 0), 2)
                    cv2.putText(img, f'{label} {conf:.2f}', (x_min + x1, y_min + y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the image
    cv2.imshow("Camera", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()