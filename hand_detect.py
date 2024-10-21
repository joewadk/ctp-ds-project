import cv2
from cvzone.HandTrackingModule import HandDetector
from dotenv import load_dotenv
from roboflow import Roboflow
from ultralytics import YOLO
import os
load_dotenv() #environment variables, make a .env file with your ROBOFLOW_API_KEY


#roboflow stuff (it just works)
apikey = os.getenv("ROBOFLOW_API_KEY")
rf = Roboflow(api_key=apikey)

#uncomment these and move the dataset to /data
'''
project = rf.workspace("david-lee-d0rhs").project("american-sign-language-letters")
version = project.version(6)
dataset = version.download("yolov8")
model = YOLO(dataset)
'''
# if this isnt ur first time running, then do this:

dataset_path = "data/American-Sign-Language-Letters-6/data.yaml"
model=YOLO(dataset_path) #fit model with dataset
results = model.train(data=dataset_path, epochs=3)  # train the model

print(results) #print results




#camera stuff
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Failed to open camera")
    exit()
detector=HandDetector(maxHands=2)

while True:
    success, img = cap.read()
    hands,img=detector.findHands(img)
    if not success:
        print("Failed to capture image")
        continue
    cv2.imshow("Camera", img)
    cv2.waitKey(1)
