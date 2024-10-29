import cv2
from cvzone.HandTrackingModule import HandDetector
from dotenv import load_dotenv
from roboflow import Roboflow
from ultralytics import YOLO
import os
import mediapipe as mp
load_dotenv() #environment variables, make a .env file with your ROBOFLOW_API_KEY


#roboflow stuff (it just works)
apikey = os.getenv("ROBOFLOW_API_KEY")
rf = Roboflow(api_key=apikey)
project = rf.workspace("david-lee-d0rhs").project("american-sign-language-letters")
version = project.version(6)
dataset = version.download("yolov8")
model = YOLO(dataset)
