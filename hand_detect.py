import cv2
from cvzone.HandTrackingModule import HandDetector
#roboflow stuff (it just works)
#rf = Roboflow(api_key=apikey)
#project = rf.workspace("david-lee-d0rhs").project("american-sign-language-letters")
#version = project.version(1)
#dataset = version.download("yolov8")
#model = YOLO(dataset)
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
