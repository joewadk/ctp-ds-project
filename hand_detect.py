import cv2
from cvzone.HandTrackingModule import HandDetector
from ultralytics import YOLO # Load a pretrained YOLO model model = YOLO("yolov8n.pt")
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Failed to open camera")
    exit()
detector = HandDetector(maxHands=2)

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image")
        continue
    hands, img = detector.findHands(img)
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()