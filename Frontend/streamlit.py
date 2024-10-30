import streamlit as st
import cv2 ## opencv-python
import numpy as np

# title and header
st.title("A Voiceless Voice")
st.header("Webcam Capture with OpenCV")

start_video = st.button("Start Camera")
stop_video = st.button("Stop Camera")

def capture_camera():

    cap = cv2.VideoCapture(0)  
    stframe = st.empty()       

    while cap.isOpened():
        ret, frame = cap.read()  
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        stframe.image(rgb_frame, channels="RGB")
        
        ## This was for grayscale 
        #gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #stframe.image(gray_frame, channels="GRAY")

        if stop_video:
            break

    cap.release()
    cv2.destroyAllWindows()

if start_video:
    capture_camera()

## TO DO:
#st.write("Build Text Display div")
#st.write("Build Voice display div")
#st.write("Build Voice select button")
#st.write("Volume button w/ icon")
