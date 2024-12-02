import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
## import tensorflow as tf (uncomment as needed) 
import pyttsx3 

# language list for dropdown lang-select, dummy languages NOT all langugaes are implemented yet.
LANGUAGES = [
   "English", "Spanish", "Urdu"
]

# Get to tracking! --> this won't work without tf model...
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# model using relative path ~~ Tensor Flow issue must resolve ~~
###model = tf.keras.models.load_model('../action.h5')

## text to yap engine
engine = pyttsx3.init()

# speech speed
engine.setProperty('rate', 150) # Speed
engine.setProperty('volume', 1) # (0.0 to 1.0)

# language dictionary 
language_dict = {
    "English",
    "Spanish",
    "Urdu"
}

### HEADER + TITLE
def display_app_header():
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>A Voiceless Voice</h1>", unsafe_allow_html=True)

def setup_controls():
    with st.container():
        ## camera buttons
        col1, col2 = st.columns(2)
        start_video = col1.button("Start Camera")
        stop_video = col2.button("Stop Camera")     

        selected_language = st.selectbox("Select Language", LANGUAGES, index=LANGUAGES.index("English"), help="Choose the language for ASL interpretation")

    with st.expander("Audio Settings"):
        volume_control = st.checkbox("Mute Audio", value=False)
        volume_level = st.slider("Adjust Volume", min_value=0, max_value=100, value=50) 

    return start_video, stop_video, selected_language, volume_control, volume_level

def capture_camera(stop_video, selected_language, volume_control, volume_level):
    """Captures and displays video feed from the webcam."""
    cap = cv2.VideoCapture(0)  
    FRAME_WINDOW = st.image([], use_container_width=True) 

    # Initialize sentence in session state if not already done **chat gpt'ed will change ltr**
    if "sentence" not in st.session_state:
        st.session_state.sentence = []

    threshold = 0.8  # Set threshold for gesture recognition

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB (Mediapipe expects RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        ## reliant on the model ##
        # Process the image and get the landmarks
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            # draw landmarks on the frame
            for landmarks in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

            # extract keypoints 
            keypoints = []
            for landmark in results.multi_hand_landmarks[0].landmark:
                keypoints.append([landmark.x, landmark.y, landmark.z])
            keypoints = np.array(keypoints).flatten()

            # gesture prediction
            sequence = []
            sequence.append(keypoints)
            sequence = sequence[-30:]

            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                if np.max(res) > threshold:
                    if len(st.session_state.sentence) > 0:
                        if actions[np.argmax(res)] != st.session_state.sentence[-1]:
                            st.session_state.sentence.append(actions[np.argmax(res)])
                    else:
                        st.session_state.sentence.append(actions[np.argmax(res)])

                if len(st.session_state.sentence) > 5:
                    st.session_state.sentence = st.session_state.sentence[-5:]
                st.text(" ".join(st.session_state.sentence))

        # Webcam frame with predictions 
        FRAME_WINDOW.image(frame, channels="BGR", use_container_width=True)

        if stop_video:
            cap.release()
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    display_app_header()
    
    # manage camera feed via session state
    if "camera_started" not in st.session_state:
        st.session_state.camera_started = False
    
    start_video, stop_video, selected_language, volume_control, volume_level = setup_controls()

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center;'>Live Video Feed</h4>", unsafe_allow_html=True)

    # Add control buttons for "Play Translated Speech" button outside of video container
    if st.button("Play Translated Speech"):
        if "sentence" in st.session_state and st.session_state.sentence:
            text_to_speak = " ".join(st.session_state.sentence)
            if not volume_control:  # Check if audio is not muted
                engine.setProperty('volume', volume_level / 100)
                engine.say(text_to_speak)  
                engine.runAndWait()  

    # Start or stop the camera based on button presses
    if start_video and not st.session_state.camera_started:
        st.session_state.camera_started = True
        capture_camera(stop_video, selected_language, volume_control, volume_level)
    elif stop_video and st.session_state.camera_started:
        st.session_state.camera_started = False
        capture_camera(stop_video, selected_language, volume_control, volume_level)

if __name__ == "__main__":
    main()

    ## main issue tf even with venv won't work --> still need to debug
    ## need to fix + confirm audio is in the program and works properly --> in progress
    ## currently on old model. Not Jawad's current model.
