import streamlit as st
import cv2 ## opencv-python
import numpy as np

# language list for dropdown lang-select
## change to our top 6 languages for demo
LANGUAGES = [
    "albanian", "arabic", "azerbaijani", "basque", "bengali", "bulgarian",
    "catalan", "simplified chinese", "traditional chinese", "czech", "danish",
    "dutch", "english", "esperanto", "estonian", "finnish", "french", "galician",
    "german", "greek", "hebrew", "hindi", "hungarian", "indonesian", "irish",
    "italian", "japanese", "korean", "latvian", "lithuanian", "malay", 
    "norwegian bokmal", "persian", "polish", "portuguese", "romanian", "russian",
    "slovak", "slovenian", "spanish", "swedish", "tagalog", "thai", "turkish", 
    "ukrainian", "urdu"
]

### HEADER + TITLE
def display_app_header():
    #st.title("A Voiceless Voice")
    #st.header("Webcam Capture with OpenCV")
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>A Voiceless Voice</h1>", unsafe_allow_html=True)
    ##st.markdown("<h3 style='text-align: center;'>Webcam Capture with OpenCV</h3>", unsafe_allow_html=True)

def setup_controls():

    with st.container():
        
        ## camera buttons
        col1, col2 = st.columns(2)
        start_video = col1.button("Start Camera")
        stop_video = col2.button("Stop Camera")     

        selected_language = st.selectbox("Select Language", LANGUAGES, index=LANGUAGES.index("english"), help="Choose the language for ASL interpretation")

    with st.expander("Audio Settings"):
        volume_control = st.checkbox("Mute Audio", value=False)
        volume_level = st.slider("Adjust Volume", min_value=0, max_value=100, value=50)  # Default volume level at 50%

    return start_video, stop_video, selected_language, volume_control, volume_level

def capture_camera(stop_video):
    """Captures and displays video feed from the webcam."""
    cap = cv2.VideoCapture(0)  
    stframe = st.empty()
    text_display = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(rgb_frame, channels="RGB")

        detected_text = "Interpreted ASL Text Here"
        text_display.markdown(f"<div style='text-align: center; font-size: 18px;'>Detected Text: {detected_text}</div>", unsafe_allow_html=True)

        if stop_video:
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    display_app_header()
    start_video, stop_video, selected_language, volume_control, volume_level = setup_controls()

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center;'>Live Video Feed</h4>", unsafe_allow_html=True)
    if start_video:
        capture_camera(stop_video)

    # checking to see if my settings are actually working.
    ##st.write(f"Selected Language: {selected_language}")
    ##st.write(f"Mute Audio: {volume_control}")
    ##st.write(f"Volume Level: {volume_level}")

if __name__ == "__main__":
    main()
