import streamlit as st
import cv2
import numpy as np
from gtts import gTTS
import os
import openai
import tensorflow as tf ## still need to debug
from dotenv import load_dotenv
from our_Translate import translate_text
#from translate import translate_text, download_languages_for_demo ## still need to implement functions
#from classifier import classifier ## implement function
#from capture_sign import capture_sign ## implement function

#download_languages_for_demo() ## from translate

##classifier()
##capture_sign()
##translate_text()


### Environment variables --> loading + initializing api key ###
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

## load my api key + initialize it ##
if not OPENAI_API_KEY:
    st.error("API key not found! Check for a valid .env file with 'OPENAI_API_KEY'.")

openai.api_key = OPENAI_API_KEY

# Languages # ~~ from mckenzie's lang for demo function
LANGUAGES = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "Japanese": "ja",
    "German": "de",
    "Chinese": "zh",
    "Hindi": "hi",
    "Arabic": "ar",
    "Russian": "ru",
    "Urdu": "ur",
    "Bengali": "bn",
}

# header #
def display_app_header():
    """Displays the app header."""
    st.markdown(
        "<h1 style='text-align: center; color: #4CAF50;'>A Voiceless Voice</h1>",
        unsafe_allow_html=True,
    )


# controls #
def setup_controls():
    """Sets up the user interface controls."""

    ## camera buttons ## 
    with st.container():
        col1, col2 = st.columns(2)
        start_video = col1.button("Start Camera")
        stop_video = col2.button("Stop Camera")
        to_code=''
    ## language select ## 
        selected_language = st.selectbox(
            "Select Language",
            list(LANGUAGES.keys()),
            index=0,
            help="Choose the language for ASL interpretation",
        )
        #print(selected_language)
        to_code=LANGUAGES[selected_language]
    ## audio settings ## 
    with st.expander("Audio Settings"):
        volume_control = st.checkbox("Mute Audio", value=False)
        return_audio = st.checkbox("Return Audio Response", value=True)

    return start_video, stop_video, selected_language, volume_control, return_audio,to_code

### grabbed from openAI documentation ### 
def generate_audio_with_openai(text, selected_language):
    """Generates TTS audio using OpenAI's API and gTTS for speech synthesis."""
    try:
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", ## didn't work with voice param
            messages=[
                {"role": "system", "content": f"You are a translator for {selected_language}."},
                {"role": "user", "content": f"Translate this text to {selected_language}: {text}"},
            ]
        )
        translated_text = response['choices'][0]['message']['content'].strip()

        st.text(f"[Translated Text in {selected_language}]: {translated_text}")

        ## Text to Speech via gTTS --> this has to be through translate.py##
        tts_language_code = LANGUAGES.get(selected_language, "en")  
        tts = gTTS(text=translated_text, lang=tts_language_code)
        audio_file = "speech.mp3"
        tts.save(audio_file)

        # drop a streamlit audio file 
        with open(audio_file, "rb") as audio:
            st.audio(audio.read(), format="audio/mp3")

        # get rid of any temporary files (this is from gTTs) #
        os.remove(audio_file)

    except Exception as e:
        st.error(f"Error generating audio: {e}")

## camera function ##
def capture_camera(stop_video, selected_language, volume_control, return_audio):
    """Captures live video from the camera."""
    cap = cv2.VideoCapture(0) 
    FRAME_WINDOW = st.image([], use_container_width=True)

    # Initialize or reset the session sentence
    if "sentence" not in st.session_state:
        st.session_state.sentence = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Unable to capture video. Please check your camera.")
            break

        # Allow color for our camera
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(rgb_frame, channels="RGB", use_container_width=True)

        # Simulate sentence generation **HARD CODED SENTENCE**
        if len(st.session_state.sentence) < 1:
            st.session_state.sentence.append("Hello World")

        if stop_video:
            break

    # Release camera + clean up
    cap.release()
    cv2.destroyAllWindows()

def main():
    """Main function to run the Streamlit app."""
    display_app_header()

    # Starting up the session state for the camera
    if "camera_started" not in st.session_state:
        st.session_state.camera_started = False

    start_video, stop_video, selected_language, volume_control, return_audio, to_lang= setup_controls()
    
    translated=translate_text('fuck you',to_lang)
    print(translated)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center;'>Live Video Feed</h4>", unsafe_allow_html=True)
    
    code_TO_LANG = {
    "en" : "English",
    "es": "Spanish" ,
    "fr": "French",
    "ja" : "Japanese",
    "de": "German",
    "zh": "Chinese",
    "hi": "Hindi",
    "ar" : "Arabic",
    "ru" : "Russian",
    "ur" : "Urdu",
    "bn" : "Bengali",
}
    # Audio playback
    if st.button("Play Translated Speech"):
        '''if "sentence" in st.session_state and st.session_state.sentence:
            text_to_speak = " ".join(st.session_state.sentence)
        if not volume_control and return_audio:'''
        generate_audio_with_openai(translated, code_TO_LANG[to_lang])

    # Interactivity for camera actions
    if start_video and not st.session_state.camera_started:
        st.session_state.camera_started = True
        capture_camera(stop_video, selected_language, volume_control, return_audio)
    elif stop_video and st.session_state.camera_started:
        st.session_state.camera_started = False
        capture_camera(stop_video, selected_language, volume_control, return_audio)

if __name__ == "__main__":
    main()

## TO DO: ##
## --> connect tranlsate.py 
