import streamlit as st
import cv2
import numpy as np
import os
import openai
import tensorflow as tf ## still need to debug
from dotenv import load_dotenv
from our_Translate import translate_text
from capture_sign import capture_sign
from classifier import classifier
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
from pathlib import Path
from openai import OpenAI
client = OpenAI()

# Languages # ~~ from mackenzie's lang for demo function
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
        start_video = col1.button("Next")
        stop_video = col2.button("End")
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
def generate_audio_with_openai(text):
    """Generates TTS audio using OpenAI's API and gTTS for speech synthesis."""
    try:
        speech_file_path = Path(__file__).parent / "speech.mp3"
        response = client.audio.speech.create(
        model="tts-1",
        voice="onyx",
        input=text
        )

        response.stream_to_file(speech_file_path)
        st.audio(str(speech_file_path))
    except Exception as e:
        st.error(f"Error generating audio: {e}")


def main():
    """Main function to run the Streamlit app."""
    display_app_header()


    start_video, stop_video, selected_language, volume_control, return_audio, to_lang= setup_controls()
    
    translated=translate_text('kill yourself',to_lang)
    print(translated)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center;'>Live Video Feed</h4>", unsafe_allow_html=True)
    
    
    # Audio playback
    if st.button("Play Translated Speech"):
        generate_audio_with_openai(translated)


if __name__ == "__main__":
    main()

## TO DO: ##
## --> connect tranlsate.py 
