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
from translate_hf_model import translate_hf_model
#from translate import translate_text, download_languages_for_demo ## still need to implement functions
#from classifier import classifier ## implement function
#from capture_sign import capture_sign ## implement function

#download_languages_for_demo() ## from translate


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("API key not found! Check for a valid .env file with 'OPENAI_API_KEY'.")

openai.api_key = OPENAI_API_KEY
from pathlib import Path
from openai import OpenAI
client = OpenAI()

#dctionary for languages
LANGUAGES = {

    "English": "en_XX",
    "Spanish": "es_XX",
    "French": "fr_XX",
    "Gujarati": "gu_IN",
    "Hindi": "hi_IN",
    "Italian": "it_IT",
    "Japanese": "ja_XX",
    "Kazakh": "kk_KZ",
    "Korean": "ko_KR",
    "Russian": "ru_RU",
    "Turkish": "tr_TR",
    "Vietnamese": "vi_VN",
    "Chinese": "zh_CN",
    "Bengali": "bn_IN",
    "Polish": "pl_PL",
    "Portuguese": "pt_XX",
    "Swedish": "sv_SE",
    "Ukrainian": "uk_UA",
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
    if 'signs' not in st.session_state:
        st.session_state.signs = []

    # Camera buttons
    with st.container():
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button('Capture Sign'):
                capture_sign()
                sign = classifier()
                st.session_state.signs.append(sign)
                st.write(f"Captured signs: {st.session_state.signs}")
        
        with col2:
            if st.button('Clear Sentence'):
                st.session_state.signs=[]
                st.write(f"Sentence Cleared!")
    
    # Language select
    selected_language = st.selectbox(
        "Select Language",
        list(LANGUAGES.keys()),
        index=0,
        help="Choose the language for ASL interpretation",
    )
    to_code = LANGUAGES[selected_language]

    return selected_language, to_code, st.session_state.signs
    

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
    selected_language, to_lang, signs = setup_controls()
    
    if st.button('Translate'):
        sentence=''
        
        for sign in signs:
            #print(sign)
            if sign=='None' or sign==' ' or sign=='':
                continue #skip
            sentence+= sign
  
        if to_lang=='en_XX':
            output=sentence #skip translation
        else:
            output = translate_hf_model(sentence, to_lang)
        if output=='Environment Canada, Inc.':
            output='No signs detected' 
        print(output)#weird translation error
        st.session_state.translated_text = output
        st.write(f"Translated text: {output}")
    
    if 'translated_text' in st.session_state:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<h4 style='text-align: center;'>Translated Audio</h4>", unsafe_allow_html=True)
        
        # Audio playback
        if st.button("Play Translated Speech"):
            generate_audio_with_openai(st.session_state.translated_text)

if __name__ == "__main__":
    main()

