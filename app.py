import streamlit as st
st.write("SpeechRecognition loaded")
import speech_recognition as sr
from transformers import pipeline
from googletrans import Translator
import os

st.set_page_config(page_title="AI Voice Note Summarizer", layout="centered")

st.title("AI Voice Note Summarizer")
st.write("Tamil & English voice → AI summary")

# Language selection
language = st.selectbox(
    "Select Language",
    ("English", "Tamil")
)

# Language code
if language == "Tamil":
    lang_code = "ta-IN"
else:
    lang_code = "en-IN"

# Upload audio
audio_file = st.file_uploader("Upload WAV audio file", type=["wav"])

if audio_file is not None:
    with open("voice.wav", "wb") as f:
        f.write(audio_file.read())

    st.success("Audio uploaded successfully")

    if st.button("Generate Summary"):
        r = sr.Recognizer()

        #Speech to Text
        try:
            with sr.AudioFile("voice.wav") as source:
                audio = r.record(source)

            text = r.recognize_google(audio, language=lang_code)
            st.subheader("Extracted Text")
            st.write(text)

        except Exception as e:
            st.error(f"Speech Recognition Error: {e}")
            st.stop()

        #Tamil - English Translation
        if language == "Tamil":
            translator = Translator()
            translated = translator.translate(text, src="ta", dest="en")
            english_text = translated.text

            st.subheader("Translated English Text")
            st.write(english_text)
        else:
            english_text = text

        #Summarization
        summarizer = pipeline("summarization")

        if len(english_text.split()) < 20:
            st.warning("Text is too short to summarize.")
        else:
            summary = summarizer(
                english_text,
                max_length=60,
                min_length=25,
                do_sample=False
            )

            st.subheader("AI Summary")
            st.write(summary[0]['summary_text'])
