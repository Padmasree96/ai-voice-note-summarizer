import gradio as gr
from transformers import pipeline

# Load models (cached automatically on HF Spaces)
speech_to_text = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-small"
)

summarizer = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-12-6"
)

def voice_note_summarizer(audio):
    if audio is None:
        return "No audio uploaded", ""

    # Speech → Text
    result = speech_to_text(audio)
    text = result["text"]

    # If text is short, skip summarization
    if len(text.split()) < 20:
        return text, "Text is too short to summarize."

    # Text → Summary
    summary = summarizer(
        text,
        max_length=50,
        min_length=20,
        do_sample=False
    )

    return text, summary[0]["summary_text"]


# Gradio UI
interface = gr.Interface(
    fn=voice_note_summarizer,
    inputs=gr.Audio(type="filepath", label="Upload Voice Note"),
    outputs=[
        gr.Textbox(label="Full Transcription"),
        gr.Textbox(label="Summary")
    ],
    title="AI Voice Note Summarizer",
    description="Upload a voice note. The app converts speech to text and generates a short summary using AI."
)

interface.launch()
