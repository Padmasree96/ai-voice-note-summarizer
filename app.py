import gradio as gr
import whisper
from transformers import pipeline

# Load Whisper model (Speech → English Translation)
whisper_model = whisper.load_model("base")

# Load Summarization model
summarizer = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-12-6"
)

def process_audio(audio_path):
    if audio_path is None:
        return "No audio uploaded", "No summary"

    # Tamil audio → English text
    result = whisper_model.transcribe(audio_path, task="translate")
    english_text = result["text"]

    if len(english_text.strip()) == 0:
        return "Speech not recognized", "No summary"

    # Generate summary
    if len(english_text.split()) < 20:
        summary_text = "Text is too short to summarize."
    else:
        summary = summarizer(
            english_text,
            max_length=60,
            min_length=20,
            do_sample=False
        )
        summary_text = summary[0]["summary_text"]

    return english_text, summary_text


# UI
with gr.Blocks() as demo:
    gr.Markdown("## 🎙️ AI Voice Note Summarizer")
    gr.Markdown("Tamil voice → English translation → Summary")

    audio_input = gr.Audio(type="filepath", label="Upload Tamil Audio")
    text_output = gr.Textbox(label="English Translation", lines=6)
    summary_output = gr.Textbox(label="Summary", lines=4)

    btn = gr.Button("Convert & Summarize")
    btn.click(process_audio, audio_input, [text_output, summary_output])

demo.launch(share=True)
