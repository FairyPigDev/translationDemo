import os
import tempfile

import docx2txt
import gradio as gr
import pypdf
import whisper
from deep_translator import GoogleTranslator
from gtts import gTTS

DIRECTIONS = ["English → Korean", "Korean → English"]

_whisper_model = None


def get_whisper():
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = whisper.load_model("base")
    return _whisper_model


def _lang_pair(direction):
    return ("en", "ko") if direction == "English → Korean" else ("ko", "en")


def translate_text(text, direction):
    if not text or not text.strip():
        return ""
    src, tgt = _lang_pair(direction)
    translator = GoogleTranslator(source=src, target=tgt)
    chunks = [text[i:i + 4500] for i in range(0, len(text), 4500)]
    return "".join(translator.translate(c) for c in chunks if c.strip())


def synthesize_speech(text, direction):
    if not text or not text.strip():
        return None
    _, tgt = _lang_pair(direction)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    gTTS(text=text, lang=tgt).save(tmp.name)
    return tmp.name


def extract_file_text(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".txt":
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    if ext in (".doc", ".docx"):
        return docx2txt.process(path) or ""
    if ext == ".pdf":
        reader = pypdf.PdfReader(path)
        return "\n".join((p.extract_text() or "") for p in reader.pages)
    raise gr.Error(f"Unsupported file type: {ext}")


def translate_file(file_obj, direction):
    if file_obj is None:
        return "", ""
    path = file_obj if isinstance(file_obj, str) else file_obj.name
    extracted = extract_file_text(path)
    return extracted, translate_text(extracted, direction)


def translate_audio(audio_path, direction):
    if audio_path is None:
        return "", "", None
    transcription = get_whisper().transcribe(audio_path)["text"].strip()
    translated = translate_text(transcription, direction)
    audio_out = synthesize_speech(translated, direction)
    return transcription, translated, audio_out


with gr.Blocks(title="English ↔ Korean Translator") as app:
    gr.Markdown("# English ↔ Korean Translator\nTranslate text, files, audio files, or live microphone input.")

    with gr.Tabs():
        with gr.Tab("Text"):
            t_dir = gr.Dropdown(DIRECTIONS, value=DIRECTIONS[0], label="Direction")
            t_in = gr.Textbox(label="Input text", lines=6, placeholder="Type text to translate...")
            t_btn = gr.Button("Translate", variant="primary")
            t_out = gr.Textbox(label="Translation", lines=6)
            t_tts_btn = gr.Button("Play translation as audio")
            t_audio = gr.Audio(label="Translated audio", type="filepath")
            t_btn.click(translate_text, [t_in, t_dir], t_out)
            t_tts_btn.click(synthesize_speech, [t_out, t_dir], t_audio)

        with gr.Tab("File"):
            f_dir = gr.Dropdown(DIRECTIONS, value=DIRECTIONS[0], label="Direction")
            f_in = gr.File(
                label="Upload .txt, .doc, .docx, or .pdf",
                file_types=[".txt", ".doc", ".docx", ".pdf"],
            )
            f_btn = gr.Button("Extract & Translate", variant="primary")
            f_extracted = gr.Textbox(label="Extracted text", lines=8)
            f_out = gr.Textbox(label="Translation", lines=8)
            f_btn.click(translate_file, [f_in, f_dir], [f_extracted, f_out])

        with gr.Tab("Audio file"):
            a_dir = gr.Dropdown(DIRECTIONS, value=DIRECTIONS[0], label="Direction")
            a_in = gr.Audio(sources=["upload"], type="filepath", label="Upload .mp3 or .wav")
            a_btn = gr.Button("Transcribe & Translate", variant="primary")
            a_transcribed = gr.Textbox(label="Transcription", lines=4)
            a_translated = gr.Textbox(label="Translation", lines=4)
            a_out = gr.Audio(label="Translated audio", type="filepath")
            a_btn.click(translate_audio, [a_in, a_dir], [a_transcribed, a_translated, a_out])

        with gr.Tab("Microphone"):
            m_dir = gr.Dropdown(DIRECTIONS, value=DIRECTIONS[0], label="Direction")
            m_in = gr.Audio(sources=["microphone"], type="filepath", label="Record live speech")
            m_btn = gr.Button("Transcribe & Translate", variant="primary")
            m_transcribed = gr.Textbox(label="Transcription", lines=4)
            m_translated = gr.Textbox(label="Translation", lines=4)
            m_out = gr.Audio(label="Translated audio", type="filepath")
            m_btn.click(translate_audio, [m_in, m_dir], [m_transcribed, m_translated, m_out])


if __name__ == "__main__":
    app.launch()
