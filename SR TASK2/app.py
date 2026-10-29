
import os
import tempfile
from pathlib import Path

import streamlit as st
from asr_engine import ASREngine, render_transcript

st.set_page_config(
    page_title="ASR Tool",
    page_icon="🎙️",
    layout="centered",
)

st.title("🎙️ Automatic Speech Recognition")
st.write("Convert your audio recordings into text.")

with st.sidebar:
    st.header("Settings")

    model_size = st.selectbox(
        "Model size",
        ["tiny", "base", "small", "medium", "large-v3"],
        index=2,
    )

    language_name = st.selectbox(
        "Audio language",
        [
            "Auto-detect",
            "English",
            "Hindi",
            "Tamil",
            "Telugu",
            "Bengali",
        ],
    )

    language_codes = {
        "English": "en",
        "Hindi": "hi",
        "Tamil": "ta",
        "Telugu": "te",
        "Bengali": "bn",
    }

    timestamps = st.checkbox("Include timestamps")

audio_file = st.file_uploader(
    "Upload an audio file",
    type=["wav", "mp3", "m4a", "flac", "ogg", "webm"],
)

if audio_file is not None:
    st.audio(audio_file)

    if st.button("Transcribe Audio", type="primary"):
        temp_path = None

        try:
            suffix = Path(audio_file.name).suffix or ".wav"

            with tempfile.NamedTemporaryFile(
                delete=False, suffix=suffix
            ) as temp_file:
                temp_file.write(audio_file.getvalue())
                temp_path = temp_file.name

            with st.spinner(
                "Transcribing audio. Please wait..."
            ):
                engine = ASREngine(model_size=model_size)

                language = language_codes.get(language_name)

                result = engine.transcribe(
                    temp_path,
                    language=language,
                )

            transcript = render_transcript(
                result,
                timestamps=timestamps,
            )

            st.subheader("Transcription Result")

            st.text_area(
                "Recognized text",
                value=transcript,
                height=250,
            )

            if result.language:
                probability = result.language_probability

                message = f"Detected language: {result.language}"
                if probability is not None:
                    message += f" (confidence: {probability:.0%})"

                st.info(message)

            st.download_button(
                label="Download Transcript",
                data=transcript,
                file_name="transcript.txt",
                mime="text/plain",
            )

        except Exception as error:
            st.error(f"Transcription failed: {error}")

        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
else:
    st.info("Upload an audio file to get started.")
