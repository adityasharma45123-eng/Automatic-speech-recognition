
# ASR Tool Implementation

## Overview

This project implements an Automatic Speech Recognition (ASR)
tool that converts spoken audio into text using Faster-Whisper.

## Features

- Audio-to-text conversion
- Automatic language detection
- English, Hindi, Tamil and other language support
- Timestamped transcription
- Downloadable text transcripts
- Streamlit web interface
- Command-line transcription
- Automated unit tests

## Technologies Used

- Python 3.10+
- Faster-Whisper
- Streamlit
- Pytest
- Git and GitHub

## Installation

Create and activate a virtual environment.

Windows:

    python -m venv .venv
    .venv\Scripts\Activate.ps1

Install dependencies:

    python -m pip install -r requirements.txt

## Run the Web Application

    streamlit run app.py

Open http://localhost:8501 in your browser.

## Run the Command-Line Tool

    python transcribe.py audio.wav

Specify a language:

    python transcribe.py audio.wav --language hi

Save a transcript:

    python transcribe.py audio.wav --output transcript.txt

Include timestamps:

    python transcribe.py audio.wav --timestamps

## Run Tests

    python -m pytest -q

## Project Structure

    app.py
    asr_engine.py
    transcribe.py
    requirements.txt
    README.md
    tests/
        test_asr_engine.py

## Limitations

Transcription accuracy depends on audio quality,
background noise, accent and language.

The first transcription may require an internet connection
to download the selected model.
