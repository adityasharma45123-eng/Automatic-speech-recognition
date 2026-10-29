
from pathlib import Path
from types import SimpleNamespace

import pytest

from asr_engine import (
    ASREngine,
    TranscriptSegment,
    TranscriptionResult,
    format_timestamp,
    render_transcript,
)


class FakeModel:
    def transcribe(self, audio, **kwargs):
        assert Path(audio).is_file()

        segments = [
            SimpleNamespace(start=0.0, end=1.0, text="Hello"),
            SimpleNamespace(start=1.0, end=2.0, text="world!"),
        ]

        info = SimpleNamespace(
            language="en",
            language_probability=0.98,
        )

        return iter(segments), info


def test_transcription(tmp_path):
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"test audio")

    engine = ASREngine(model_size="tiny")
    engine.model = FakeModel()

    result = engine.transcribe(audio, language="en")

    assert result.text == "Hello world!"
    assert result.language == "en"
    assert len(result.segments) == 2


def test_missing_audio_file():
    engine = ASREngine(model_size="tiny")
    engine.model = FakeModel()

    with pytest.raises(FileNotFoundError):
        engine.transcribe("missing.wav")


def test_prepare_audio_uses_ffmpeg_conversion(monkeypatch, tmp_path):
    audio = tmp_path / "input.mp3"
    audio.write_bytes(b"test audio")

    captured = {}

    def fake_run(command, check, capture_output, text):
        captured["command"] = command
        output_file = Path(command[-1])
        output_file.write_bytes(b"converted wav")
        return None

    monkeypatch.setattr("asr_engine.shutil.which", lambda _: "ffmpeg")
    monkeypatch.setattr("asr_engine.subprocess.run", fake_run)

    prepared = ASREngine(model_size="tiny")._prepare_audio(audio)

    assert prepared.suffix == ".wav"
    assert prepared.exists()
    assert captured["command"][0] == "ffmpeg"
    assert captured["command"][1] == "-y"


def test_invalid_model():
    with pytest.raises(ValueError):
        ASREngine(model_size="invalid")


def test_cpu_compute_type_is_supported(monkeypatch):
    captured = {}

    class DummyModel:
        def __init__(self, *args, **kwargs):
            captured["kwargs"] = kwargs

    monkeypatch.setitem(
        __import__("sys").modules,
        "faster_whisper",
        SimpleNamespace(WhisperModel=DummyModel),
    )

    ASREngine(model_size="tiny", device="cpu")._load_model()

    assert captured["kwargs"]["device"] == "cpu"
    assert captured["kwargs"]["compute_type"] == "int8"


def test_av_open_wrapper_strips_metadata_errors(monkeypatch):
    import av

    calls = {}

    def fake_open(*args, **kwargs):
        calls["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(av, "open", fake_open)

    import asr_engine

    asr_engine._patch_av_open_compatibility()
    av.open("sample.wav", mode="r", metadata_errors="ignore")

    assert "metadata_errors" not in calls["kwargs"]


def test_timestamp():
    assert format_timestamp(65.125) == "00:01:05.125"


def test_render_transcript():
    result = TranscriptionResult(
        text="Hello world!",
        language="en",
        language_probability=0.98,
        segments=[
            TranscriptSegment(0.0, 1.0, "Hello"),
            TranscriptSegment(1.0, 2.0, "world!"),
        ],
    )

    assert render_transcript(result) == "Hello world!"

    assert render_transcript(result, timestamps=True) == (
        "[00:00:00.000 --> 00:00:01.000] Hello\n"
        "[00:00:01.000 --> 00:00:02.000] world!"
    )
