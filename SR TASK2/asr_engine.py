
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    import av
except ImportError:  # pragma: no cover - optional dependency for audio decoding
    av = None


def _patch_av_open_compatibility():
    """Support PyAV versions that do not expose the `metadata_errors` kwarg."""
    if av is None:
        return

    original_open = av.open

    def compatible_open(*args, metadata_errors=None, **kwargs):
        kwargs.pop("metadata_errors", None)
        return original_open(*args, **kwargs)

    av.open = compatible_open


_patch_av_open_compatibility()


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str


@dataclass
class TranscriptionResult:
    text: str
    language: Optional[str]
    language_probability: Optional[float]
    segments: list[TranscriptSegment]


class ASREngine:
    ALLOWED_MODELS = {"tiny", "base", "small", "medium", "large-v3"}

    def __init__(self, model_size="small", device="cpu"):
        if model_size not in self.ALLOWED_MODELS:
            raise ValueError(f"Unsupported model: {model_size}")

        self.model_size = model_size
        self.device = device
        self.model = None

    def _prepare_audio(self, audio_path):
        path = Path(audio_path)
        if not path.is_file():
            raise FileNotFoundError(f"Audio file not found: {path}")

        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            return path

        temp_audio = Path(
            tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        )
        command = [
            ffmpeg,
            "-y",
            "-i",
            str(path),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(temp_audio),
        ]

        try:
            subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as error:
            raise RuntimeError(
                "Audio conversion failed. Please use a valid audio file. "
                f"ffmpeg output: {error.stderr.strip() or str(error)}"
            ) from error

        return temp_audio

    def _load_model(self):
        if self.model is None:
            from faster_whisper import WhisperModel

            compute_type = "int8" if self.device == "cpu" else "float16"
            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=compute_type,
            )
        return self.model

    def transcribe(self, audio_path, language=None):
        path = Path(audio_path)
        prepared_path = self._prepare_audio(path)

        temp_created = prepared_path != path

        try:
            model = self._load_model()
            segments_iterator, info = model.transcribe(
                str(prepared_path),
                language=language or None,
                task="transcribe",
                beam_size=8,
                best_of=5,
                patience=1.0,
                temperature=[0.0, 0.2, 0.4, 0.6],
                condition_on_previous_text=False,
                vad_filter=True,
                vad_parameters={
                    "min_silence_duration_ms": 250,
                    "speech_pad_ms": 200,
                },
            )

            segments = [
                TranscriptSegment(
                    start=float(segment.start),
                    end=float(segment.end),
                    text=segment.text.strip(),
                )
                for segment in segments_iterator
            ]

            text = " ".join(s.text for s in segments if s.text).strip()

            return TranscriptionResult(
                text=text,
                language=getattr(info, "language", None),
                language_probability=getattr(
                    info, "language_probability", None
                ),
                segments=segments,
            )
        finally:
            if temp_created and prepared_path.exists():
                prepared_path.unlink(missing_ok=True)


def format_timestamp(seconds):
    milliseconds = int(round(max(0, seconds) * 1000))
    hours, remainder = divmod(milliseconds, 3600000)
    minutes, remainder = divmod(remainder, 60000)
    secs, millis = divmod(remainder, 1000)

    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def render_transcript(result, timestamps=False):
    if not timestamps:
        return result.text

    lines = []
    for segment in result.segments:
        start = format_timestamp(segment.start)
        end = format_timestamp(segment.end)
        lines.append(f"[{start} --> {end}] {segment.text}")

    return "\n".join(lines)
