
import argparse
import sys

from asr_engine import ASREngine, render_transcript


def main():
    parser = argparse.ArgumentParser(
        description="Automatic Speech Recognition CLI"
    )

    parser.add_argument("audio", help="Path to the audio file")
    parser.add_argument(
        "--model",
        default="small",
        choices=sorted(ASREngine.ALLOWED_MODELS),
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Language code, for example en, hi, or ta",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--timestamps", action="store_true")

    args = parser.parse_args()

    try:
        engine = ASREngine(model_size=args.model)

        result = engine.transcribe(
            args.audio,
            language=args.language,
        )

        transcript = render_transcript(
            result,
            timestamps=args.timestamps,
        )

        if args.output:
            with open(args.output, "w", encoding="utf-8") as file:
                file.write(transcript + "\n")
            print(f"Transcript saved to {args.output}")
        else:
            print(transcript)

        if result.language:
            print(
                f"Detected language: {result.language}",
                file=sys.stderr,
            )

    except (ValueError, FileNotFoundError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
