import sys
from pathlib import Path

import librosa
import soundfile as sf
from faster_whisper import WhisperModel

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_WAV = BASE_DIR / "ref_audio" / "bujji_voice.wav"
ENV_FILE = BASE_DIR / ".env"


def prepare_reference_audio(input_audio_path: str, start_sec: float = 10.0, duration_sec: float = 8.5):
    src = Path(input_audio_path)
    if not src.exists():
        print(f"[Error] Audio file not found: {src}")
        sys.exit(1)

    print(f"[1/3] Loading and resampling '{src.name}' to 24 kHz mono...")
    y, sr = librosa.load(str(src), sr=24000, mono=True)
    total_sec = len(y) / sr

    # Extract clean English segment if audio is longer than 12 seconds
    if total_sec > 12.0 and (start_sec + duration_sec) <= total_sec:
        start_idx = int(start_sec * sr)
        end_idx = int((start_sec + duration_sec) * sr)
        y = y[start_idx:end_idx]

    y, _ = librosa.effects.trim(y, top_db=28)
    sf.write(str(OUT_WAV), y, 24000, subtype="PCM_16")
    print(f"[2/3] Saved clean reference clip ({len(y)/24000:.1f}s) -> {OUT_WAV}")

    print("[3/3] Transcribing exact reference text with Faster-Whisper...")
    model = WhisperModel("base", device="cpu", compute_type="int8")
    segments, _ = model.transcribe(str(OUT_WAV), language="en")
    ref_text = " ".join(s.text.strip() for s in segments).strip()
    print(f"Detected Reference Transcript: \"{ref_text}\"")

    if ENV_FILE.exists():
        lines = ENV_FILE.read_text(encoding="utf-8").splitlines()
        updated = []
        for line in lines:
            if line.startswith("REF_AUDIO_TEXT="):
                updated.append(f"REF_AUDIO_TEXT={ref_text}")
            else:
                updated.append(line)
        ENV_FILE.write_text("\n".join(updated) + "\n", encoding="utf-8")
        print("Updated REF_AUDIO_TEXT in .env successfully.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python voice_client/prepare_voice.py <path_to_your_uploaded_audio_file>")
        sys.exit(1)
    prepare_reference_audio(sys.argv[1])