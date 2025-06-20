import argparse
import gc
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from faster_whisper import WhisperModel


def run(cmd):
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def extract_audio(video: Path, wav: Path, duration: int | None = None):
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video),
           "-ac", "1", "-ar", "16000", "-vn"]
    if duration:
        cmd += ["-t", str(duration)]
    cmd.append(str(wav))
    run(cmd)


def remove_tiny_cache():
    cache = Path.home() / ".cache" / "huggingface" / "hub"
    for p in cache.glob("**/openai--whisper-tiny*"):
        shutil.rmtree(p, ignore_errors=True)


def transcribe(video_path: Path, device: str = "cpu"):
    if not video_path.exists():
        sys.exit(f"❌ File not found: {video_path}")

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        full_wav = tmp / "audio_full.wav"
        sample_wav = tmp / "audio_30s.wav"

        print("• Extracting audio …")
        extract_audio(video_path, full_wav)
        extract_audio(video_path, sample_wav, 30)

        print("• Detecting language (Whisper‑tiny) …")
        tiny = WhisperModel("tiny", device=device, compute_type="int8")
        segments, info = tiny.transcribe(str(sample_wav), language=None, beam_size=1)
        lang = info.language or "en"
        print(f"  ➜ Detected language: {lang}")

        tiny = None
        gc.collect()
        remove_tiny_cache()

        print("• Transcribing full audio with Whisper‑small …")
        model_id = "small.en" if lang == "en" else "small"
        small = WhisperModel(model_id, device=device, compute_type="int8")
        segments, _ = small.transcribe(str(full_wav), language=lang, beam_size=5, vad_filter=True)

        transcript = " ".join(s.text.strip() for s in segments)

    out_file = video_path.with_name(f"{video_path.stem}_transcript.txt")
    out_file.write_text(transcript, encoding="utf-8")
    print(f"\n✅ Transcript saved to: {out_file}")
    print("\n📝 Transcript preview:\n")
    print(transcript[:800] + ("..." if len(transcript) > 800 else ""))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Transcribe a video to text with Whisper.")
    p.add_argument("video", nargs="?", type=Path, help="Path to video file")
    p.add_argument("--device", default="cpu", help="cpu | cuda")
    args = p.parse_args()

    video_path = args.video or Path(input("Enter the path to your video file: ").strip())
    transcribe(video_path, args.device)
