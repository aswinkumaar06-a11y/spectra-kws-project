"""
Converts all .m4a and .aac files in dataset/raw/positive/ to 16 kHz mono
.wav files (in-place).  Also deduplicates '- Copy' variants.
Run this ONCE before augment.py.
"""
import os
import glob
import subprocess
import hashlib

SRC = "dataset/raw/positive"
SR = 16000


def file_hash(path, chunk=8192):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            data = f.read(chunk)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


def convert_to_wav(src_path, dst_path):
    """Use ffmpeg to convert any audio file to 16 kHz mono WAV."""
    cmd = [
        "ffmpeg", "-y", "-i", src_path,
        "-ar", str(SR), "-ac", "1",
        "-sample_fmt", "s16",
        dst_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  FAILED: {src_path}\n    {result.stderr[:200]}")
        return False
    return True


def main():
    # --- Pass 1: Remove exact '- Copy' duplicates ---
    all_files = glob.glob(os.path.join(SRC, "**", "*"), recursive=True)
    all_files = [f for f in all_files if os.path.isfile(f)]
    removed = 0
    for f in all_files:
        if " - Copy" in os.path.basename(f):
            os.remove(f)
            removed += 1
    print(f"Removed {removed} '- Copy' duplicates")

    # --- Pass 2: Convert .m4a / .aac -> .wav ---
    non_wav = []
    for ext in ("*.m4a", "*.aac"):
        non_wav.extend(glob.glob(os.path.join(SRC, "**", ext), recursive=True))
    print(f"Converting {len(non_wav)} non-WAV files to 16 kHz mono .wav ...")

    converted = 0
    failed = 0
    for src_path in non_wav:
        base = os.path.splitext(src_path)[0]
        dst_path = base + ".wav"
        # Skip if a .wav version already exists
        if os.path.exists(dst_path):
            os.remove(src_path)
            converted += 1
            continue
        if convert_to_wav(src_path, dst_path):
            os.remove(src_path)   # Remove the original after successful conversion
            converted += 1
        else:
            failed += 1

    print(f"Converted: {converted}  Failed: {failed}")

    # --- Final count ---
    wav_files = glob.glob(os.path.join(SRC, "**", "*.wav"), recursive=True)
    print(f"Total .wav files in {SRC}: {len(wav_files)}")


if __name__ == "__main__":
    main()
