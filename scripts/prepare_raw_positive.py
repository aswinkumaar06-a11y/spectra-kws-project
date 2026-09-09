"""
prepare_raw_positive.py — Step 0 (Non-destructive Raw Data Preprocessing)
Converts .m4a and .aac files in dataset/raw/positive/ to 16 kHz mono .wav files.
Preserves original files by default (non-destructive) unless explicitly flagged.
"""
import os
import glob
import subprocess
import hashlib
import argparse

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
    """Use ffmpeg to convert audio file to 16 kHz mono WAV."""
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

def main(delete_originals=False):
    # --- Pass 1: Deduplicate '- Copy' variants safely ---
    all_files = glob.glob(os.path.join(SRC, "**", "*"), recursive=True)
    all_files = [f for f in all_files if os.path.isfile(f)]
    copies = [f for f in all_files if " - Copy" in os.path.basename(f)]
    if copies:
        print(f"Found {len(copies)} '- Copy' files.")
        if delete_originals:
            for f in copies:
                os.remove(f)
            print(f"Deleted {len(copies)} redundant copies.")
        else:
            print("Preserving '- Copy' files (run with --delete-originals to purge).")

    # --- Pass 2: Convert non-WAV files non-destructively ---
    non_wav = []
    for ext in ("*.m4a", "*.aac"):
        non_wav.extend(glob.glob(os.path.join(SRC, "**", ext), recursive=True))
    print(f"Checking {len(non_wav)} non-WAV files in {SRC}...")

    converted = 0
    skipped = 0
    failed = 0
    for src_path in non_wav:
        base = os.path.splitext(src_path)[0]
        dst_path = base + ".wav"
        if os.path.exists(dst_path):
            skipped += 1
            continue
        if convert_to_wav(src_path, dst_path):
            converted += 1
            if delete_originals:
                os.remove(src_path)
        else:
            failed += 1

    print(f"Converted: {converted}, Skipped (already exist): {skipped}, Failed: {failed}")
    wav_files = glob.glob(os.path.join(SRC, "**", "*.wav"), recursive=True)
    print(f"Total .wav files in {SRC}: {len(wav_files)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare raw positive audio files safely")
    parser.add_argument("--delete-originals", action="store_true", help="Delete source files after conversion")
    args = parser.parse_args()
    main(delete_originals=args.delete_originals)
