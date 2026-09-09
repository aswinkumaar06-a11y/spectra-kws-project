import os
import subprocess

WORDS = ["spectrum", "inspector", "spectacle", "extra"]
OUTPUT_DIR = "dataset/raw/negative_words/hard_negatives"

def generate():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # This requires piper-tts to be installed and models downloaded,
    # or you can write a simple loop that calls piper.
    # For now, it's a placeholder script showing how to generate them.
    print("To generate hard negatives, you need a TTS engine like piper-tts.")
    print(f"Words to generate: {WORDS}")
    print(f"Output directory: {OUTPUT_DIR}")

if __name__ == "__main__":
    generate()
