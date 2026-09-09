"""
generate_hard_negatives.py — STEP 4.
Generates phonetically adjacent negative words across multiple synthetic voices/rates.
Respects D02: outputs are saved into distinct subfolders per voice profile so speaker
splitting ensures synthetic voices never leak between train/val/test.
"""

import os
import pyttsx3
import librosa
import soundfile as sf
import numpy as np

WORDS = [
    "spectrum",
    "inspector",
    "spectacle",
    "extra",
    "spectral",
    "spectator",
    "speculate",
    "vector",
    "expect",
    "respect",
    "inspection",
    "sector"
]

OUTPUT_ROOT = "dataset/raw/negative_words/hard_negatives"
TARGET_SR = 16000
TARGET_DURATION = 1.0 # seconds
TARGET_SAMPLES = int(TARGET_SR * TARGET_DURATION)

def fix_and_save_audio(temp_path, final_path):
    audio, sr = librosa.load(temp_path, sr=TARGET_SR)
    if len(audio) < TARGET_SAMPLES:
        audio = np.pad(audio, (0, TARGET_SAMPLES - len(audio)), mode="constant")
    else:
        audio = audio[:TARGET_SAMPLES]
    # Peak-normalize
    max_amp = np.max(np.abs(audio))
    if max_amp > 1e-6:
        audio = audio / max_amp
    sf.write(final_path, audio, TARGET_SR)
    if os.path.exists(temp_path):
        os.remove(temp_path)

def generate():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    temp_engine = pyttsx3.init()
    voices = temp_engine.getProperty('voices')
    del temp_engine

    profiles = []
    for v in voices:
        v_name = "david" if "david" in v.name.lower() or "david" in v.id.lower() else "zira"
        for rate in [130, 160, 190, 220]:
            profile_id = f"synth_{v_name}_r{rate}"
            profiles.append({
                "id": profile_id,
                "voice_id": v.id,
                "rate": rate
            })

    print(f"Generating hard negatives across {len(profiles)} synthetic speaker profiles...")
    print(f"Words: {WORDS}")

    total_generated = 0
    for p in profiles:
        voice_dir = os.path.join(OUTPUT_ROOT, p["id"])
        os.makedirs(voice_dir, exist_ok=True)

        engine = pyttsx3.init()
        engine.setProperty('voice', p["voice_id"])
        engine.setProperty('rate', p["rate"])

        temp_map = {}
        for word in WORDS:
            temp_file = os.path.join(voice_dir, f"temp_{word}.wav")
            final_file = os.path.join(voice_dir, f"{word}.wav")
            temp_map[temp_file] = final_file
            engine.save_to_file(word, temp_file)

        engine.runAndWait()
        del engine

        for temp_file, final_file in temp_map.items():
            if os.path.exists(temp_file):
                fix_and_save_audio(temp_file, final_file)
                total_generated += 1

        print(f"  Completed voice profile: {p['id']} ({len(temp_map)} words)")

    print(f"\nGenerated {total_generated} hard negative samples in {OUTPUT_ROOT}/ across {len(profiles)} voice profiles.")

if __name__ == "__main__":
    generate()
