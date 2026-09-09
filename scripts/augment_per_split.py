"""
augment_per_split.py — STEP 2. Augments each split independently so
augmented copies can never leak into a different split than their
source file. Run AFTER split_raw_only.py.
"""
import os
import glob
import random
import numpy as np
import librosa
import soundfile as sf

SR = 16000
NOISE_FILES = glob.glob("dataset/raw/noise/*.wav") + \
              glob.glob("dataset/raw/noise/**/*.wav", recursive=True)

def add_noise(clean, noise, snr_db):
    noise = np.resize(noise, clean.shape)
    clean_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2) + 1e-9
    factor = np.sqrt(clean_power / (noise_power * 10 ** (snr_db / 10)))
    return clean + factor * noise

def augment_folder(src_folder, dest_folder, do_augment=True):
    os.makedirs(dest_folder, exist_ok=True)
    files = glob.glob(os.path.join(src_folder, "*.wav"))
    print(f"Processing {len(files)} files: {src_folder} -> {dest_folder}")

    for f in files:
        try:
            clean, _ = librosa.load(f, sr=SR)
        except Exception as e:
            print(f"Skipping {f}: {e}")
            continue

        base = os.path.splitext(os.path.basename(f))[0]

        # Always copy the original clean clip into the split too
        sf.write(f"{dest_folder}/{base}_orig.wav", clean, SR)

        if not do_augment:
            continue

        if NOISE_FILES:
            for snr in [15, 5, 0]:
                noise, _ = librosa.load(random.choice(NOISE_FILES), sr=SR)
                mixed = add_noise(clean, noise, snr)
                sf.write(f"{dest_folder}/{base}_snr{snr}.wav", mixed, SR)

        shifted = np.roll(clean, random.randint(-800, 800))
        sf.write(f"{dest_folder}/{base}_shift.wav", shifted, SR)

        quieter = clean * random.uniform(0.5, 0.8)
        sf.write(f"{dest_folder}/{base}_vol.wav", quieter, SR)

if __name__ == "__main__":
    # Train and val get full augmentation
    augment_folder("dataset/splits/train_raw/positive", "dataset/splits/train/positive", do_augment=True)
    augment_folder("dataset/splits/train_raw/negative", "dataset/splits/train/negative", do_augment=True)
    augment_folder("dataset/splits/val_raw/positive", "dataset/splits/val/positive", do_augment=True)
    augment_folder("dataset/splits/val_raw/negative", "dataset/splits/val/negative", do_augment=True)

    # Test stays clean/unaugmented — it should reflect real conditions
    augment_folder("dataset/splits/test_raw/positive", "dataset/splits/test/positive", do_augment=False)
    augment_folder("dataset/splits/test_raw/negative", "dataset/splits/test/negative", do_augment=False)

    print("Augmentation complete.")
