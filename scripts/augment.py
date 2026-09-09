"""
Augments raw positive clips with noise mixing, time-shifting, and
volume changes. Only augments the POSITIVE (minority) class — the
negative class already has 38K+ raw samples.  Run this BEFORE split.py.
"""
import os
import glob
import random
import numpy as np
import librosa
import soundfile as sf

SR = 16000
NOISE_FILES = glob.glob("dataset/raw/noise/*.wav")

def add_noise(clean, noise, snr_db):
    noise = np.resize(noise, clean.shape)
    clean_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2) + 1e-9
    factor = np.sqrt(clean_power / (noise_power * 10 ** (snr_db / 10)))
    return clean + factor * noise

def augment_folder(src_folder, dest_folder):
    os.makedirs(dest_folder, exist_ok=True)
    files = glob.glob(os.path.join(src_folder, "**", "*.wav"), recursive=True)
    print(f"Augmenting {len(files)} files from {src_folder} -> {dest_folder}")

    for f in files:
        try:
            clean, _ = librosa.load(f, sr=SR)
        except Exception as e:
            print(f"Skipping {f}: {e}")
            continue

        base = os.path.splitext(os.path.basename(f))[0]

        # Noise-mixed versions at a few SNR levels
        if NOISE_FILES:
            for snr in [15, 5, 0]:
                noise, _ = librosa.load(random.choice(NOISE_FILES), sr=SR)
                mixed = add_noise(clean, noise, snr)
                sf.write(f"{dest_folder}/{base}_snr{snr}.wav", mixed, SR)

        # Time-shifted version
        shifted = np.roll(clean, random.randint(-800, 800))
        sf.write(f"{dest_folder}/{base}_shift.wav", shifted, SR)

        # Volume-reduced version
        quieter = clean * random.uniform(0.5, 0.8)
        sf.write(f"{dest_folder}/{base}_vol.wav", quieter, SR)

if __name__ == "__main__":
    # Only augment positives — negatives already have 38K+ samples
    augment_folder("dataset/raw/positive", "dataset/augmented/positive")
    print("Done.")
