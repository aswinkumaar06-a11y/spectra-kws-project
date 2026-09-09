"""
augment_per_split.py — STEP 2. Augments each split independently so
augmented copies can never leak into a different split than their
source file. Run AFTER split_raw_only.py.

Fixes D04: Partitions background noise files so train noise is mutually
exclusive from val and test noise.
"""
import os
import glob
import json
import random
import shutil
import numpy as np
import librosa
import soundfile as sf

SR = 16000

def load_noise_pools():
    manifest_path = "dataset/splits/noise_pools.json"
    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as fp:
            manifest = json.load(fp)
        # Convert filenames to paths
        all_noise = glob.glob("dataset/raw/noise/*.wav") + glob.glob("dataset/raw/noise/_background_noise_/*.wav")
        noise_map = {os.path.basename(f): f for f in all_noise}
        pools = {}
        for split, names in manifest.items():
            pools[split] = [noise_map[n] for n in names if n in noise_map]
        return pools

    # Fallback if manifest is missing: split dynamically
    all_noise = list(set(glob.glob("dataset/raw/noise/*.wav") + glob.glob("dataset/raw/noise/_background_noise_/*.wav")))
    unique_names = sorted(list(set(os.path.basename(f) for f in all_noise)))
    rng = random.Random(42)
    rng.shuffle(unique_names)
    test_n = set(unique_names[:1])
    val_n = set(unique_names[1:2])
    train_n = set(unique_names[2:])
    noise_map = {os.path.basename(f): f for f in all_noise}
    return {
        "train": [noise_map[n] for n in train_n if n in noise_map],
        "val": [noise_map[n] for n in val_n if n in noise_map],
        "test": [noise_map[n] for n in test_n if n in noise_map]
    }

def add_noise(clean, noise, snr_db):
    noise = np.resize(noise, clean.shape)
    clean_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2) + 1e-9
    factor = np.sqrt(clean_power / (noise_power * 10 ** (snr_db / 10)))
    return clean + factor * noise

def augment_folder(src_folder, dest_folder, noise_files=None, do_augment=True, seed=42):
    # Wipe destination directory first to eliminate stale outputs (D03)
    if os.path.exists(dest_folder):
        shutil.rmtree(dest_folder)
    os.makedirs(dest_folder, exist_ok=True)

    rng = random.Random(seed)
    files = sorted(glob.glob(os.path.join(src_folder, "*.wav")))
    print(f"Processing {len(files)} files: {src_folder} -> {dest_folder} (augment={do_augment}, noise_pool_size={len(noise_files) if noise_files else 0})")

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

        if noise_files:
            for snr in [15, 5, 0]:
                chosen_noise_path = rng.choice(noise_files)
                noise, _ = librosa.load(chosen_noise_path, sr=SR)
                mixed = add_noise(clean, noise, snr)
                sf.write(f"{dest_folder}/{base}_snr{snr}.wav", mixed, SR)

        shifted = np.roll(clean, rng.randint(-800, 800))
        sf.write(f"{dest_folder}/{base}_shift.wav", shifted, SR)

        quieter = clean * rng.uniform(0.5, 0.8)
        sf.write(f"{dest_folder}/{base}_vol.wav", quieter, SR)

if __name__ == "__main__":
    pools = load_noise_pools()
    print("Loaded Disjoint Noise Pools:")
    for split, nfiles in pools.items():
        print(f"  {split}: {[os.path.basename(x) for x in nfiles]}")

    # Train gets train noise only
    augment_folder("dataset/splits/train_raw/positive", "dataset/splits/train/positive", noise_files=pools.get("train", []), do_augment=True, seed=42)
    augment_folder("dataset/splits/train_raw/negative", "dataset/splits/train/negative", noise_files=pools.get("train", []), do_augment=True, seed=43)

    # Val gets val noise only
    augment_folder("dataset/splits/val_raw/positive", "dataset/splits/val/positive", noise_files=pools.get("val", []), do_augment=True, seed=44)
    augment_folder("dataset/splits/val_raw/negative", "dataset/splits/val/negative", noise_files=pools.get("val", []), do_augment=True, seed=45)

    # Test stays clean/unaugmented — reflects real testing conditions
    augment_folder("dataset/splits/test_raw/positive", "dataset/splits/test/positive", noise_files=[], do_augment=False)
    augment_folder("dataset/splits/test_raw/negative", "dataset/splits/test/negative", noise_files=[], do_augment=False)

    print("\nAugmentation complete with strict disjoint noise pools (D04 resolved).")
