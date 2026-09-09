"""
Converts every split .wav clip into a fixed-size MFCC feature array
and saves them as .npy files for training. Run AFTER split.py.
"""
import os
import glob
import numpy as np
import librosa

SR = 16000
DURATION = 1.0
N_MFCC = 40
N_FFT = 1024
HOP_LENGTH = 512
FIXED_FRAMES = 32   # pad/truncate every clip's MFCC to this many time frames

from audio_utils import preprocess_audio_clip, N_MFCC, N_FRAMES

FEATURES_DIR = "dataset/features"

def extract_mfcc(path):
    return preprocess_audio_clip(path, normalize=True)

import json

def build_split(split_name):
    X, y, files_meta = [], [], []
    for label, class_idx in [("positive", 1), ("negative", 0)]:
        folder = os.path.join("dataset/splits", split_name, label)
        files = sorted(glob.glob(os.path.join(folder, "*.wav")))
        print(f"{split_name}/{label}: {len(files)} files")
        for f in files:
            X.append(extract_mfcc(f))
            y.append(class_idx)
            files_meta.append({
                "filename": os.path.basename(f),
                "label": label,
                "is_hard_negative": ("synth_" in os.path.basename(f))
            })

    X = np.array(X)[..., np.newaxis]   # shape: (N, N_MFCC, FIXED_FRAMES, 1)
    y = np.array(y)
    return X, y, files_meta

if __name__ == "__main__":
    os.makedirs(FEATURES_DIR, exist_ok=True)
    for split_name in ["train", "val", "test"]:
        X, y, meta = build_split(split_name)
        np.save(os.path.join(FEATURES_DIR, f"X_{split_name}.npy"), X)
        np.save(os.path.join(FEATURES_DIR, f"y_{split_name}.npy"), y)
        if split_name == "test":
            manifest_path = os.path.join(FEATURES_DIR, "test_manifest.json")
            with open(manifest_path, "w") as fp:
                json.dump(meta, fp, indent=2)
            print(f"Saved {manifest_path}")
        print(f"Saved {split_name}: X{X.shape}, y{y.shape}")
