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

FEATURES_DIR = "dataset/features"

def extract_mfcc(path):
    audio, _ = librosa.load(path, sr=SR)
    target_len = int(SR * DURATION)
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)))
    else:
        audio = audio[:target_len]

    mfcc = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=N_MFCC,
                                 n_fft=N_FFT, hop_length=HOP_LENGTH)
    if mfcc.shape[1] < FIXED_FRAMES:
        pad_width = FIXED_FRAMES - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)))
    else:
        mfcc = mfcc[:, :FIXED_FRAMES]
    return mfcc.astype(np.float32)

def build_split(split_name):
    X, y = [], []
    for label, class_idx in [("positive", 1), ("negative", 0)]:
        folder = os.path.join("dataset/splits", split_name, label)
        files = glob.glob(os.path.join(folder, "*.wav"))
        print(f"{split_name}/{label}: {len(files)} files")
        for f in files:
            X.append(extract_mfcc(f))
            y.append(class_idx)

    X = np.array(X)[..., np.newaxis]   # shape: (N, N_MFCC, FIXED_FRAMES, 1)
    y = np.array(y)
    return X, y

if __name__ == "__main__":
    os.makedirs(FEATURES_DIR, exist_ok=True)
    for split_name in ["train", "val", "test"]:
        X, y = build_split(split_name)
        np.save(os.path.join(FEATURES_DIR, f"X_{split_name}.npy"), X)
        np.save(os.path.join(FEATURES_DIR, f"y_{split_name}.npy"), y)
        print(f"Saved {split_name}: X{X.shape}, y{y.shape}")
