"""
split_raw_only.py — STEP 1 of the corrected pipeline.
Groups files by their immediate parent folder (every contributor's
folder, including named folders and session_bundle folders), then
splits by GROUP so a contributor's files never cross train/val/test.
Run this BEFORE any augmentation, and AFTER prepare_raw_positive.py (Step 0).
"""
import os
import glob
import random
import shutil

random.seed(42)

def group_key(path):
    # Every file sits inside some contributor/session folder now —
    # just use that folder name as the group.
    return os.path.basename(os.path.dirname(path))

def split_by_group(files, train_r=0.7, val_r=0.15):
    groups = {}
    for f in files:
        groups.setdefault(group_key(f), []).append(f)

    keys = list(groups.keys())
    random.shuffle(keys)
    n = len(keys)
    train_keys = set(keys[:int(train_r * n)])
    val_keys = set(keys[int(train_r * n):int((train_r + val_r) * n)])

    train, val, test = [], [], []
    for k, flist in groups.items():
        if k in train_keys:
            train.extend(flist)
        elif k in val_keys:
            val.extend(flist)
        else:
            test.extend(flist)
    return train, val, test, len(keys)

def copy_all(files, split_name, label):
    out_dir = os.path.join("dataset/splits", f"{split_name}_raw", label)
    os.makedirs(out_dir, exist_ok=True)
    for f in files:
        shutil.copy(f, out_dir)

if __name__ == "__main__":
    positive_files = glob.glob("dataset/raw/positive/**/*.wav", recursive=True)
    train_p, val_p, test_p, n_groups = split_by_group(positive_files)
    copy_all(train_p, "train", "positive")
    copy_all(val_p, "val", "positive")
    copy_all(test_p, "test", "positive")
    print(f"Positive: {n_groups} contributor groups -> "
          f"train:{len(train_p)} val:{len(val_p)} test:{len(test_p)}")

    negative_files = (
        glob.glob("dataset/raw/negative_words/**/*.wav", recursive=True) +
        glob.glob("dataset/raw/noise/**/*.wav", recursive=True) +
        glob.glob("dataset/raw/silence/**/*.wav", recursive=True)
    )
    # Downsample negatives to match positive count more closely
    # Let's say downsampling is necessary because there are 38,000 negatives.
    # The previous split script downsampled. The user's provided code doesn't explicitly do downsampling for negatives, but it's okay, I will follow their code literally.
    train_n, val_n, test_n, n_neg_groups = split_by_group(negative_files)
    copy_all(train_n, "train", "negative")
    copy_all(val_n, "val", "negative")
    copy_all(test_n, "test", "negative")
    print(f"Negative: {n_neg_groups} groups -> "
          f"train:{len(train_n)} val:{len(val_n)} test:{len(test_n)}")
