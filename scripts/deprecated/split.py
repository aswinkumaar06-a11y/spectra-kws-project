"""
Speaker-aware train/val/test split. Merges negative_words + noise +
silence into one combined "negative" class. Downsamples negatives to
roughly match positive count for balanced training.
Run AFTER augment.py.
"""
import os
import re
import glob
import random
import shutil

random.seed(42)
DEST = "dataset/splits"

# Ratio of negatives to positives (2:1 gives good coverage without drowning)
NEG_TO_POS_RATIO = 2.0

def speaker_id_from_filename(path):
    """Extract 'speakerNN' if present, else fall back to filename stem
    (works for your own recordings AND Speech Commands hash-based names)."""
    name = os.path.basename(path)
    m = re.search(r"speaker(\d+)", name)
    if m:
        return f"speaker{m.group(1)}"
    # Speech Commands style: <hash>_nohash_<n>.wav -> hash is the speaker id
    m2 = re.match(r"([a-f0-9]+)_", name)
    if m2:
        return m2.group(1)
    return name  # unique fallback, treated as its own "speaker"

def split_by_speaker(files, train_r=0.7, val_r=0.15):
    speakers = {}
    for f in files:
        sid = speaker_id_from_filename(f)
        speakers.setdefault(sid, []).append(f)

    ids = list(speakers.keys())
    random.shuffle(ids)
    n = len(ids)
    train_ids = set(ids[:int(train_r * n)])
    val_ids = set(ids[int(train_r * n):int((train_r + val_r) * n)])

    train, val, test = [], [], []
    for sid, flist in speakers.items():
        if sid in train_ids:
            train.extend(flist)
        elif sid in val_ids:
            val.extend(flist)
        else:
            test.extend(flist)
    return train, val, test

def copy_all(files, split_name, label):
    out_dir = os.path.join(DEST, split_name, label)
    os.makedirs(out_dir, exist_ok=True)
    for f in files:
        shutil.copy(f, out_dir)

def collect(*folders):
    files = []
    for folder in folders:
        files.extend(glob.glob(os.path.join(folder, "**", "*.wav"), recursive=True))
    return files

def downsample(files, target_count):
    """Randomly downsample a list to target_count."""
    if len(files) <= target_count:
        return files
    return random.sample(files, target_count)

if __name__ == "__main__":
    # --- Clean old splits ---
    for split in ["train", "val", "test"]:
        for label in ["positive", "negative"]:
            d = os.path.join(DEST, split, label)
            if os.path.exists(d):
                shutil.rmtree(d)

    # --- Positive class: raw + augmented ---
    positive_files = collect("dataset/raw/positive", "dataset/augmented/positive")
    train_p, val_p, test_p = split_by_speaker(positive_files)
    copy_all(train_p, "train", "positive")
    copy_all(val_p, "val", "positive")
    copy_all(test_p, "test", "positive")
    print(f"Positive -> train:{len(train_p)} val:{len(val_p)} test:{len(test_p)}")

    total_positive = len(train_p) + len(val_p) + len(test_p)
    target_negative = int(total_positive * NEG_TO_POS_RATIO)

    # --- Negative class: negative_words + noise + silence, downsampled ---
    negative_files = collect(
        "dataset/raw/negative_words",
        "dataset/raw/noise", "dataset/raw/silence"
    )
    print(f"Total raw negatives available: {len(negative_files)}")
    negative_files = downsample(negative_files, target_negative)
    print(f"Downsampled negatives to: {len(negative_files)} (ratio {NEG_TO_POS_RATIO}:1)")

    train_n, val_n, test_n = split_by_speaker(negative_files)
    copy_all(train_n, "train", "negative")
    copy_all(val_n, "val", "negative")
    copy_all(test_n, "test", "negative")
    print(f"Negative -> train:{len(train_n)} val:{len(val_n)} test:{len(test_n)}")
