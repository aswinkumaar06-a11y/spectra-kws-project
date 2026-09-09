"""
split_raw_only.py — STEP 1 of the corrected pipeline.
Performs strict speaker/source-level grouped splitting to guarantee:
1. Zero speaker overlap across train/val/test splits (resolves D02).
2. Complete prevention of filename collisions by prefixing with group ID (resolves D03).
3. Clean wiping of destination folders before writing to eliminate stale artifacts (resolves D03).
4. Disjoint partitioning of background noise pools across train/val/test (resolves D04).
5. Balanced downsampling of negatives to ~2x positive count (respecting speaker groups).
"""

import os
import glob
import json
import random
import shutil

random.seed(42)

SPLITS_DIR = "dataset/splits"
NEG_TO_POS_RATIO = 2.0  # 2 negative clips per positive clip

def wipe_directory(path):
    """Safely wipe directory to prevent stale output."""
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

def positive_group_key(path):
    """
    Identifies speaker/session group for positive recordings.
    If file is in a subfolder (e.g. contributor or session_bundle), use that subfolder.
    If file is directly in dataset/raw/positive, treat file stem as its own group.
    """
    rel = os.path.relpath(path, "dataset/raw/positive")
    parts = rel.split(os.sep)
    if len(parts) > 1:
        return parts[0]
    return os.path.splitext(parts[0])[0]

def negative_group_key(path):
    """
    Identifies speaker/source group for negative recordings.
    - Speech Commands files: use speaker ID prefix before '_nohash_'.
    - Hard negatives: use synthetic voice folder name.
    - Silence / other: use parent folder name.
    """
    base = os.path.basename(path)
    if "_nohash_" in base:
        return base.split("_nohash_")[0]
    parent = os.path.basename(os.path.dirname(path))
    return parent if parent else "misc_neg"

def downsample_groups(files, group_fn, target_count, seed=42):
    """
    Downsamples negatives to target_count while preserving complete speaker groups.
    """
    if len(files) <= target_count:
        return files
    rng = random.Random(seed)
    groups = {}
    for f in files:
        g = group_fn(f)
        groups.setdefault(g, []).append(f)

    keys = sorted(list(groups.keys()))
    rng.shuffle(keys)

    selected = []
    for k in keys:
        selected.extend(groups[k])
        if len(selected) >= target_count:
            break
    return selected

def split_by_group(files, group_fn, train_r=0.70, val_r=0.15, seed=42):
    rng = random.Random(seed)
    groups = {}
    for f in files:
        g = group_fn(f)
        groups.setdefault(g, []).append(f)

    keys = sorted(list(groups.keys()))
    rng.shuffle(keys)
    n = len(keys)
    n_train = max(1, int(train_r * n))
    n_val = max(1, int(val_r * n))

    train_keys = set(keys[:n_train])
    val_keys = set(keys[n_train:n_train + n_val])
    test_keys = set(keys[n_train + n_val:])

    train, val, test = [], [], []
    for k in keys:
        flist = groups[k]
        if k in train_keys:
            train.extend(flist)
        elif k in val_keys:
            val.extend(flist)
        else:
            test.extend(flist)

    return train, val, test, len(keys)

def copy_with_safe_prefix(files, group_fn, split_name, label):
    out_dir = os.path.join(SPLITS_DIR, f"{split_name}_raw", label)
    os.makedirs(out_dir, exist_ok=True)
    for f in files:
        grp = group_fn(f)
        parent_dir = os.path.basename(os.path.dirname(f))
        orig_name = os.path.basename(f)
        if parent_dir and parent_dir != grp and parent_dir not in ("positive", "negative_words"):
            safe_name = f"{grp}__{parent_dir}__{orig_name}"
        else:
            safe_name = f"{grp}__{orig_name}"
        dst_path = os.path.join(out_dir, safe_name)
        shutil.copy(f, dst_path)

def partition_noise_pools(seed=42):
    """
    Disjointly partition noise files across train, val, and test splits (D04).
    Guarantees no background noise heard in training is present in test.
    """
    all_noise = sorted(list(set(glob.glob("dataset/raw/noise/*.wav") + glob.glob("dataset/raw/noise/_background_noise_/*.wav"))))
    unique_basenames = sorted(list(set(os.path.basename(f) for f in all_noise)))

    rng = random.Random(seed)
    rng.shuffle(unique_basenames)

    # For 6 distinct noise types: 4 train, 1 val, 1 test
    n_test = 1
    n_val = 1
    test_noise_names = set(unique_basenames[:n_test])
    val_noise_names = set(unique_basenames[n_test:n_test + n_val])
    train_noise_names = set(unique_basenames[n_test + n_val:])

    pools = {"train": [], "val": [], "test": []}
    for f in all_noise:
        name = os.path.basename(f)
        if name in train_noise_names and f not in pools["train"]:
            pools["train"].append(f)
        elif name in val_noise_names and f not in pools["val"]:
            pools["val"].append(f)
        elif name in test_noise_names and f not in pools["test"]:
            pools["test"].append(f)

    noise_manifest_path = os.path.join(SPLITS_DIR, "noise_pools.json")
    with open(noise_manifest_path, "w") as fp:
        json.dump({k: [os.path.basename(x) for x in v] for k, v in pools.items()}, fp, indent=2)

    print("\n--- Disjoint Noise Pool Partitioning (D04) ---")
    for k, v in pools.items():
        print(f"  {k} noise pool ({len(v)} files): {[os.path.basename(x) for x in v]}")
    return pools

if __name__ == "__main__":
    print("=" * 60)
    print("  Spectra KWS - Step 1: Safe Group-Based Splitting")
    print("=" * 60)

    # Wipe splits directory completely to eliminate stale outputs (D03)
    wipe_directory(SPLITS_DIR)

    # 1. Positive clips
    positive_files = sorted(glob.glob("dataset/raw/positive/**/*.wav", recursive=True))
    train_p, val_p, test_p, n_pos_groups = split_by_group(positive_files, positive_group_key, train_r=0.70, val_r=0.15)
    copy_with_safe_prefix(train_p, positive_group_key, "train", "positive")
    copy_with_safe_prefix(val_p, positive_group_key, "val", "positive")
    copy_with_safe_prefix(test_p, positive_group_key, "test", "positive")
    print(f"Positive: {n_pos_groups} speaker groups ({len(positive_files)} files) -> Train: {len(train_p)}, Val: {len(val_p)}, Test: {len(test_p)}")

    # 2. Negative speech clips: Partition Hard Negatives by voice group (D02) + downsample speech commands
    hard_neg_files = sorted(glob.glob("dataset/raw/negative_words/hard_negatives/**/*.wav", recursive=True))
    other_neg_files = sorted(
        [f for f in glob.glob("dataset/raw/negative_words/**/*.wav", recursive=True) if "hard_negatives" not in f] +
        glob.glob("dataset/raw/silence/**/*.wav", recursive=True)
    )

    if hard_neg_files:
        train_hn, val_hn, test_hn, n_hn_groups = split_by_group(hard_neg_files, negative_group_key, train_r=0.60, val_r=0.15)
        print(f"Hard Negatives: {n_hn_groups} voice groups ({len(hard_neg_files)} files) -> Train: {len(train_hn)}, Val: {len(val_hn)}, Test: {len(test_hn)}")
    else:
        train_hn, val_hn, test_hn = [], [], []

    target_negs = max(0, int(len(positive_files) * NEG_TO_POS_RATIO) - len(hard_neg_files))
    downsampled_other = downsample_groups(other_neg_files, negative_group_key, target_negs, seed=42)
    train_on, val_on, test_on, n_on_groups = split_by_group(downsampled_other, negative_group_key, train_r=0.70, val_r=0.15)

    train_n = train_hn + train_on
    val_n = val_hn + val_on
    test_n = test_hn + test_on
    copy_with_safe_prefix(train_n, negative_group_key, "train", "negative")
    copy_with_safe_prefix(val_n, negative_group_key, "val", "negative")
    copy_with_safe_prefix(test_n, negative_group_key, "test", "negative")
    print(f"Total Negative: Train: {len(train_n)}, Val: {len(val_n)}, Test: {len(test_n)} (includes {len(test_hn)} held-out hard negatives in test!)")

    # 3. Partition noise pools into mutually exclusive sets (D04)
    partition_noise_pools()

    print("\nStage 1 Splitting complete. Output written to dataset/splits/*_raw/.")
    print("=" * 60)
