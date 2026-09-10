# Spectra KWS — Dataset Datasheet

**Dataset Name:** Spectra Keyword Spotting Audio Dataset  
**Target Wake-Word:** *"Spectra"* (Class 1)  
**Negative Class:** General English speech, phonetically adjacent confounders, background acoustic noise (Class 0)  
**Format:** 16,000 Hz, 1.0 second duration (16,000 samples), 16-bit signed PCM mono  
**Feature Representation:** 40-band MFCC (1024-pt FFT, 512-pt hop, 32 frames)

---

## 1. Positive Keyword ("Spectra") Audio

- **Total Positive Utterance Clips:**
  - Training set: **2,838** clips (includes clean raw and noise/shift augmented derivatives)
  - Validation set: **216** clips
  - Test set: **119** clips
  - Total across splits: **3,173** positive clips (derived from 154 base recording sessions)
- **Number of Unique Human Speakers:**
  - Training: **116** unique anonymous human speakers
  - Validation: **24** unique anonymous human speakers
  - Test: **14** unique anonymous human speakers
  - Total Unique Speakers: **154** (anonymized consistently via SHA-256 hashes `SPK_POS_XXXXXXXX`)
- **Recording Modality & Environments:**
  - Collected via web browser audio recorder (`Data Collection.html`) and local test scripts.
  - Microphones: Built-in laptop microphones, smartphone condenser mics, and USB headsets.
  - Acoustic Environments: Realistic indoor living rooms, domestic kitchens, quiet offices, and multi-speaker open areas.
  - Distances: Near-field (0.3 m to 1.0 m) and mid-field (1.5 m to 3.0 m).

---

## 2. Negative Class Audio

Total negative audio samples across splits: **6,151 clips**

| Split | Category | Source Description | Clip Count |
|---|---|---|---|
| **Train** | Speech Negatives | Google Speech Commands v2 (*down, go, up, on, yes, right, stop, left, off, no*) | 4,260 |
| **Train** | Background Noise | 4 acoustic noise recordings (*doing_the_dishes, exercise_bike, running_tap, white_noise*) | 384 |
| **Train** | Phonetic Hard Negatives | Microsoft David (130, 160, 190, 220 WPM) & Microsoft Zira (130, 160 WPM) | 360 |
| **Val** | Speech Negatives | Google Speech Commands v2 (disjoint human speakers) | 798 |
| **Val** | Background Noise | 1 held-out noise recording (*dude_miaowing*) | 0 (val raw) |
| **Val** | Phonetic Hard Negatives | Microsoft David & Zira synthetic confounders | 72 |
| **Test** | Speech Negatives | Google Speech Commands v2 across 14 unseen human speakers | 253 |
| **Test** | Phonetic Hard Negatives | Microsoft Zira synthetic confounders @ 190 & 220 WPM | 24 |

---

## 3. Augmentation Pipeline

The augmentation pipeline is executed per-split in `scripts/augment_per_split.py`:
1. **Time-Shifting:**
   Random cyclic roll within $[-1600, +1600]$ samples ($\pm 100\text{ ms}$).
2. **Pitch-Shifting:**
   Small semitone perturbations ($\pm 1.0$ semitones) on raw audio.
3. **Additive Background Noise:**
   Mixed at Signal-to-Noise Ratios (SNR) of $0\text{ dB}$, $5\text{ dB}$, and $10\text{ dB}$.
   Noise audio is sampled strictly from split-isolated pools defined in `dataset/splits/noise_pools.json`:
   - **Train Pool:** `doing_the_dishes.wav`, `exercise_bike.wav`, `running_tap.wav`, `white_noise.wav`
   - **Val Pool:** `dude_miaowing.wav`
   - **Test Pool:** Clean test audio only (`pink_noise.wav` reserved, no synthetic noise injected into held-out evaluation).

---

## 4. Leakage Statement & Bounding Claims

### Base Recording Disjointness
- Base recordings were partitioned into Train, Validation, and Test sets by `scripts/split_raw_only.py` **prior to any augmentation**.
- All augmented variants of any given base recording remain strictly inside that recording's assigned split. No augmented copy of a training utterance exists in the test split.

### Speaker Disjointness
- **Human Speakers:**
  `train_test_leakage_count = 0`, `train_val_leakage_count = 0`, and `val_test_leakage_count = 0`.
  Every human speaker in the test set (14 speakers) is strictly disjoint from all human speakers in the training set (116 speakers).
- **Synthetic Speakers (Explicit Scope Boundary):**
  Microsoft Zira Desktop was present in the training set at 130 and 160 WPM. The 24 synthetic hard negative test clips evaluate held-out speaking rates (190 and 220 WPM) of that known synthetic voice. **This evaluates speaking-rate generalization on a known voice, not an unseen synthetic speaker.**
