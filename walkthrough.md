# Spectra KWS — Post-Handover Upgrades & Verification Walkthrough

## 1. Overview of Accomplishments

In response to Arena.ai's code review and post-handover action items, the following comprehensive fixes and validations were executed across the Spectra KWS project:

1. **Dynamic Class Balancing (M08):** Computed inverse-frequency class weights ($W_{\text{neg}}=0.7836$, $W_{\text{pos}}=1.3816$) directly from training split labels (`y_train`) to correct majority-class bias without arbitrary hardcoding.
2. **Quantitative INT8 TFLite Evaluation Script (M04 / M07):** Built `scripts/evaluate_tflite.py` with exact affine quantization math. Resolved previous INT8 recall collapse (which suffered from 69.03% recall and 118 false negatives) by implementing a balanced representative calibration dataset generator (`seed=42`, 200 samples). The production INT8 model achieved **99.75% accuracy**, **100.0% recall** (0 false negatives), and **0.9917 precision** on the held-out test split.
3. **Noise Pool & Group Leakage Elimination (D04 / D02 / D03):** 
   - Partitioned background noise into mutually exclusive train and validation pools, recorded in `dataset/splits/noise_pools.json`. Kept the test split 100% clean.
   - Formatted copied files as `{grp}__{word}__{orig_name}` to prevent inter-word filename collisions across Google Speech Commands.
   - Grouped synthetic voice profiles and human speaker directories to ensure zero cross-split leakage.
4. **Phonetically Adjacent Hard Negatives (Step 4):**
   - Synthesized 96 audio clips across 12 phonetically adjacent words (*spectrum, spectacle, inspect, inspector, specter, extra, suspect, perspective, respect, sector, nectar, vector*) over 8 distinct voice profiles using Windows SAPI TTS.
   - Kept 2 voice profiles completely held-out in the test split.
   - Achieved **0 False Triggers** across all 24 held-out hard negative test samples (**0.00% False Activation Rate**).
5. **Preprocessing & Feature Parity Unification (M02 / M05):**
   - Centralized audio normalization, length fixing (1.0s @ 16kHz), 40x32 MFCC extraction, and energy gating in `scripts/audio_utils.py`.
   - Verified bit-identical feature extraction between training and live inference ($0.000000$ max difference).
6. **Live Streaming Latency & Queue Stability (M06):**
   - Verified that processing latency averages **12.32 ms** (peak 74.05 ms) per 500 ms audio hop, with maximum queue depth strictly bounded at 1.

---

## 2. Changes Made

### A. Core Scripts & Library Modules
- [audio_utils.py](file:///d:/spectra-kws-project/scripts/audio_utils.py): Centralized audio processing library (1.0s @ 16kHz, peak normalization `[-1, 1]`, energy gate at threshold=0.03, MFCC extraction, and affine INT8 quantization/dequantization).
- [evaluate_tflite.py](file:///d:/spectra-kws-project/scripts/evaluate_tflite.py): Standalone quantitative evaluation tool for INT8 TFLite models against numpy test splits, computing accuracy, precision, recall, specificity, F1, confusion matrix, and hard-negative false activation rate.
- [generate_hard_negatives.py](file:///d:/spectra-kws-project/scripts/generate_hard_negatives.py): Procedural TTS generator for multi-voice confounder words across isolated voice profiles.
- [prepare_raw_positive.py](file:///d:/spectra-kws-project/scripts/prepare_raw_positive.py): Non-destructive raw audio preparation scanning nested directories and copying normalized 16kHz WAVs.
- [split_raw_only.py](file:///d:/spectra-kws-project/scripts/split_raw_only.py): Partitioning with directory pre-wiping, group-aware speaker hashing, collision prevention (`{grp}__{word}__{orig_name}`), and mutual-exclusive noise pool assignment.
- [augment_per_split.py](file:///d:/spectra-kws-project/scripts/augment_per_split.py): Enforces split-isolated noise augmentation using `noise_pools.json`.
- [extract_features.py](file:///d:/spectra-kws-project/scripts/extract_features.py): Feature extraction refactored to use `audio_utils.py` and output `test_manifest.json`.
- [train_model.py](file:///d:/spectra-kws-project/scripts/train_model.py): Dynamic class weighting, balanced representative calibration (seed=42, 200 samples), and SHA256 manifest logging.
- [run_pipeline.bat](file:///d:/spectra-kws-project/run_pipeline.bat): Hardened with errorlevel checks after every step and calls to modern scripts.

### B. Regression Test Suite
- [test_feature_parity.py](file:///d:/spectra-kws-project/tests/test_feature_parity.py): Asserts max absolute difference between training and live MFCC features is $0.000000$, and validates affine INT8 quantization math.
- [test_model_cli_and_quant.py](file:///d:/spectra-kws-project/tests/test_model_cli_and_quant.py): Asserts `--model` flag routing and INT8 clipping/rounding within `[-128, 127]`.
- [test_energy_gate.py](file:///d:/spectra-kws-project/tests/test_energy_gate.py): Verifies energy gate attenuation across amplitude scales from 1.0 down to 0.01.
- [test_live_buffering.py](file:///d:/spectra-kws-project/tests/test_live_buffering.py): Verifies real-time streaming queue boundedness (depth $\le 1$) and latency well under 500 ms.

---

## 3. Empirical Validation Results

### A. Full Unit Test Suite
```powershell
d:/spectra-kws-project/venv/Scripts/python.exe -m unittest discover tests/
Ran 10 tests in 9.902s
OK
```

### B. Float32 vs. INT8 TFLite Quantitative Comparison (Held-Out Test Split)

| Metric | Keras Float32 | TFLite INT8 | Delta |
| :--- | :--- | :--- | :--- |
| **Accuracy** | 100.0% (396/396) | 99.75% (395/396) | -0.25% |
| **Precision** | 1.0000 | 0.9917 | -0.0083 |
| **Recall** | 1.0000 | 1.0000 | 0.0000 |
| **Specificity** | 1.0000 | 0.9964 | -0.0036 |
| **F1-Score** | 1.0000 | 0.9958 | -0.0042 |
| **True Positives (TP)** | 119 | 119 | 0 |
| **False Positives (FP)** | 0 | 1 | +1 |
| **False Negatives (FN)** | 0 | 0 | 0 |
| **True Negatives (TN)** | 277 | 276 | -1 |
| **Model Size** | 341.6 KB | 38.6 KB | **-88.7%** |

### C. Hard Negatives Breakdown (Held-Out Synthetic Voices)
```
  Total Hard Negatives in Test: 24
  False Triggers (FP):          0
  Correct Rejections (TN):       24
  False Activation Rate (FAR):  0.00%
  True Rejection Rate:          100.00%
```

### D. Streaming Latency & Real-Time Backlog
```
[Sustained Stream Test] 20 steps (10s audio):
  Average processing latency: 12.32 ms
  Maximum processing latency: 74.05 ms
  Max queue depth:            1
  Budget per hop:             500.00 ms
```

---

## 4. Artifact & Model Cryptographic Hashes

From [model_manifest.json](file:///d:/spectra-kws-project/models/model_manifest.json):
- **Float32 H5 Model (`models/spectra_model.h5`):** `a76587d11fbacce626f141ad6e435c195d469041ca701476445489a4da947f02`
- **INT8 TFLite Model (`models/spectra_model.tflite`):** `b75295632240d022d77e3465a02b33ec7cfed01dd962aabace8e2ef15902ccc5`
- **C-Array Header (`models/spectra_model.cc`):** `560c14e1377a3f0e759f2c9552fc8318bc856c4136f21a55634ae794f67a1b5f`
