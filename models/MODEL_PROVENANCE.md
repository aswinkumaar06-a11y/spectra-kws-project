# Spectra KWS — Model Provenance Statement

**Model Artifact:** `models/spectra_model.tflite`  
**File Size:** 39,552 bytes (~38.6 KB)  
**SHA-256 Checksum:** `b75295632240d022d77e3465a02b33ec7cfed01dd962aabace8e2ef15902ccc5`  
**Architecture:** DS-CNN (Depthwise Separable Convolutional Neural Network)  
**Input Contract:** `int8[1, 40, 32, 1]`, scale = 3.9962144, zero_point = 60  
**Output Contract:** `int8[1, 2]`, scale = 0.00390625, zero_point = -128  
**Evaluation Status:** `HOST-PASS` (frozen artifact, unretrained)

---

## 1. Retraining & Historical Timeline

### Was `models/spectra_model.tflite` retrained after the "accepts any keyword" behavior was observed?
**YES.** An earlier version of the model (`models/spectra_model_v1_leaky.tflite`, SHA-256: `7c2ebc2695997e8b74815806d63785f61c410b247ce3cc7c60cf2341b154b12f`) exhibited high false-positive sensitivity across unvoiced sounds and arbitrary non-keyword speech.

### When was it retrained?
- **Timestamp:** September 9, 2026 at 19:44:28 IST (committed in `c3e24c1e`).
- **Pipeline Scripts:** `python scripts/extract_features.py` (19:39:15) followed by `python scripts/train_model.py` (19:44:28).

### What training data and methodology changes were introduced?
1. **Added Hard Negatives (Phonetically Adjacent Words):**
   Generated via `scripts/generate_hard_negatives.py` using Windows SAPI TTS (Microsoft David and Microsoft Zira). Included confounder words: *expect, extra, inspection, inspector, respect, sector, spectacle, spectator, spectral, spectrum, speculate, vector*.
2. **Added Speech Negatives (Google Speech Commands v2):**
   Integrated 4,260 speech negative clips across 10 classes (*down, go, up, on, yes, right, stop, left, off, no*).
3. **Amplitude Normalization:**
   Added standard peak normalization ($x / \max(|x|)$) in `audio_utils.py` prior to MFCC extraction, eliminating loudness-induced false triggers.
4. **Dynamic Class Weighting:**
   Applied inverse frequency class weighting during Keras model fitting (`{0: 0.7836, 1: 1.3816}`) to prevent negative-class bias.
5. **Integer Quantization Calibration:**
   Calibrated with 200 representative balanced samples (`seed=42`) to produce `models/spectra_model.tflite` (`b7529563...`).

### Has `models/spectra_model.tflite` been retrained since September 9, 2026?
**NO.** The model binary has remained strictly frozen across Phase 0b, Phase 1, Phase 2, Phase 3, Phase 4, Phase 5, and Phase 6. Per project paradigm guidelines, **no further retraining or parameter tuning of the fixed-word "spectra" model will take place** pending the enrollment-based classifier decision.

---

## 2. Negative-Class Composition of Training Set

The training dataset utilized to produce `models/spectra_model.tflite` consisted of **7,842 total samples** (2,838 positive, 5,004 negative):

| Negative Category | Subcategory / Source | Clip Count | Percentage of Negatives |
|---|---|---|---|
| **Speech Negatives** | Google Speech Commands v2 (*down, go, up, on, yes, right, stop, left, off, no*) | 4,260 | 85.13% |
| **Phonetic Hard Negatives** | Synthetic confounders (Microsoft David @ 130–220 WPM, Microsoft Zira @ 130–160 WPM) | 360 | 7.19% |
| **Background Noise** | Domestic & mechanical noise (*dishes, exercise bike, running tap, white noise*) | 384 | 7.68% |
| **Total Negatives (Class 0)** | — | **5,004** | **100.00%** |

---

## 3. Relationship of Precision-Loop Negatives to Training Data

The 277 negative clips in the test split (`dataset/features/X_test.npy`, `y_test.npy`) relate to the training distribution as follows:
1. **Google Speech Commands v2 (253 clips):**
   Drawn from 14 distinct human speakers whose speaker IDs are strictly disjoint from all speakers in the training set (`train_test_leakage_count = 0`).
2. **Synthetic Confounders (24 clips):**
   Composed of phonetically adjacent words synthesized with Microsoft Zira at 190 WPM and 220 WPM. Because Microsoft Zira Desktop was present during training at 130 and 160 WPM, these 24 test clips evaluate generalization across unseen speaking rates of a known synthetic voice rather than an unseen synthetic speaker identity.
