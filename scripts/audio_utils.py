"""
Spectra KWS - Unified Audio Preprocessing and Feature Extraction Utilities
Single source of truth for training, offline testing, and live inference.
"""
import numpy as np
import librosa

SAMPLE_RATE = 16000
CLIP_DURATION = 1.0          # seconds
TARGET_SAMPLES = int(SAMPLE_RATE * CLIP_DURATION)  # 16000 samples
N_MFCC = 40
N_FFT = 1024
HOP_LENGTH = 512
N_FRAMES = 32                # fixed time frames for DS-CNN
ENERGY_GATE_THRESHOLD = 0.03 # min peak amplitude to process audio (below this = silence/noise)

def load_and_fix_audio(path_or_audio, sr=SAMPLE_RATE, duration=CLIP_DURATION):
    """
    Loads an audio file (or accepts an existing numpy array) and fixes length
    to exactly target_samples (16000 for 1.0s @ 16kHz).
    Pads with trailing zeros if shorter, truncates if longer.
    """
    if isinstance(path_or_audio, str):
        audio, _ = librosa.load(path_or_audio, sr=sr)
    else:
        audio = np.asarray(path_or_audio, dtype=np.float32)

    target_len = int(sr * duration)
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)), mode="constant")
    else:
        audio = audio[:target_len]
    return audio.astype(np.float32)

def normalize_amplitude(audio):
    """
    Peak-normalizes audio waveform to [-1.0, 1.0].
    Preserves relative wave shape while making spectral features invariant to recording gain.
    """
    max_amp = np.max(np.abs(audio))
    if max_amp > 1e-6:
        audio = audio / max_amp
    return audio.astype(np.float32)

def check_energy_gate(audio, threshold=ENERGY_GATE_THRESHOLD):
    """
    Returns True if audio has enough energy to contain speech (peak amp >= threshold).
    Returns False if audio is silence or very low-level background noise.
    """
    max_amp = np.max(np.abs(audio))
    return bool(max_amp >= threshold), float(max_amp)

def extract_mfcc(audio, sr=SAMPLE_RATE):
    """
    Extracts 40 MFCCs across 32 fixed time frames using 1024 FFT and 512 hop length.
    Input audio MUST be fixed-length (1.0s) and peak-normalized.
    Output shape: (N_MFCC, N_FRAMES) -> (40, 32)
    """
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )
    if mfcc.shape[1] < N_FRAMES:
        pad_width = N_FRAMES - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode="constant")
    else:
        mfcc = mfcc[:, :N_FRAMES]
    return mfcc.astype(np.float32)

def preprocess_audio_clip(path_or_audio, normalize=True):
    """
    Full standard preprocessing pipeline:
    1. Fix length to 16000 samples (1.0s @ 16kHz)
    2. Peak-normalize to [-1.0, 1.0] (if normalize=True)
    3. Extract 40x32 MFCC feature map
    """
    audio = load_and_fix_audio(path_or_audio)
    if normalize:
        audio = normalize_amplitude(audio)
    mfcc = extract_mfcc(audio)
    return mfcc

def quantize_to_int8(x, scale, zero_point):
    """
    Mathematically correct TFLite INT8 affine quantization:
    q = clip(round(x / scale) + zero_point, -128, 127)
    """
    q = np.round(x / scale) + zero_point
    return np.clip(q, -128, 127).astype(np.int8)

def dequantize_from_int8(q, scale, zero_point):
    """
    Mathematically correct TFLite INT8 dequantization:
    r = (q - zero_point) * scale
    """
    return (q.astype(np.float32) - zero_point) * scale
