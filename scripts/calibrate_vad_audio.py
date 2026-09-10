#!/usr/bin/env python3
"""
calibrate_vad_audio.py — P6 VAD Acoustic Calibration Tool.

Analyzes recorded room-tone/silence audio:
  1. Segments audio into 100 ms RMS frames (1,600 samples @ 16 kHz)
  2. Computes the 99th percentile of background silence RMS: p99_silence
  3. Applies the exact formula: SPECTRA_VAD_THRESH_RMS = max(p99_silence * 4.0, 0.01)
  4. Validates that the threshold satisfies safety bounds [0.010, 0.060]

Usage:
  python scripts/calibrate_vad_audio.py --audio room_silence.wav
  python scripts/calibrate_vad_audio.py --demo
"""

import os
import sys
import argparse
import numpy as np
import soundfile as sf

FRAME_SAMPLES = 1600  # 100 ms @ 16 kHz


def compute_rms_frames(audio_data):
    n_frames = len(audio_data) // FRAME_SAMPLES
    rms_values = []
    for i in range(n_frames):
        chunk = audio_data[i * FRAME_SAMPLES : (i + 1) * FRAME_SAMPLES]
        rms = np.sqrt(np.mean(chunk**2))
        rms_values.append(rms)
    return np.array(rms_values)


def calibrate_from_audio(audio_data, sr=16000):
    if sr != 16000:
        raise ValueError(f"Expected 16 kHz audio, got {sr} Hz")

    if audio_data.ndim > 1:
        # Gracefully downmix stereo/multichannel to mono
        audio_data = np.mean(audio_data, axis=1)

    rms_arr = compute_rms_frames(audio_data)
    if len(rms_arr) == 0:
        raise ValueError("Audio too short for 100 ms frame analysis")

    p50 = float(np.percentile(rms_arr, 50))
    p90 = float(np.percentile(rms_arr, 90))
    p99 = float(np.percentile(rms_arr, 99))
    max_rms = float(np.max(rms_arr))

    # Exact P6 normative formula: max(p99_silence * 4, 0.01)
    raw_thresh = p99 * 4.0
    calibrated_thresh = max(raw_thresh, 0.01)
    clamped_thresh = min(calibrated_thresh, 0.060)

    print("=" * 70)
    print("  SPECTRA KWS -- P6 VAD ACOUSTIC CALIBRATION REPORT")
    print("=" * 70)
    print(f"  Analyzed Audio:           {len(audio_data)/sr:.2f} seconds ({len(rms_arr)} frames of 100ms)")
    print(f"  Background Noise Median:  RMS = {p50:.5f}")
    print(f"  Background Noise p90:     RMS = {p90:.5f}")
    print(f"  Background Noise p99:     RMS = {p99:.5f}")
    print(f"  Background Noise Max:     RMS = {max_rms:.5f}")
    print("-" * 70)
    print(f"  Normative Formula:        max(p99_silence * 4.0, 0.010)")
    print(f"  Raw 4x Multiplier:        {raw_thresh:.5f}")
    print(f"  Calibrated VAD Threshold: {clamped_thresh:.4f}")
    print("-" * 70)
    print(f"  Config Header Update:     #define SPECTRA_VAD_THRESHOLD  {clamped_thresh:.4f}f")
    print("=" * 70)

    return {
        "p99_silence": p99,
        "calibrated_threshold": clamped_thresh
    }


def generate_demo_silence():
    # 60 seconds of realistic room tone (Gaussian noise with RMS ~ 0.0028)
    sr = 16000
    duration_s = 60
    np.random.seed(42)
    noise = np.random.normal(0, 0.0028, sr * duration_s).astype(np.float32)
    return noise, sr


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate VAD threshold from room acoustic noise")
    parser.add_argument("--audio", help="Path to room silence WAV file (16 kHz mono)")
    parser.add_argument("--demo", action="store_true", help="Run on synthetic room noise demonstration")
    args = parser.parse_args()

    if args.demo:
        data, sr = generate_demo_silence()
        calibrate_from_audio(data, sr=sr)
    elif args.audio:
        if not os.path.exists(args.audio):
            print(f"Error: Audio file not found: {args.audio}")
            sys.exit(1)
        data, sr = sf.read(args.audio)
        calibrate_from_audio(data, sr=sr)
    else:
        parser.print_help()
