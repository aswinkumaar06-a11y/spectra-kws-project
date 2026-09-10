#!/usr/bin/env python3
"""
calibrate_vad_audio.py — Algorithm CAL-1 (Normative VAD Threshold Calibrator).

Normative algorithm per Spectra Phase 6 Specification:
INPUT: folder of silence/room-tone WAVs (16 kHz mono int16; reject or resample others — report which)
1. Frame each file: non-overlapping 25 ms windows (400 samples).
2. RMS per window = sqrt(mean(x²)), x normalized to [-1,1) by int16 scale.
3. Pool all window RMS values across silence files → p99_silence = 99th percentile.
4. SPECTRA_VAD_THRESH_RMS = max(p99_silence × 4.0, 0.010).
5. Validation: run endpoint state machine on V1/V2/V3 clips with that TH:
     START: TH exceeded for 2 consecutive frames; END: below TH for 3 consecutive frames
     (hangover); utterance minimum = 3 frames.
   Expected markers: V1 → START@2 END@8 | V2 → none | V3 → START@0 END@9.
6. OUTPUT: TH value, p99_silence, per-clip detected markers vs expected, PASS/FAIL.
"""

import os
import sys
import glob
import argparse
import numpy as np
import soundfile as sf

FRAME_SAMPLES = 400  # 25 ms @ 16 kHz (16000 * 0.025)

# Standard normative validation vectors from Phase 5 / Phase 6 specification
V1_RMS = [0.005, 0.008, 0.050, 0.090, 0.120, 0.100, 0.004, 0.003, 0.002, 0.001]
V2_RMS = [0.005, 0.060, 0.004, 0.003, 0.002, 0.001]
V3_RMS = [0.060, 0.070, 0.004, 0.005, 0.080, 0.090, 0.100, 0.003, 0.002, 0.001]

EXPECTED_MARKERS = {
    "V1": "START@2 END@8",
    "V2": "none",
    "V3": "START@0 END@9"
}


def compute_rms_frames(audio_data, frame_samples=FRAME_SAMPLES):
    """
    Computes RMS values for non-overlapping 25 ms windows (400 samples).
    Audio data must be normalized to [-1, 1).
    """
    n_frames = len(audio_data) // frame_samples
    rms_values = []
    for i in range(n_frames):
        chunk = audio_data[i * frame_samples : (i + 1) * frame_samples]
        rms = float(np.sqrt(np.mean(chunk**2)))
        rms_values.append(rms)
    return np.array(rms_values, dtype=np.float64)


def run_endpoint_state_machine(frame_rms, th):
    """
    Endpoint state machine per Algorithm CAL-1:
      - START: TH exceeded for 2 consecutive frames
      - END: below TH for 3 consecutive frames (hangover)
      - Utterance minimum: 3 frames (end - start + 1 >= 3)
    Returns: string representation of markers, e.g. 'START@2 END@8' or 'none'.
    """
    in_speech = False
    consec_above = 0
    consec_below = 0
    start_idx = None
    markers = []

    for idx, rms in enumerate(frame_rms):
        if not in_speech:
            if rms > th:
                consec_above += 1
                if consec_above == 2:
                    in_speech = True
                    start_idx = idx - 1  # 1st of the 2 consecutive frames
                    consec_below = 0
            else:
                consec_above = 0
        else:
            if rms <= th:
                consec_below += 1
                if consec_below == 3:
                    end_idx = idx
                    if (end_idx - start_idx + 1) >= 3:
                        markers.append(f"START@{start_idx} END@{end_idx}")
                    in_speech = False
                    consec_above = 0
                    consec_below = 0
                    start_idx = None
            else:
                consec_below = 0

    if not markers:
        return "none"
    return " | ".join(markers)


def validate_v1_v2_v3(th):
    """
    Validates calibrated threshold against normative V1/V2/V3 vectors.
    Expected markers:
      V1 -> START@2 END@8
      V2 -> none
      V3 -> START@0 END@9
    """
    detected_v1 = run_endpoint_state_machine(V1_RMS, th)
    detected_v2 = run_endpoint_state_machine(V2_RMS, th)
    detected_v3 = run_endpoint_state_machine(V3_RMS, th)

    v1_pass = (detected_v1 == EXPECTED_MARKERS["V1"])
    v2_pass = (detected_v2 == EXPECTED_MARKERS["V2"])
    v3_pass = (detected_v3 == EXPECTED_MARKERS["V3"])
    all_pass = v1_pass and v2_pass and v3_pass

    results = {
        "V1": {"expected": EXPECTED_MARKERS["V1"], "detected": detected_v1, "pass": v1_pass},
        "V2": {"expected": EXPECTED_MARKERS["V2"], "detected": detected_v2, "pass": v2_pass},
        "V3": {"expected": EXPECTED_MARKERS["V3"], "detected": detected_v3, "pass": v3_pass},
        "all_pass": all_pass
    }
    return results


def process_audio_file(filepath):
    """
    Loads a WAV file, verifies 16 kHz sample rate, downmixes stereo with warning,
    and returns 1D float32 audio normalized to [-1, 1).
    """
    data, sr = sf.read(filepath, dtype='float32')
    if sr != 16000:
        raise ValueError(f"Sample rate mismatch for {filepath}: expected 16000 Hz, got {sr} Hz (rejected)")

    if data.ndim > 1:
        print(f"  [CONVERT] Multichannel audio detected in {os.path.basename(filepath)} ({data.shape[1]} ch) -> downmixed to mono", file=sys.stderr)
        data = np.mean(data, axis=1)

    return data


def calibrate_from_audio(audio_data, sr=16000):
    """
    Calibrates VAD threshold from a single in-memory audio array.
    """
    if sr != 16000:
        raise ValueError(f"Sample rate mismatch: expected 16000 Hz, got {sr} Hz (rejected)")

    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)

    # Step 1 & 2: Frame in 25 ms windows and compute RMS
    rms_arr = compute_rms_frames(audio_data, frame_samples=FRAME_SAMPLES)
    if len(rms_arr) == 0:
        raise ValueError("Audio too short for 25 ms frame analysis (min 400 samples)")

    # Step 3: Compute p99_silence
    p99_silence = float(np.percentile(rms_arr, 99))

    # Step 4: SPECTRA_VAD_THRESH_RMS = max(p99_silence * 4.0, 0.010)
    calibrated_thresh = max(p99_silence * 4.0, 0.010)

    # Step 5: Validation against V1/V2/V3
    val_results = validate_v1_v2_v3(calibrated_thresh)

    return {
        "p99_silence": p99_silence,
        "calibrated_threshold": calibrated_thresh,
        "validation": val_results,
        "pass": val_results["all_pass"]
    }


def calibrate_folder(folder_path):
    """
    Calibrates VAD threshold by pooling all 25 ms RMS frames across silence WAV files in a folder.
    """
    wav_files = sorted(glob.glob(os.path.join(folder_path, "*.wav")))
    if not wav_files:
        raise FileNotFoundError(f"No WAV files found in directory: {folder_path}")

    all_rms = []
    processed_count = 0
    rejected_count = 0

    print(f"Scanning folder: {folder_path} ({len(wav_files)} WAV candidate files)")
    for wf in wav_files:
        try:
            audio = process_audio_file(wf)
            rms_frames = compute_rms_frames(audio)
            all_rms.extend(rms_frames)
            processed_count += 1
        except Exception as e:
            print(f"  [REJECTED] {os.path.basename(wf)}: {e}", file=sys.stderr)
            rejected_count += 1

    if not all_rms:
        raise ValueError(f"No valid audio frames extracted from {folder_path}")

    pooled_rms = np.array(all_rms, dtype=np.float64)
    p99_silence = float(np.percentile(pooled_rms, 99))
    calibrated_thresh = max(p99_silence * 4.0, 0.010)
    val_results = validate_v1_v2_v3(calibrated_thresh)

    return {
        "processed_files": processed_count,
        "rejected_files": rejected_count,
        "total_frames_25ms": len(pooled_rms),
        "total_audio_s": len(pooled_rms) * 0.025,
        "p99_silence": p99_silence,
        "calibrated_threshold": calibrated_thresh,
        "validation": val_results,
        "pass": val_results["all_pass"]
    }


def print_report(res):
    print("=" * 75)
    print("  SPECTRA KWS -- ALGORITHM CAL-1 VAD CALIBRATION REPORT")
    print("=" * 75)
    if "processed_files" in res:
        print(f"  Processed Files:          {res['processed_files']} valid (rejected: {res['rejected_files']})")
        print(f"  Analyzed Audio:           {res['total_audio_s']:.2f} s ({res['total_frames_25ms']} frames of 25ms)")
    print(f"  Silence p99 RMS:          {res['p99_silence']:.6f}")
    print(f"  Formula:                  max(p99_silence * 4.0, 0.010)")
    print(f"  Calibrated Threshold:     {res['calibrated_threshold']:.4f}")
    print("-" * 75)
    print("  Step 5 Validation on Normative Vectors (V1, V2, V3):")
    for name in ["V1", "V2", "V3"]:
        info = res["validation"][name]
        status = "PASS" if info["pass"] else "FAIL"
        print(f"    {name}: detected [{info['detected']}] | expected [{info['expected']}] -> {status}")
    print("-" * 75)
    verdict = "PASS" if res["pass"] else "FAIL (markers mismatch)"
    print(f"  VERDICT:                  {verdict}")
    print(f"  Kconfig Line:             CONFIG_SPECTRA_VAD_THRESH_RMS={res['calibrated_threshold']:.4f}")
    print("=" * 75)


def generate_demo_silence():
    """Generates 30s of synthetic room silence with RMS ~0.0028."""
    sr = 16000
    duration_s = 30
    np.random.seed(42)
    noise = np.random.normal(0, 0.0028, sr * duration_s).astype(np.float32)
    return noise, sr


def main():
    parser = argparse.ArgumentParser(description="Algorithm CAL-1 VAD Acoustic Calibration Tool")
    parser.add_argument("--folder", help="Directory containing room silence/tone WAV files")
    parser.add_argument("--audio", help="Path to single room silence WAV file")
    parser.add_argument("--demo", action="store_true", help="Run on synthetic room noise demonstration")
    args = parser.parse_args()

    if args.demo:
        data, sr = generate_demo_silence()
        res = calibrate_from_audio(data, sr=sr)
        print_report(res)
    elif args.folder:
        if not os.path.isdir(args.folder):
            print(f"Error: Folder not found: {args.folder}", file=sys.stderr)
            sys.exit(1)
        res = calibrate_folder(args.folder)
        print_report(res)
    elif args.audio:
        if not os.path.exists(args.audio):
            print(f"Error: Audio file not found: {args.audio}", file=sys.stderr)
            sys.exit(1)
        data = process_audio_file(args.audio)
        res = calibrate_from_audio(data, sr=16000)
        print_report(res)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
