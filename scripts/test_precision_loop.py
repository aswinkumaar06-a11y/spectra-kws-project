#!/usr/bin/env python3
"""
test_precision_loop.py — Locked-Test Remediation of the Precision Loop.

Methodology (Strict Avoidance of Selection Bias):
  1. Partitions the 396 test samples into Validation (Tune) and Locked Test splits
     using speaker-disjoint stratification (14 positive speakers, 16 negative speakers).
  2. Runs an iterative threshold sweep ONLY on the Validation (Tune) set.
  3. Selects the optimal threshold on Validation where precision >= 99.0%.
  4. Evaluates the selected threshold ONCE on the Locked Test set.
  5. Computes exact Wilson 95% Score Confidence Intervals for all metrics.
  6. Reports metrics using rigorous count + CI phrasing (no bare "100%" claims).
"""

import os
import sys
import json
import argparse
import numpy as np

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "spectra_model.tflite")
FEATURES_DIR = os.path.join(PROJECT_ROOT, "dataset", "features")
MANIFEST_PATH = os.path.join(PROJECT_ROOT, "dataset", "splits", "test_manifest.json")


def compute_wilson_ci(k, n, confidence=0.95):
    """
    Computes exact Wilson score confidence interval for a binomial proportion k/n.
    Returns: (point_estimate, lower_bound, upper_bound)
    """
    if n == 0:
        return 0.0, 0.0, 1.0
    z = 1.95996  # 95% confidence standard normal quantile
    p_hat = k / n
    denominator = 1.0 + (z**2) / n
    center = (p_hat + (z**2) / (2.0 * n)) / denominator
    spread = (z / denominator) * np.sqrt((p_hat * (1.0 - p_hat) / n) + (z**2) / (4.0 * (n**2)))
    lower = max(0.0, float(center - spread))
    upper = min(1.0, float(center + spread))
    return float(p_hat), lower, upper


def quantize_input(sample, scale, zero_point):
    quant = np.round(sample / scale) + zero_point
    return np.clip(quant, -128, 127).astype(np.int8)


def dequantize_output(raw_output, scale, zero_point):
    return (raw_output.astype(np.float32) - zero_point) * scale


def get_speaker_disjoint_split(manifest):
    """
    Partitions indices of manifest into Validation (Tune) and Locked Test sets,
    guaranteeing strict human and synthetic speaker disjointness.
    """
    val_pos_spks = {
        'SPK_POS_E64D2D87', 'SPK_POS_75CCF416', 'SPK_POS_F1DEBFDF',
        'SPK_POS_EE49163B', 'SPK_POS_7B790D3E', 'SPK_POS_4E760388', 'SPK_POS_FBBF42C6'
    }
    val_neg_spks = {
        'SPK_GSC_0ba018fc', 'SPK_GSC_1e9e6bdd', 'VOICE_SYNTH_DAVID',
        'SPK_GSC_28ef2a01', 'SPK_GSC_0d393936', 'SPK_GSC_5af0ca83', 'SPK_GSC_e7117d00'
    }

    val_indices = []
    test_indices = []

    for idx, item in enumerate(manifest):
        spk = item.get('anonymous_speaker_id')
        lbl = item.get('label')

        if lbl == 'positive':
            if spk in val_pos_spks:
                val_indices.append(idx)
            else:
                test_indices.append(idx)
        else:
            if spk in val_neg_spks:
                val_indices.append(idx)
            else:
                test_indices.append(idx)

    return np.array(val_indices), np.array(test_indices)


def run_locked_precision_test(model_path=DEFAULT_MODEL_PATH, target_precision=0.99):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.exists(MANIFEST_PATH):
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    X_test_all = np.load(os.path.join(FEATURES_DIR, "X_test.npy"))
    y_test_all = np.load(os.path.join(FEATURES_DIR, "y_test.npy"))

    val_idx, test_idx = get_speaker_disjoint_split(manifest)
    print("=" * 76)
    print("  SPECTRA KWS -- LOCKED-TEST REMEDIATION (A3)")
    print("=" * 76)
    print(f"  Model:                {model_path}")
    print(f"  Total Test Pool:      {len(X_test_all)} samples")
    print(f"  Validation (Tune):    {len(val_idx)} samples (Pos: {np.sum(y_test_all[val_idx]==1)}, Neg: {np.sum(y_test_all[val_idx]==0)})")
    print(f"  Locked Test (Held):   {len(test_idx)} samples (Pos: {np.sum(y_test_all[test_idx]==1)}, Neg: {np.sum(y_test_all[test_idx]==0)})")
    print(f"  Disjoint Speakers:    YES (Zero speaker leakage between Tune and Locked Test)")
    print("-" * 76)

    # 1. Run inference across all samples
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    in_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]
    in_scale, in_zp = in_det["quantization"]
    out_scale, out_zp = out_det["quantization"]

    all_probs = np.zeros(len(X_test_all), dtype=np.float32)
    for i in range(len(X_test_all)):
        q_in = quantize_input(X_test_all[i:i+1], in_scale, in_zp)
        interpreter.set_tensor(in_det["index"], q_in)
        interpreter.invoke()
        raw_out = interpreter.get_tensor(out_det["index"])
        all_probs[i] = float(dequantize_output(raw_out, out_scale, out_zp)[0, 1])

    val_probs = all_probs[val_idx]
    val_y = y_test_all[val_idx]
    test_probs = all_probs[test_idx]
    test_y = y_test_all[test_idx]

    # 2. Threshold Calibration Loop (Validation set ONLY)
    print("\n--- [STEP 1] Threshold Calibration Curve (Validation / Tune Set Only) ---")
    print(f"  {'Threshold':<12} {'Val TP':<8} {'Val FP':<8} {'Val FN':<8} {'Val TN':<8} {'Precision':<12} {'Recall':<10} {'Status'}")
    print("  " + "-" * 72)

    selected_threshold = None
    curve_data = []

    for th in np.arange(0.40, 0.96, 0.05):
        v_pred = (val_probs > th).astype(int)
        tp = int(np.sum((v_pred == 1) & (val_y == 1)))
        fp = int(np.sum((v_pred == 1) & (val_y == 0)))
        fn = int(np.sum((v_pred == 0) & (val_y == 1)))
        tn = int(np.sum((v_pred == 0) & (val_y == 0)))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        curve_data.append((th, tp, fp, fn, tn, prec, rec))

        status = "Eligible (>=99%)" if prec >= target_precision else "Seeking..."
        print(f"  {th:<12.2f} {tp:<8} {fp:<8} {fn:<8} {tn:<8} {prec*100:<11.2f}% {rec*100:<9.2f}% {status}")

        if prec >= target_precision and selected_threshold is None:
            selected_threshold = float(th)

    if selected_threshold is None:
        selected_threshold = 0.50

    print(f"\n  Chosen Operating Threshold: {selected_threshold:.2f} (locked on validation curve)")
    print("-" * 76)

    # 3. Single-Pass Evaluation on Locked Test Set
    print("\n--- [STEP 2] Single-Pass Evaluation on Locked Test Set ---")
    t_pred = (test_probs > selected_threshold).astype(int)
    test_tp = int(np.sum((t_pred == 1) & (test_y == 1)))
    test_fp = int(np.sum((t_pred == 1) & (test_y == 0)))
    test_fn = int(np.sum((t_pred == 0) & (test_y == 1)))
    test_tn = int(np.sum((t_pred == 0) & (test_y == 0)))

    n_pos = test_tp + test_fn
    n_neg = test_tn + test_fp
    total_test = len(test_idx)

    # Calculate Wilson 95% CIs
    _, prec_lo, prec_hi = compute_wilson_ci(test_tp, test_tp + test_fp)
    _, rec_lo, rec_hi = compute_wilson_ci(test_tp, n_pos)
    _, acc_lo, acc_hi = compute_wilson_ci(test_tp + test_tn, total_test)
    _, fpr_lo, fpr_hi = compute_wilson_ci(test_fp, n_neg)

    prec_obs = test_tp / (test_tp + test_fp) if (test_tp + test_fp) > 0 else 1.0
    rec_obs = test_tp / n_pos if n_pos > 0 else 0.0
    acc_obs = (test_tp + test_tn) / total_test
    fpr_obs = test_fp / n_neg if n_neg > 0 else 0.0

    print(f"  Operating Threshold:      {selected_threshold:.2f} (selected from validation)")
    print(f"  Test Sample Size:         n = {total_test} ({n_pos} positive, {n_neg} negative)")
    print(f"  Confusion Matrix:         TP = {test_tp:3d} | FP = {test_fp:3d}")
    print(f"                            FN = {test_fn:3d} | TN = {test_tn:3d}")
    print("\n  Rigorous Empirical Metrics with Wilson 95% Confidence Intervals:")
    print(f"  * Precision:     {test_tp}/{test_tp + test_fp} observed = {prec_obs*100:.2f}% (95% CI: [{prec_lo*100:.2f}%, {prec_hi*100:.2f}%])")
    print(f"  * Recall:        {test_tp}/{n_pos} observed = {rec_obs*100:.2f}% (95% CI: [{rec_lo*100:.2f}%, {rec_hi*100:.2f}%], lower bound >= {rec_lo*100:.1f}%)")
    print(f"  * False Pos Rate: {test_fp}/{n_neg} observed = {fpr_obs*100:.2f}% (95% CI: [{fpr_lo*100:.2f}%, {fpr_hi*100:.2f}%], upper bound <= {fpr_hi*100:.1f}%)")
    print(f"  * Accuracy:      {test_tp + test_tn}/{total_test} observed = {acc_obs*100:.2f}% (95% CI: [{acc_lo*100:.2f}%, {acc_hi*100:.2f}%])")

    print("\n" + "=" * 76)
    print("  NORMATIVE CLAIMS STATEMENT (Global Rule 3 Compliant):")
    print(f"  \"{test_fp} FP observed on n={n_neg} (95% CI upper ~ {fpr_hi*100:.1f}% FP rate)\"")
    print(f"  \"recall {test_tp}/{n_pos} observed (95% CI lower ~ {rec_lo*100:.1f}%)\"")
    print("=" * 76)

    return True


if __name__ == "__main__":
    run_locked_precision_test()
