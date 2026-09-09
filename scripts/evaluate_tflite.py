"""
evaluate_tflite.py — STEP 2. Standalone quantitative evaluation of INT8 TFLite model.
Evaluates the exported INT8 .tflite model across the entire held-out test split (X_test, y_test).
Strictly verifies and applies scale and zero-point quantization/dequantization with rounding and clipping.
"""
import os
import sys
import argparse
import numpy as np

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

from audio_utils import quantize_to_int8, dequantize_from_int8

DEFAULT_MODEL_PATH = "models/spectra_model.tflite"
FEATURES_DIR = "dataset/features"

def evaluate_model(model_path=DEFAULT_MODEL_PATH, features_dir=FEATURES_DIR, threshold=0.5):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    x_test_path = os.path.join(features_dir, "X_test.npy")
    y_test_path = os.path.join(features_dir, "y_test.npy")
    if not os.path.exists(x_test_path) or not os.path.exists(y_test_path):
        raise FileNotFoundError(f"Test feature files not found in: {features_dir}")

    X_test = np.load(x_test_path)
    y_test = np.load(y_test_path)

    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    in_shape = input_details["shape"]
    in_dtype = input_details["dtype"]
    in_scale, in_zero_point = input_details["quantization"]

    out_shape = output_details["shape"]
    out_dtype = output_details["dtype"]
    out_scale, out_zero_point = output_details["quantization"]

    print("=" * 60)
    print("  Spectra KWS - Quantitative TFLite INT8 Evaluation")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"  Input Tensor:  shape={in_shape}, dtype={in_dtype}, scale={in_scale:.6f}, zero_point={in_zero_point}")
    print(f"  Output Tensor: shape={out_shape}, dtype={out_dtype}, scale={out_scale:.6f}, zero_point={out_zero_point}")
    print(f"Test Set Size: {len(X_test)} samples (Class 0: {np.sum(y_test == 0)}, Class 1: {np.sum(y_test == 1)})")
    print(f"Decision Threshold: {threshold}")
    print("-" * 60)

    tp = 0
    fp = 0
    fn = 0
    tn = 0
    probabilities = []
    manifest_path = os.path.join(features_dir, "test_manifest.json")
    meta = None
    if os.path.exists(manifest_path):
        import json
        with open(manifest_path, "r") as mf:
            meta = json.load(mf)

    hard_total = 0
    hard_fp = 0
    hard_tn = 0

    for i in range(len(X_test)):
        sample = X_test[i:i+1] # shape (1, 40, 32, 1)

        # Explicitly verify tensor metadata and quantize
        if in_dtype == np.int8:
            quantized_input = quantize_to_int8(sample, in_scale, in_zero_point)
        else:
            quantized_input = sample.astype(np.float32)

        interpreter.set_tensor(input_details["index"], quantized_input)
        interpreter.invoke()
        raw_output = interpreter.get_tensor(output_details["index"])

        if out_dtype == np.int8:
            dequantized_output = dequantize_from_int8(raw_output, out_scale, out_zero_point)
        else:
            dequantized_output = raw_output.astype(np.float32)

        pos_prob = float(dequantized_output[0, 1])
        probabilities.append(pos_prob)
        actual = int(y_test[i])
        predicted = 1 if pos_prob > threshold else 0

        if actual == 1 and predicted == 1:
            tp += 1
        elif actual == 1 and predicted == 0:
            fn += 1
        elif actual == 0 and predicted == 1:
            fp += 1
        else:
            tn += 1

        if meta and i < len(meta):
            if meta[i].get("is_hard_negative", False):
                hard_total += 1
                if predicted == 1:
                    hard_fp += 1
                else:
                    hard_tn += 1

    total = len(X_test)
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    specificity = tn / (tn + fp + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    print("\n--- Quantitative Evaluation Results ---")
    print(f"Accuracy:    {accuracy:.4f} ({tp + tn}/{total})")
    print(f"Precision:   {precision:.4f}")
    print(f"Recall:      {recall:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"F1-Score:    {f1:.4f}")
    print(f"Confusion Matrix:")
    print(f"  True Positives  (TP): {tp:4d}  | False Positives (FP): {fp:4d}")
    print(f"  False Negatives (FN): {fn:4d}  | True Negatives  (TN): {tn:4d}")

    if hard_total > 0:
        hard_far = hard_fp / hard_total
        print("\n--- Hard Negatives Breakdown (Phonetically Adjacent Words) ---")
        print(f"  Total Hard Negatives in Test: {hard_total}")
        print(f"  False Triggers (FP):          {hard_fp}")
        print(f"  Correct Rejections (TN):       {hard_tn}")
        print(f"  False Activation Rate (FAR):  {hard_far*100:.2f}%")
        print(f"  True Rejection Rate:          {(1.0 - hard_far)*100:.2f}%")
    print("=" * 60)

    return {
        "model": model_path,
        "total_samples": total,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1_score": float(f1),
        "quantization_info": {
            "input_scale": float(in_scale),
            "input_zero_point": int(in_zero_point),
            "output_scale": float(out_scale),
            "output_zero_point": int(out_zero_point)
        }
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate INT8 TFLite model on full test set")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Path to .tflite model")
    parser.add_argument("--features-dir", default=FEATURES_DIR, help="Path to features directory")
    parser.add_argument("--threshold", type=float, default=0.5, help="Positive classification threshold")
    args = parser.parse_args()

    evaluate_model(model_path=args.model, features_dir=args.features_dir, threshold=args.threshold)
