"""
Spectra KWS - Local model testing script
Tests the quantized spectra_model.tflite the same way the ESP32 will run it.

Two modes:
  python scripts/test_spectra_model.py --mode files [--model models/spectra_model.tflite]
  python scripts/test_spectra_model.py --mode live  [--device 4] [--model models/spectra_model.tflite]
"""

import os
import glob
import time
import queue
import argparse
import numpy as np

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

from audio_utils import (
    preprocess_audio_clip,
    normalize_amplitude,
    check_energy_gate,
    extract_mfcc,
    quantize_to_int8,
    dequantize_from_int8,
    SAMPLE_RATE,
    CLIP_DURATION,
    N_MFCC,
    N_FRAMES,
    ENERGY_GATE_THRESHOLD
)

DEFAULT_MODEL_PATH = "models/spectra_model.tflite"
CONFIDENCE_THRESHOLD = 0.5

class SpectraModel:
    def __init__(self, model_path=DEFAULT_MODEL_PATH):
        self.model_path = model_path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        print(f"Loaded model: {model_path}")
        print(f"  Input shape: {self.input_details[0]['shape']}, dtype: {self.input_details[0]['dtype']}")
        print(f"  Input quant: {self.input_details[0]['quantization']}")
        print(f"  Output shape: {self.output_details[0]['shape']}, dtype: {self.output_details[0]['dtype']}")
        print(f"  Output quant: {self.output_details[0]['quantization']}")

    def predict(self, mfcc):
        # Shape: (1, 40, 32, 1)
        x = mfcc.reshape(1, N_MFCC, N_FRAMES, 1)

        # Handle INT8 quantized input with proper rounding & clipping
        in_dtype = self.input_details[0]['dtype']
        if in_dtype == np.int8:
            scale, zero_point = self.input_details[0]['quantization']
            x = quantize_to_int8(x, scale, zero_point)
        else:
            x = x.astype(np.float32)

        self.interpreter.set_tensor(self.input_details[0]['index'], x)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])

        out_dtype = self.output_details[0]['dtype']
        if out_dtype == np.int8:
            scale, zero_point = self.output_details[0]['quantization']
            output = dequantize_from_int8(output, scale, zero_point)

        return output[0]  # [prob_negative, prob_positive]

def run_on_files(test_dir="dataset/splits/test_raw", model_path=DEFAULT_MODEL_PATH, threshold=CONFIDENCE_THRESHOLD):
    model = SpectraModel(model_path=model_path)
    results = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}

    for label, subdir in [("positive", "positive"), ("negative", "negative")]:
        files = glob.glob(os.path.join(test_dir, subdir, "*.wav"))
        print(f"\nTesting {len(files)} {label} files from {os.path.join(test_dir, subdir)}...")
        for f in files:
            # Shared preprocessing guaranteed bit-for-bit identical to training
            mfcc = preprocess_audio_clip(f, normalize=True)
            probs = model.predict(mfcc)
            pred_positive = bool(probs[1] > threshold)
            actual_positive = (label == "positive")

            if actual_positive and pred_positive:
                results["tp"] += 1
            elif actual_positive and not pred_positive:
                results["fn"] += 1
                print(f"  MISSED: {os.path.basename(f)} (conf={probs[1]:.3f})")
            elif not actual_positive and pred_positive:
                results["fp"] += 1
                print(f"  FALSE TRIGGER: {os.path.basename(f)} (conf={probs[1]:.3f})")
            else:
                results["tn"] += 1

    precision = results["tp"] / (results["tp"] + results["fp"] + 1e-9)
    recall = results["tp"] / (results["tp"] + results["fn"] + 1e-9)
    total = sum(results.values())
    accuracy = (results["tp"] + results["tn"]) / (total + 1e-9)
    print(f"\n--- File Test Results (threshold={threshold}) ---")
    print(results)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    return results

def run_live(device=None, model_path=DEFAULT_MODEL_PATH, threshold=CONFIDENCE_THRESHOLD):
    import sounddevice as sd

    model = SpectraModel(model_path=model_path)
    # Bounded queue (maxsize=10) to prevent unbounded memory growth
    audio_q = queue.Queue(maxsize=10)

    def callback(indata, frames, time_info, status):
        if status:
            print(f"\nAudio callback status: {status}")
        try:
            audio_q.put_nowait(indata.copy())
        except queue.Full:
            print("\nWARNING: Audio buffer overflow — inference is lagging!")

    device_name = sd.query_devices(device)['name'] if device is not None else sd.query_devices(sd.default.device[0])['name']
    print(f"Using audio input device: {device} ({device_name})")
    print(f"Energy Gate Threshold: {ENERGY_GATE_THRESHOLD}")
    print("Listening... say 'SPECTRA' (Ctrl+C to stop)")

    block_size = int(SAMPLE_RATE * CLIP_DURATION)

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=callback,
                         blocksize=block_size, dtype="float32", device=device):
        buffer = np.zeros(0, dtype=np.float32)
        try:
            while True:
                chunk = audio_q.get()
                buffer = np.concatenate([buffer, chunk.flatten()])
                if len(buffer) >= block_size:
                    clip = buffer[:block_size]
                    buffer = buffer[block_size // 2:]  # 50% overlap sliding window

                    # Energy gate: check if clip contains sufficient energy
                    has_energy, max_amp = check_energy_gate(clip, threshold=ENERGY_GATE_THRESHOLD)
                    if not has_energy:
                        q_depth = audio_q.qsize()
                        print(f"\r[Silence... amp={max_amp:.3f} | q_depth={q_depth}]               ", end="", flush=True)
                        continue

                    # Preprocess with peak normalization & MFCC extraction
                    t_start = time.perf_counter()
                    norm_clip = normalize_amplitude(clip)
                    mfcc = extract_mfcc(norm_clip)
                    probs = model.predict(mfcc)
                    inference_ms = (time.perf_counter() - t_start) * 1000.0

                    confidence = probs[1]
                    q_depth = audio_q.qsize()
                    bar = "#" * int(confidence * 40)
                    marker = "  <-- TRIGGER" if confidence > threshold else ""
                    print(f"\r[{bar:<40}] {confidence:.3f} (amp={max_amp:.3f} | inf={inference_ms:.1f}ms | q={q_depth}){marker}", end="", flush=True)
                    if confidence > threshold:
                        print()
        except KeyboardInterrupt:
            print("\nStopped.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spectra KWS Inference and Evaluation")
    parser.add_argument("--mode", choices=["files", "live"], required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Path to .tflite model file")
    parser.add_argument("--threshold", type=float, default=CONFIDENCE_THRESHOLD, help="Confidence trigger threshold")
    parser.add_argument("--device", type=int, default=None, help="Audio input device index")
    parser.add_argument("--test-dir", default="dataset/splits/test_raw", help="Directory containing test files")
    args = parser.parse_args()

    if args.mode == "files":
        run_on_files(test_dir=args.test_dir, model_path=args.model, threshold=args.threshold)
    else:
        run_live(device=args.device, model_path=args.model, threshold=args.threshold)
