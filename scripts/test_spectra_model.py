"""
Spectra KWS - Local model testing script
Tests the quantized spectra_model.tflite the same way the ESP32 will run it.

Two modes:
  python test_spectra_model.py --mode files   -> runs your existing test WAVs through the tflite model
  python test_spectra_model.py --mode live    -> live mic testing, prints predictions in real time

Requirements (install in your venv):
    pip install tflite-runtime sounddevice numpy librosa
    (if tflite-runtime fails to install on Windows, use: pip install tensorflow
     and change the import below to `import tensorflow.lite as tflite`)
"""

import argparse
import numpy as np
import librosa
import time
import queue

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

MODEL_PATH = "models/spectra_model.tflite"
SAMPLE_RATE = 16000
CLIP_DURATION = 1.0          # seconds - MATCH whatever you used in training
N_MFCC = 40                  # matches your model input shape (40, 32, 1)
N_FRAMES = 32                # matches your model input shape
CONFIDENCE_THRESHOLD = 0.5   # adjust after watching live results

def normalize_amplitude(audio):
    """
    Peak-normalize audio to [-1, 1] range.
    This MUST match the normalization used in extract_features.py during training.
    """
    max_amp = np.max(np.abs(audio))
    if max_amp > 1e-6:  # avoid division by zero for silence
        audio = audio / max_amp
    return audio


def extract_mfcc(audio, sr=SAMPLE_RATE):
    """
    IMPORTANT: This MUST exactly match the preprocessing used in extract_features.py
    during training. If these settings differ even slightly, accuracy will be wrong.
    Adjust n_fft / hop_length / n_mfcc here to match your training script exactly.
    """
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC, n_fft=1024, hop_length=512)
    # pad or truncate to fixed frame count
    if mfcc.shape[1] < N_FRAMES:
        pad_width = N_FRAMES - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode="constant")
    else:
        mfcc = mfcc[:, :N_FRAMES]
    return mfcc.astype(np.float32)


class SpectraModel:
    def __init__(self, model_path=MODEL_PATH):
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        print(f"Loaded model. Input shape: {self.input_details[0]['shape']}, "
              f"dtype: {self.input_details[0]['dtype']}")

    def predict(self, mfcc):
        x = mfcc.reshape(1, N_MFCC, N_FRAMES, 1)

        # Handle INT8 quantized input if applicable
        in_dtype = self.input_details[0]['dtype']
        if in_dtype == np.int8:
            scale, zero_point = self.input_details[0]['quantization']
            x = (x / scale + zero_point).astype(np.int8)
        else:
            x = x.astype(np.float32)

        self.interpreter.set_tensor(self.input_details[0]['index'], x)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])

        out_dtype = self.output_details[0]['dtype']
        if out_dtype == np.int8:
            scale, zero_point = self.output_details[0]['quantization']
            output = (output.astype(np.float32) - zero_point) * scale

        return output[0]  # [prob_negative, prob_positive] most likely


def run_on_files(test_dir="dataset/splits/test_raw"):
    # Note: I changed the default test_dir to 'test_raw' since the actual split 
    # folder structure with unaugmented data created by split_raw_only.py 
    # saves the files into splits/<split_name>_raw/. 
    import os
    import glob

    model = SpectraModel()
    results = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}

    for label, subdir in [("positive", "positive"), ("negative", "negative")]:
        files = glob.glob(os.path.join(test_dir, subdir, "*.wav"))
        print(f"\nTesting {len(files)} {label} files...")
        for f in files:
            audio, sr = librosa.load(f, sr=SAMPLE_RATE, duration=CLIP_DURATION)
            audio = librosa.util.fix_length(audio, size=int(SAMPLE_RATE * CLIP_DURATION))
            mfcc = extract_mfcc(audio, sr)
            probs = model.predict(mfcc)
            pred_positive = probs[1] > CONFIDENCE_THRESHOLD

            actual_positive = (label == "positive")
            if actual_positive and pred_positive:
                results["tp"] += 1
            elif actual_positive and not pred_positive:
                results["fn"] += 1
                print(f"  MISSED: {f} (confidence={probs[1]:.3f})")
            elif not actual_positive and pred_positive:
                results["fp"] += 1
                print(f"  FALSE TRIGGER: {f} (confidence={probs[1]:.3f})")
            else:
                results["tn"] += 1

    precision = results["tp"] / (results["tp"] + results["fp"] + 1e-9)
    recall = results["tp"] / (results["tp"] + results["fn"] + 1e-9)
    print(f"\n--- Results (threshold={CONFIDENCE_THRESHOLD}) ---")
    print(results)
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")


def run_live(device=None):
    import sounddevice as sd

    model = SpectraModel()
    audio_q = queue.Queue()

    def callback(indata, frames, time_info, status):
        audio_q.put(indata.copy())

    if device is not None:
        print(f"Using audio input device: {device} ({sd.query_devices(device)['name']})")
    else:
        print(f"Using default audio input device: {sd.query_devices(sd.default.device[0])['name']}")
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

                    # Normalize amplitude to match training data levels
                    clip = normalize_amplitude(clip)

                    mfcc = extract_mfcc(clip)
                    probs = model.predict(mfcc)
                    confidence = probs[1]

                    bar = "#" * int(confidence * 40)
                    marker = "  <-- TRIGGER" if confidence > CONFIDENCE_THRESHOLD else ""
                    print(f"\r[{bar:<40}] {confidence:.3f}{marker}", end="", flush=True)
                    if confidence > CONFIDENCE_THRESHOLD:
                        print()  # newline after a trigger so it's visible in history
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["files", "live"], required=True)
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--threshold", type=float, default=CONFIDENCE_THRESHOLD)
    parser.add_argument("--device", type=int, default=None,
                        help="Audio input device index (run scripts/mic_test.py to find yours)")
    args = parser.parse_args()

    MODEL_PATH = args.model
    CONFIDENCE_THRESHOLD = args.threshold

    if args.mode == "files":
        run_on_files()
    else:
        run_live(device=args.device)
