"""
Tests if the model fails because of:
  A) Audio level mismatch (mic is louder than training data)
  B) INT8 quantization loss
  C) The model truly can't generalize to new voices/mics
"""
import numpy as np
import librosa
import sounddevice as sd
import os

SR = 16000
DEVICE = 4

print("=" * 60)
print("  Model Generalization Diagnostic")
print("=" * 60)

# --- Record ---
print(f"\nRecording 3s... SAY 'SPECTRA' NOW!")
audio = sd.rec(int(SR * 3), samplerate=SR, channels=1, dtype='float32',
               device=DEVICE, blocking=True).flatten()
print(f"  Recorded. Max amplitude: {np.max(np.abs(audio)):.4f}")

# --- Load a known-good training file for comparison ---
train_file = None
for root, dirs, files in os.walk("dataset/splits/train/positive"):
    for f in files:
        if f.endswith(".wav"):
            train_file = os.path.join(root, f)
            break
    break

train_audio, _ = librosa.load(train_file, sr=SR, duration=1.0)
train_audio = librosa.util.fix_length(train_audio, size=SR)
print(f"  Training file max amplitude: {np.max(np.abs(train_audio)):.4f}")

def extract_mfcc(audio_clip):
    mfcc = librosa.feature.mfcc(y=audio_clip, sr=SR, n_mfcc=40, n_fft=1024, hop_length=512)
    if mfcc.shape[1] < 32:
        mfcc = np.pad(mfcc, ((0, 0), (0, 32 - mfcc.shape[1])), mode="constant")
    else:
        mfcc = mfcc[:, :32]
    return mfcc.astype(np.float32)

# --- Prepare different versions of mic audio ---
mic_clip = audio[:SR]  # first 1 second

# Version 1: Raw mic audio
mic_raw = mic_clip.copy()

# Version 2: Normalized to match training file amplitude
train_max = np.max(np.abs(train_audio))
mic_max = np.max(np.abs(mic_clip))
if mic_max > 0:
    mic_normalized = mic_clip * (train_max / mic_max)
else:
    mic_normalized = mic_clip

# Version 3: Peak normalized to 1.0
if mic_max > 0:
    mic_peak_norm = mic_clip / mic_max
else:
    mic_peak_norm = mic_clip

print(f"\n  Mic raw max:        {np.max(np.abs(mic_raw)):.4f}")
print(f"  Mic normalized max: {np.max(np.abs(mic_normalized)):.4f} (matched to training)")
print(f"  Mic peak-norm max:  {np.max(np.abs(mic_peak_norm)):.4f}")

# --- Test with TFLite (quantized) model ---
try:
    import tensorflow.lite as tflite
except ImportError:
    import tflite_runtime.interpreter as tflite

def run_tflite(audio_clip, label):
    interpreter = tflite.Interpreter(model_path="models/spectra_model.tflite")
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()
    out = interpreter.get_output_details()

    mfcc = extract_mfcc(audio_clip)
    x = mfcc.reshape(1, 40, 32, 1)

    if inp[0]['dtype'] == np.int8:
        scale, zp = inp[0]['quantization']
        x = (x / scale + zp).astype(np.int8)

    interpreter.set_tensor(inp[0]['index'], x)
    interpreter.invoke()
    output = interpreter.get_tensor(out[0]['index'])

    if out[0]['dtype'] == np.int8:
        scale, zp = out[0]['quantization']
        output = (output.astype(np.float32) - zp) * scale

    probs = output[0]
    detected = "YES" if probs[1] > 0.5 else "no"
    print(f"  {label:35s} -> neg={probs[0]:.4f}  pos={probs[1]:.4f}  [{detected}]")
    return probs

# --- Test with Keras (float) model ---
import tensorflow as tf

def run_keras(audio_clip, label):
    model = tf.keras.models.load_model("models/spectra_model.h5")
    mfcc = extract_mfcc(audio_clip)
    x = mfcc.reshape(1, 40, 32, 1)
    probs = model.predict(x, verbose=0)[0]
    detected = "YES" if probs[1] > 0.5 else "no"
    print(f"  {label:35s} -> neg={probs[0]:.4f}  pos={probs[1]:.4f}  [{detected}]")
    return probs

print("\n" + "=" * 60)
print("  TFLite (INT8 quantized) Model Results")
print("=" * 60)
run_tflite(train_audio,    "Training WAV (reference)")
run_tflite(mic_raw,        "Mic: raw audio")
run_tflite(mic_normalized, "Mic: amplitude-matched")
run_tflite(mic_peak_norm,  "Mic: peak-normalized")

print("\n" + "=" * 60)
print("  Keras (float32) Model Results")
print("=" * 60)
run_keras(train_audio,    "Training WAV (reference)")
run_keras(mic_raw,        "Mic: raw audio")
run_keras(mic_normalized, "Mic: amplitude-matched")
run_keras(mic_peak_norm,  "Mic: peak-normalized")

print("\n" + "=" * 60)
print("  MFCC Comparison (mean values)")
print("=" * 60)
mfcc_train = extract_mfcc(train_audio)
mfcc_mic = extract_mfcc(mic_raw)
mfcc_norm = extract_mfcc(mic_normalized)
print(f"  Training WAV MFCC mean: {np.mean(mfcc_train):.2f}, std: {np.std(mfcc_train):.2f}")
print(f"  Mic raw MFCC mean:      {np.mean(mfcc_mic):.2f}, std: {np.std(mfcc_mic):.2f}")
print(f"  Mic normalized MFCC:    {np.mean(mfcc_norm):.2f}, std: {np.std(mfcc_norm):.2f}")
print("=" * 60)
