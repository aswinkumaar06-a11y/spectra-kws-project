"""
Records a short clip and analyzes the audio to check for DC offset,
noise, and whether speech is actually being captured.
Also tests a known-good positive WAV through the model for comparison.
"""
import numpy as np
import sounddevice as sd
import os, glob, librosa, wave

DEVICE = 4
SR = 16000
DURATION = 3  # seconds

print("=" * 60)
print("  Audio Capture Diagnostic")
print("=" * 60)

# --- Record from mic ---
print(f"\nRecording {DURATION}s from device {DEVICE}... SAY 'SPECTRA' NOW!")
audio = sd.rec(int(SR * DURATION), samplerate=SR, channels=1, dtype='float32',
               device=DEVICE, blocking=True)
audio = audio.flatten()

print(f"\n  Raw audio stats:")
print(f"    Length:    {len(audio)} samples ({len(audio)/SR:.1f}s)")
print(f"    Mean:     {np.mean(audio):.6f}  (should be ~0.0, nonzero = DC offset)")
print(f"    Std dev:  {np.std(audio):.6f}  (should vary with speech)")
print(f"    Min:      {np.min(audio):.6f}")
print(f"    Max:      {np.max(audio):.6f}")
print(f"    Max |amp|:{np.max(np.abs(audio)):.6f}")

# Check for DC offset
dc_offset = np.mean(audio)
if abs(dc_offset) > 0.01:
    print(f"\n  WARNING: Large DC offset detected ({dc_offset:.4f})!")
    print(f"  Removing DC offset...")
    audio_clean = audio - dc_offset
    print(f"  After DC removal:")
    print(f"    Mean:     {np.mean(audio_clean):.6f}")
    print(f"    Max |amp|:{np.max(np.abs(audio_clean)):.6f}")
else:
    audio_clean = audio

# Check if audio is all the same value (stuck)
unique_vals = len(np.unique(np.round(audio, 4)))
print(f"\n  Unique amplitude values: {unique_vals}")
if unique_vals < 100:
    print("  WARNING: Very few unique values - mic may be stuck/not working!")

# Save recording for inspection
out_path = "dataset/mic_test_recording.wav"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
import scipy.io.wavfile as wav_io
wav_io.write(out_path, SR, (audio * 32767).astype(np.int16))
print(f"\n  Saved recording to: {out_path}")
print(f"  (Open this file in an audio player to hear what the mic captured)")

# --- Run mic recording through the model ---
print("\n" + "=" * 60)
print("  Running YOUR mic recording through the model")
print("=" * 60)

try:
    import tensorflow.lite as tflite
except ImportError:
    import tflite_runtime.interpreter as tflite

interpreter = tflite.Interpreter(model_path="models/spectra_model.tflite")
interpreter.allocate_tensors()
inp = interpreter.get_input_details()
out = interpreter.get_output_details()

# Take the first 1 second of mic audio
mic_clip = audio[:SR]
mic_clip = librosa.util.fix_length(mic_clip, size=SR)

mfcc = librosa.feature.mfcc(y=mic_clip, sr=SR, n_mfcc=40, n_fft=1024, hop_length=512)
if mfcc.shape[1] < 32:
    mfcc = np.pad(mfcc, ((0, 0), (0, 32 - mfcc.shape[1])), mode="constant")
else:
    mfcc = mfcc[:, :32]
mfcc = mfcc.astype(np.float32)

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
print(f"\n  Your mic audio (first 1s) -> neg={probs[0]:.4f}  pos={probs[1]:.4f}")
if probs[1] > 0.5:
    print(f"  DETECTED 'SPECTRA' from your mic!")
else:
    print(f"  Did NOT detect 'SPECTRA' from your mic (pos={probs[1]:.4f})")

# --- Now test a known-good positive WAV file through the model ---
print("\n" + "=" * 60)
print("  Model Sanity Check - Testing a known positive WAV file")
print("=" * 60)

try:
    import tensorflow.lite as tflite
except ImportError:
    import tflite_runtime.interpreter as tflite

pos_dir = "dataset/splits/train/positive"
if not os.path.exists(pos_dir):
    pos_dir = "dataset/splits/test_raw/positive"
pos_files = glob.glob(os.path.join(pos_dir, "*.wav"))

if pos_files:
    test_file = pos_files[0]
    print(f"\n  Testing: {test_file}")
    
    file_audio, _ = librosa.load(test_file, sr=SR, duration=1.0)
    file_audio = librosa.util.fix_length(file_audio, size=SR)
    
    print(f"  File audio - Mean: {np.mean(file_audio):.6f}, Max: {np.max(np.abs(file_audio)):.6f}")
    
    # Extract MFCC
    mfcc = librosa.feature.mfcc(y=file_audio, sr=SR, n_mfcc=40, n_fft=1024, hop_length=512)
    if mfcc.shape[1] < 32:
        mfcc = np.pad(mfcc, ((0, 0), (0, 32 - mfcc.shape[1])), mode="constant")
    else:
        mfcc = mfcc[:, :32]
    mfcc = mfcc.astype(np.float32)
    
    # Run through model
    interpreter = tflite.Interpreter(model_path="models/spectra_model.tflite")
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()
    out = interpreter.get_output_details()
    
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
    print(f"  Model output: neg={probs[0]:.4f}  pos={probs[1]:.4f}")
    if probs[1] > 0.5:
        print(f"  Model correctly identifies this as SPECTRA!")
    else:
        print(f"  WARNING: Model fails even on a training file!")
else:
    print("  No positive WAV files found to test.")

print("\n" + "=" * 60)
