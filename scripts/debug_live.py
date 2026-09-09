"""
Debug version of live test - shows both audio level and model confidence
so we can see if the mic is capturing audio properly.
"""
import numpy as np
import librosa
import queue

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

import sounddevice as sd

MODEL_PATH = "models/spectra_model.tflite"
SAMPLE_RATE = 16000
CLIP_DURATION = 1.0
N_MFCC = 40
N_FRAMES = 32
CONFIDENCE_THRESHOLD = 0.5

def extract_mfcc(audio, sr=SAMPLE_RATE):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC, n_fft=1024, hop_length=512)
    if mfcc.shape[1] < N_FRAMES:
        pad_width = N_FRAMES - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode="constant")
    else:
        mfcc = mfcc[:, :N_FRAMES]
    return mfcc.astype(np.float32)

# Load model
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(f"Model input: shape={input_details[0]['shape']}, dtype={input_details[0]['dtype']}")

audio_q = queue.Queue()

def callback(indata, frames, time_info, status):
    if status:
        print(f"  [Audio status: {status}]")
    audio_q.put(indata.copy())

DEVICE = 4  # Primary Sound Capture Driver - the one that worked

print(f"\nUsing device {DEVICE}: {sd.query_devices(DEVICE)['name']}")
print("Listening... say 'SPECTRA' loudly! (Ctrl+C to stop)\n")

block_size = int(SAMPLE_RATE * CLIP_DURATION)

with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=callback,
                     blocksize=block_size, dtype="float32", device=DEVICE):
    buffer = np.zeros(0, dtype=np.float32)
    try:
        while True:
            chunk = audio_q.get()
            buffer = np.concatenate([buffer, chunk.flatten()])
            if len(buffer) >= block_size:
                clip = buffer[:block_size]
                buffer = buffer[block_size // 2:]  # 50% overlap

                # Show raw audio stats
                max_amp = np.max(np.abs(clip))
                rms = np.sqrt(np.mean(clip**2))

                # Extract MFCC and predict
                mfcc = extract_mfcc(clip)

                x = mfcc.reshape(1, N_MFCC, N_FRAMES, 1)
                in_dtype = input_details[0]['dtype']
                if in_dtype == np.int8:
                    scale, zero_point = input_details[0]['quantization']
                    x = (x / scale + zero_point).astype(np.int8)
                else:
                    x = x.astype(np.float32)

                interpreter.set_tensor(input_details[0]['index'], x)
                interpreter.invoke()
                output = interpreter.get_tensor(output_details[0]['index'])

                out_dtype = output_details[0]['dtype']
                if out_dtype == np.int8:
                    scale, zero_point = output_details[0]['quantization']
                    output = (output.astype(np.float32) - zero_point) * scale

                probs = output[0]
                confidence = probs[1] if len(probs) > 1 else probs[0]

                # Visual bar for audio level
                mic_bar = "|" * min(int(max_amp * 100), 30)
                trigger = " <<< SPECTRA!" if confidence > CONFIDENCE_THRESHOLD else ""

                print(f"  Mic: {max_amp:.4f} [{mic_bar:30s}]  "
                      f"Model: neg={probs[0]:.3f} pos={probs[1]:.3f}{trigger}")

    except KeyboardInterrupt:
        print("\nStopped.")
