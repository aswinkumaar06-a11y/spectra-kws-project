import os
import random
import numpy as np
from scipy.io import wavfile

# Path configuration based on your structure
NOISE_DIR = "dataset/raw/noise"
SILENCE_DIR = "dataset/raw/silence"
SAMPLE_RATE = 16000
NUM_SAMPLES = int(SAMPLE_RATE * 1.0) # 1 second clips

def generate_dataset():
    os.makedirs(SILENCE_DIR, exist_ok=True)
    
    # 1. Generate pure digital silence (zeros)
    for i in range(150):
        filepath = os.path.join(SILENCE_DIR, f"pure_silence_{i}.wav")
        audio_data = np.zeros(NUM_SAMPLES, dtype=np.int16)
        wavfile.write(filepath, SAMPLE_RATE, audio_data)

    # 2. Slice 1-second chunks from background noise
    noise_files = [f for f in os.listdir(NOISE_DIR) if f.endswith('.wav')]
    if not noise_files:
        print("No ambient noise files found in dataset/raw/noise/")
        return

    for i in range(150):
        src_file = random.choice(noise_files)
        sample_rate, data = wavfile.read(os.path.join(NOISE_DIR, src_file))
        
        if len(data) > NUM_SAMPLES:
            start_idx = random.randint(0, len(data) - NUM_SAMPLES)
            segment = data[start_idx : start_idx + NUM_SAMPLES]
            
            # Reduce volume to simulate a quiet room
            segment = (segment * 0.3).astype(np.int16)
            
            filepath = os.path.join(SILENCE_DIR, f"ambient_silence_{i}.wav")
            wavfile.write(filepath, SAMPLE_RATE, segment)
            
    print(f"Generated 300 silence clips in {SILENCE_DIR}")

if __name__ == "__main__":
    generate_dataset()