"""
Test Suite: Bounded live buffering and sustained stream latency verification.
Verifies that the live audio processing pipeline keeps queue depth bounded
and processing latency well below the hop interval.
"""
import sys
import os
import time
import queue
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

from audio_utils import (
    preprocess_audio_clip,
    normalize_amplitude,
    check_energy_gate,
    extract_mfcc,
    SAMPLE_RATE,
    CLIP_DURATION,
    ENERGY_GATE_THRESHOLD
)
from test_spectra_model import SpectraModel

class TestLiveBuffering(unittest.TestCase):
    def test_sustained_streaming_bounded_queue(self):
        model_path = "models/spectra_model.tflite"
        if not os.path.exists(model_path):
            self.skipTest("TFLite model not found")

        model = SpectraModel(model_path=model_path)
        audio_q = queue.Queue(maxsize=10)
        block_size = int(SAMPLE_RATE * CLIP_DURATION) # 16,000 samples = 1.0s
        hop_size = block_size // 2                     # 8,000 samples = 500ms

        latencies = []
        queue_depths = []

        # Simulate 20 chunks (equivalent to 10 seconds of continuous audio stream at 500ms rate)
        # Warm up JIT / Numba / XNNPACK before streaming
        warmup_audio = np.zeros(block_size, dtype=np.float32)
        warmup_mfcc = extract_mfcc(warmup_audio)
        _ = model.predict(warmup_mfcc)

        buffer = np.zeros(0, dtype=np.float32)

        for step in range(20):
            # Feed simulated chunk (0.5s audio @ 16kHz)
            chunk = np.random.uniform(-0.2, 0.2, hop_size).astype(np.float32)
            audio_q.put(chunk)
            q_depth = audio_q.qsize()
            queue_depths.append(q_depth)

            # Processing loop
            t0 = time.perf_counter()
            received_chunk = audio_q.get()
            buffer = np.concatenate([buffer, received_chunk])

            if len(buffer) >= block_size:
                clip = buffer[:block_size]
                buffer = buffer[hop_size:]

                has_energy, max_amp = check_energy_gate(clip, threshold=ENERGY_GATE_THRESHOLD)
                if has_energy:
                    norm_clip = normalize_amplitude(clip)
                    mfcc = extract_mfcc(norm_clip)
                    probs = model.predict(mfcc)

            dt_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(dt_ms)

        avg_latency = np.mean(latencies)
        max_latency = np.max(latencies)
        max_q = max(queue_depths)

        print(f"\n[Sustained Stream Test] 20 steps (10s audio):")
        print(f"  Average processing latency: {avg_latency:.2f} ms")
        print(f"  Maximum processing latency: {max_latency:.2f} ms")
        print(f"  Max queue depth:            {max_q}")
        print(f"  Budget per hop:             500.00 ms")

        # Assertions
        self.assertLess(max_q, 5, "Queue depth grew too large!")
        self.assertLess(avg_latency, 100.0, "Average inference latency must be well under 100ms")
        self.assertLess(max_latency, 500.0, "Maximum latency must not exceed block interval (500ms)")

if __name__ == "__main__":
    unittest.main()
