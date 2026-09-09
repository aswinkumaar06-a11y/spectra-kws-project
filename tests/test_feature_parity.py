"""
Test Suite: Feature parity, quantization math, and energy gating.
"""
import sys
import os
import unittest
import numpy as np

# Add scripts directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

from audio_utils import (
    load_and_fix_audio,
    normalize_amplitude,
    check_energy_gate,
    extract_mfcc,
    preprocess_audio_clip,
    quantize_to_int8,
    dequantize_from_int8,
    SAMPLE_RATE,
    TARGET_SAMPLES,
    N_MFCC,
    N_FRAMES
)

class TestAudioUtils(unittest.TestCase):
    def test_audio_length_fixing(self):
        # Shorter audio
        short_audio = np.ones(8000, dtype=np.float32)
        fixed_short = load_and_fix_audio(short_audio)
        self.assertEqual(len(fixed_short), TARGET_SAMPLES)
        self.assertEqual(fixed_short[7999], 1.0)
        self.assertEqual(fixed_short[8000], 0.0)

        # Longer audio
        long_audio = np.ones(24000, dtype=np.float32)
        fixed_long = load_and_fix_audio(long_audio)
        self.assertEqual(len(fixed_long), TARGET_SAMPLES)

    def test_amplitude_normalization(self):
        audio = np.array([-0.5, 0.2, 0.4, -0.8], dtype=np.float32)
        norm_audio = normalize_amplitude(audio)
        self.assertAlmostEqual(np.max(np.abs(norm_audio)), 1.0, places=5)
        self.assertAlmostEqual(norm_audio[3], -1.0, places=5)

        # Pure silence doesn't crash or blow up
        silence = np.zeros(16000, dtype=np.float32)
        norm_silence = normalize_amplitude(silence)
        self.assertTrue(np.all(norm_silence == 0.0))

    def test_energy_gate(self):
        silence = np.zeros(16000, dtype=np.float32)
        has_energy, max_amp = check_energy_gate(silence, threshold=0.03)
        self.assertFalse(has_energy)
        self.assertEqual(max_amp, 0.0)

        quiet_noise = np.random.uniform(-0.01, 0.01, 16000).astype(np.float32)
        has_energy, max_amp = check_energy_gate(quiet_noise, threshold=0.03)
        self.assertFalse(has_energy)

        speech = np.random.uniform(-0.25, 0.25, 16000).astype(np.float32)
        has_energy, max_amp = check_energy_gate(speech, threshold=0.03)
        self.assertTrue(has_energy)

    def test_mfcc_shape_and_parity(self):
        audio = np.random.uniform(-0.5, 0.5, TARGET_SAMPLES).astype(np.float32)
        mfcc1 = preprocess_audio_clip(audio, normalize=True)
        self.assertEqual(mfcc1.shape, (N_MFCC, N_FRAMES))

        # Extracting manually step-by-step must match preprocess_audio_clip bit-for-bit
        fixed = load_and_fix_audio(audio)
        norm = normalize_amplitude(fixed)
        mfcc2 = extract_mfcc(norm)
        np.testing.assert_array_equal(mfcc1, mfcc2)

    def test_int8_quantization_math(self):
        scale = 0.05
        zero_point = 0
        x = np.array([-6.4, -1.0, 0.0, 1.0, 6.35], dtype=np.float32)
        # -6.4 / 0.05 = -128 -> clip(-128) = -128
        # -1.0 / 0.05 = -20
        # 0.0 / 0.05 = 0
        # 1.0 / 0.05 = 20
        # 6.35 / 0.05 = 127
        q = quantize_to_int8(x, scale, zero_point)
        expected_q = np.array([-128, -20, 0, 20, 127], dtype=np.int8)
        np.testing.assert_array_equal(q, expected_q)

        # Test clipping at extreme bounds
        extreme_x = np.array([-20.0, 20.0], dtype=np.float32)
        extreme_q = quantize_to_int8(extreme_x, scale, zero_point)
        np.testing.assert_array_equal(extreme_q, np.array([-128, 127], dtype=np.int8))

        # Test dequantization
        r = dequantize_from_int8(q, scale, zero_point)
        np.testing.assert_allclose(r, np.array([-6.4, -1.0, 0.0, 1.0, 6.35]), atol=1e-5)

if __name__ == "__main__":
    unittest.main()
