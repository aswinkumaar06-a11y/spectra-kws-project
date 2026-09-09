"""
Test Suite: Energy-gate validation across quiet, whisper, and far-field speech.
Measures the effect of amplitude scaling on speech detection and model recall.
"""
import sys
import os
import glob
import unittest
import numpy as np
import librosa

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

from audio_utils import (
    load_and_fix_audio,
    normalize_amplitude,
    check_energy_gate,
    extract_mfcc,
    ENERGY_GATE_THRESHOLD
)
from test_spectra_model import SpectraModel

class TestEnergyGate(unittest.TestCase):
    def test_energy_gate_and_far_field_sensitivity(self):
        model_path = "models/spectra_model.tflite"
        if not os.path.exists(model_path):
            self.skipTest("Model file not found")

        model = SpectraModel(model_path=model_path)
        test_pos_files = glob.glob("dataset/splits/test_raw/positive/*.wav")
        if not test_pos_files:
            test_pos_files = glob.glob("dataset/raw/positive/**/*.wav", recursive=True)
        if not test_pos_files:
            self.skipTest("No positive audio files found")

        # Pick up to 10 real positive clips
        sample_files = test_pos_files[:10]

        # Test gains: from loud (1.0) down to far-field (0.05, 0.03) and sub-threshold (0.01)
        scale_factors = [1.0, 0.5, 0.2, 0.1, 0.05, 0.035, 0.02, 0.01]

        results = {}
        for scale in scale_factors:
            passed_gate = 0
            detected = 0
            confidences = []

            for f in sample_files:
                audio, _ = librosa.load(f, sr=16000)
                audio = load_and_fix_audio(audio)
                # Scale peak amplitude to simulate distance/quietness
                max_a = np.max(np.abs(audio))
                if max_a > 0:
                    scaled_audio = (audio / max_a) * scale
                else:
                    scaled_audio = audio

                has_energy, max_amp = check_energy_gate(scaled_audio, threshold=ENERGY_GATE_THRESHOLD)
                if has_energy:
                    passed_gate += 1
                    norm_audio = normalize_amplitude(scaled_audio)
                    mfcc = extract_mfcc(norm_audio)
                    probs = model.predict(mfcc)
                    conf = probs[1]
                    confidences.append(conf)
                    if conf > 0.5:
                        detected += 1
                else:
                    confidences.append(0.0)

            results[scale] = {
                "gate_pass_rate": passed_gate / len(sample_files),
                "detection_rate": detected / len(sample_files),
                "mean_conf": float(np.mean(confidences))
            }

        print("\n" + "=" * 65)
        print("  Energy Gate Sensitivity & Far-Field Attenuation Evaluation")
        print("=" * 65)
        print(f"Energy Gate Threshold: {ENERGY_GATE_THRESHOLD}")
        print(f"{'Scale Factor':<15} | {'Peak Amp':<10} | {'Gate Pass Rate':<16} | {'Trigger Rate':<14} | {'Mean Conf'}")
        print("-" * 65)
        for scale, r in results.items():
            print(f"{scale:<15.3f} | {scale:<10.3f} | {r['gate_pass_rate']*100:<15.1f}% | {r['detection_rate']*100:<13.1f}% | {r['mean_conf']:.3f}")
        print("=" * 65)

        # Assertions
        # Speech at scale >= 0.05 (normal, quiet, and modest far-field) MUST pass the gate
        self.assertEqual(results[0.05]["gate_pass_rate"], 1.0, "Speech at amplitude 0.05 must pass energy gate")
        # Sub-threshold signal (0.01) MUST be rejected by the gate
        self.assertEqual(results[0.01]["gate_pass_rate"], 0.0, "Low noise (0.01) must be rejected by energy gate")

if __name__ == "__main__":
    unittest.main()
