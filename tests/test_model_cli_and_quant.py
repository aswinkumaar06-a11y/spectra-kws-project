"""
Test Suite: Model CLI loading and INT8 quantization behavior.
"""
import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

from test_spectra_model import SpectraModel
from audio_utils import quantize_to_int8, dequantize_from_int8

class TestModelQuantAndCLI(unittest.TestCase):
    def test_spectra_model_loads_and_infers(self):
        model_path = "models/spectra_model.tflite"
        if not os.path.exists(model_path):
            self.skipTest(f"{model_path} does not exist")

        model = SpectraModel(model_path=model_path)
        dummy_mfcc = np.zeros((40, 32), dtype=np.float32)
        probs = model.predict(dummy_mfcc)
        self.assertEqual(len(probs), 2)
        self.assertAlmostEqual(np.sum(probs), 1.0, delta=0.05)

    def test_missing_model_raises_error(self):
        with self.assertRaises(FileNotFoundError):
            SpectraModel(model_path="models/non_existent_model.tflite")

    def test_int8_quantization_clipping_and_rounding(self):
        scale = 0.1
        zero_point = 0

        # Rounding check: 0.14 -> 1.4 -> round to 1
        q1 = quantize_to_int8(np.array([0.14], dtype=np.float32), scale, zero_point)
        self.assertEqual(q1[0], 1)

        # 0.16 -> 1.6 -> round to 2
        q2 = quantize_to_int8(np.array([0.16], dtype=np.float32), scale, zero_point)
        self.assertEqual(q2[0], 2)

        # Clipping check: +50.0 -> 500 -> clip to 127
        q_pos = quantize_to_int8(np.array([50.0], dtype=np.float32), scale, zero_point)
        self.assertEqual(q_pos[0], 127)

        # Clipping check: -50.0 -> -500 -> clip to -128
        q_neg = quantize_to_int8(np.array([-50.0], dtype=np.float32), scale, zero_point)
        self.assertEqual(q_neg[0], -128)

if __name__ == "__main__":
    unittest.main()
