#!/usr/bin/env python3
"""
test_p6_tooling.py — Unit Tests for Phase 6 Tools (analyze_fa.py and calibrate_vad_audio.py).

Verifies:
  - analyze_fa.py:
    1. Zero TRIG lines (Poisson 95% bound computation without crash)
    2. Malformed / corrupted serial log resilience
    3. Near-threshold [0.50, 0.70) vs high-confidence (>= 0.90) bucket classification
    4. Duration and uptime parsing from timestamps
  - calibrate_vad_audio.py:
    5. Pure silence floor (enforces 0.010 minimum clamp)
    6. Silence + spike acoustic distribution (p99 * 4 path)
    7. Stereo downmixing to mono
    8. Rejection of non-16kHz sample rates
"""

import os
import sys
import unittest
import numpy as np

# Ensure project root in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.analyze_fa import analyze_log_content, compute_poisson_upper_bound_95
from scripts.calibrate_vad_audio import calibrate_from_audio, compute_rms_frames


class TestAnalyzeFA(unittest.TestCase):

    def test_zero_triggers_poisson_bound(self):
        """Zero events in 10 hours must yield Poisson 95% bound < 0.30 / h without error."""
        log_text = (
            "[HEALTH] HEALTH uptime=36000s sram_min=172000B sram_cur=172000B psram_min=8000000B inf_count=72000 inf_avg_us=15800 trigs=0 overruns=0 freq=80MHz\n"
        )
        res = analyze_log_content(log_text, declared_hours=10.0)
        self.assertEqual(res["triggers"], 0)
        self.assertEqual(res["fa_rate"], 0.0)
        # Exact Poisson bound for 0 events: 0.5 * chi2.ppf(0.95, 2) / 10 = 2.9957 / 10 ≈ 0.2996
        self.assertAlmostEqual(res["poisson_95_upper"], 0.2996, places=3)
        self.assertLess(res["poisson_95_upper"], 0.30)

    def test_malformed_and_garbage_serial_lines(self):
        """Tool must cleanly ignore binary corruption, truncated logs, and gibberish."""
        garbage_log = (
            "\x00\x01\xFF\xFE\n"
            "RANDOM CORRUPTED SERIAL DATA %^&*(\n"
            "TRIG tick= incomplete line\n"
            "TRIG tick=42 t_ms=21000 p=[0.800, 0.850, 0.950] pre_ts=20000\n"
            "MORE JUNK LINE 12345\n"
            "[HEALTH] HEALTH uptime=3600s sram_min=170000B sram_cur=170000B psram_min=8000000B inf_count=7200 inf_avg_us=15000 trigs=1 overruns=0 freq=80MHz\n"
        )
        res = analyze_log_content(garbage_log, declared_hours=1.0)
        self.assertEqual(res["triggers"], 1)
        self.assertEqual(len(res["p_values"]), 1)
        self.assertAlmostEqual(res["p_values"][0], 0.950, places=3)

    def test_confidence_bucket_classification(self):
        """Mixed near-threshold and high-confidence triggers must be accurately parsed."""
        log_text = (
            "TRIG tick=10 t_ms=5000 p=[0.400, 0.520, 0.550] pre_ts=4000\n"   # near-threshold (0.55)
            "TRIG tick=20 t_ms=10000 p=[0.600, 0.700, 0.750] pre_ts=9000\n"  # mid-range (0.75)
            "TRIG tick=30 t_ms=15000 p=[0.850, 0.920, 0.980] pre_ts=14000\n" # high-confidence (0.98)
            "[HEALTH] HEALTH uptime=7200s sram_min=172000B inf_avg_us=15000 freq=80MHz\n"
        )
        res = analyze_log_content(log_text, declared_hours=2.0)
        self.assertEqual(res["triggers"], 3)
        self.assertEqual(res["fa_rate"], 1.5)
        p_arr = np.array(res["p_values"])
        self.assertEqual(np.sum((p_arr >= 0.5) & (p_arr < 0.7)), 1)
        self.assertEqual(np.sum((p_arr >= 0.7) & (p_arr < 0.9)), 1)
        self.assertEqual(np.sum(p_arr >= 0.9), 1)

    def test_timestamp_duration_parsing(self):
        """Calculates duration accurately from maximum timestamp or health uptime."""
        log_text = (
            "TRIG tick=100 t_ms=18000000 p=[0.90, 0.95, 0.98] pre_ts=17999000\n" # 5 hours
            "[HEALTH] HEALTH uptime=18000s sram_min=170000B inf_avg_us=15000 freq=80MHz\n"
        )
        res = analyze_log_content(log_text)
        self.assertAlmostEqual(res["hours"], 5.0, places=2)


class TestCalibrateVAD(unittest.TestCase):

    def test_pure_silence_floor_clamp(self):
        """Pure silence (all zeros) must clamp to the 0.010 safety minimum."""
        audio = np.zeros(16000 * 5, dtype=np.float32)  # 5 seconds of zero silence
        res = calibrate_from_audio(audio, sr=16000)
        self.assertEqual(res["p99_silence"], 0.0)
        self.assertEqual(res["calibrated_threshold"], 0.010)

    def test_silence_plus_spike_distribution(self):
        """Synthetic ambient noise with a spike must use the p99 * 4 path correctly."""
        np.random.seed(42)
        # 10 seconds of background noise with RMS ~ 0.003
        audio = np.random.normal(0, 0.003, 16000 * 10).astype(np.float32)
        # Insert a spike in 1 frame (100 ms)
        audio[1600:3200] = 0.05
        res = calibrate_from_audio(audio, sr=16000)
        # p99 ignores rare spike (1 frame out of 100 is 1%)
        self.assertGreater(res["p99_silence"], 0.002)
        self.assertLess(res["p99_silence"], 0.010)
        self.assertGreaterEqual(res["calibrated_threshold"], 0.010)

    def test_stereo_audio_downmix(self):
        """Stereo 2-channel audio must downmix to mono without crashing."""
        stereo_audio = np.zeros((16000 * 2, 2), dtype=np.float32)
        res = calibrate_from_audio(stereo_audio, sr=16000)
        self.assertEqual(res["calibrated_threshold"], 0.010)

    def test_non_16k_rejection(self):
        """Sample rates other than 16,000 Hz must be rejected with ValueError."""
        audio = np.zeros(8000 * 2, dtype=np.float32)
        with self.assertRaises(ValueError):
            calibrate_from_audio(audio, sr=8000)


if __name__ == "__main__":
    unittest.main()
