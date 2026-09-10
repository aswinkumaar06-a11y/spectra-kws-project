#!/usr/bin/env python3
"""
test_p6_tooling.py — Unit Tests for Phase 6 Tools (analyze_fa.py and calibrate_vad_audio.py).

Verifies exact canonical algorithm behaviors required by Spectra Phase 6 Specification:
  - analyze_fa.py (Algorithm FA-1):
    1. zero TRIG lines in a 10 h log → λ bound ≈ 0.30/h, verdict PASS(strong), no crash
    2. log containing garbage/malformed lines → valid lines still parsed; unparseable count reported; no exception
    3. mixed-confidence log (p = 0.55, 0.68, 0.95) → bucket counts {near:2, mid:0, high:1}
    4. timestamp math: log with wall-clock stamps vs tick-derived duration must agree within one tick
  - calibrate_vad_audio.py (Algorithm CAL-1):
    1. pure low-level silence → TH must equal the 0.010 floor (max() branch)
    2. silence + single loud spike → TH = p99×4 branch (spike excluded by percentile)
    3. non-16-kHz or stereo input → documented reject-or-convert behavior, no silent wrong-Th
    4. endpoint validation against normative vectors (V1, V2, V3)
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
from scripts.calibrate_vad_audio import calibrate_from_audio, compute_rms_frames, validate_v1_v2_v3


class TestAnalyzeFA(unittest.TestCase):
    """Canonical test suite for Algorithm FA-1 (analyze_fa.py)."""

    def test_fixture1_zero_triggers_10h(self):
        """
        Fixture 1: zero TRIG lines in a 10 h log:
        λ bound ≈ 0.30/h, verdict PASS(strong), no crash.
        """
        log_text = (
            "[BOOT] Spectra KWS boot banner tick=0\n"
            "[HEALTH] HEALTH uptime=36000s sram_min=172000B inf_count=72000 trigs=0 freq=80MHz\n"
            "[SOAK] Soak completed tick=72000\n"
        )
        res = analyze_log_content(log_text)
        self.assertEqual(res["k_events"], 0)
        self.assertEqual(res["fa_rate"], 0.0)
        self.assertAlmostEqual(res["duration_hours"], 10.0, places=2)
        # Poisson upper bound for k=0, T=10h: 5.99146 / 20 = 0.29957 / h ≈ 0.30/h
        self.assertAlmostEqual(res["lambda_upper"], 0.30, places=2)
        self.assertLess(res["lambda_upper"], 0.3000)
        self.assertTrue(res["verdict"].startswith("PASS (strong)"))
        self.assertEqual(res["status_code"], "PASS")

    def test_fixture2_garbage_and_malformed_lines(self):
        """
        Fixture 2: log containing garbage/malformed lines:
        valid lines still parsed; unparseable count reported; no exception.
        """
        garbage_log = (
            "\x00\x01\xFF\xFE\n"
            "RANDOM CORRUPTED SERIAL STREAM %^&*(\n"
            "MALFORMED_LINE_WITHOUT_MEANING\n"
            "TRIG tick= incomplete syntax\n"
            "TRIG tick=1000 t_ms=500000 p=0.85\n"
            "MORE JUNK LINE 99999\n"
            "[HEALTH] HEALTH uptime=3600s trigs=1\n"
        )
        res = analyze_log_content(garbage_log, declared_hours=1.0)
        self.assertEqual(res["k_events"], 1)
        self.assertGreater(res["unparseable_count"], 0)
        self.assertEqual(res["unparseable_count"], 5)  # 5 invalid lines
        self.assertEqual(res["buckets"]["mid"], 1)

    def test_fixture3_mixed_confidence_buckets(self):
        """
        Fixture 3: mixed-confidence log (p = 0.55, 0.68, 0.95):
        bucket counts {near:2, mid:0, high:1}.
        """
        log_text = (
            "TRIG tick=100 p=0.55\n"   # near-threshold [0.50, 0.70)
            "TRIG tick=200 p=0.68\n"   # near-threshold [0.50, 0.70)
            "TRIG tick=300 p=0.95\n"   # high-confidence >= 0.90
            "[HEALTH] HEALTH uptime=7200s\n"
        )
        res = analyze_log_content(log_text, declared_hours=2.0)
        self.assertEqual(res["k_events"], 3)
        self.assertEqual(res["buckets"]["near"], 2)
        self.assertEqual(res["buckets"]["mid"], 0)
        self.assertEqual(res["buckets"]["high"], 1)
        self.assertEqual(res["buckets"], {"near": 2, "mid": 0, "high": 1})

    def test_fixture4_timestamp_math_wall_clock_vs_ticks(self):
        """
        Fixture 4: timestamp math:
        log with wall-clock stamps vs tick-derived duration must agree within one tick (500 ms).
        """
        log_text = (
            "2026-09-10 10:00:00.000 [BOOT] tick=0\n"
            "2026-09-10 12:00:00.000 [SOAK] tick=14400\n"
        )
        res = analyze_log_content(log_text)
        self.assertIsNotNone(res["duration_wallclock_s"])
        self.assertIsNotNone(res["duration_ticks_s"])
        delta_s = abs(res["duration_wallclock_s"] - res["duration_ticks_s"])
        # One tick = 500 ms = 0.5 s
        self.assertLessEqual(delta_s, 0.5)
        self.assertAlmostEqual(res["duration_hours"], 2.0, places=3)


class TestCalibrateVAD(unittest.TestCase):
    """Canonical test suite for Algorithm CAL-1 (calibrate_vad_audio.py)."""

    def test_fixture1_pure_silence_floor(self):
        """
        Fixture 1: pure low-level silence:
        TH must equal the 0.010 floor (max() branch).
        """
        audio = np.zeros(16000 * 5, dtype=np.float32)  # 5 seconds of zero silence
        res = calibrate_from_audio(audio, sr=16000)
        self.assertEqual(res["p99_silence"], 0.0)
        self.assertEqual(res["calibrated_threshold"], 0.010)

    def test_fixture2_silence_plus_spike(self):
        """
        Fixture 2: silence + single loud spike:
        TH = p99×4 branch (spike excluded by percentile).
        """
        np.random.seed(42)
        # 10 seconds of background noise with RMS ~ 0.0035 (400 frames of 25 ms)
        audio = np.random.normal(0, 0.0035, 16000 * 10).astype(np.float32)
        # Insert a spike in 1 frame (400 samples = 25 ms, which is 1/400 = 0.25% of frames)
        audio[1600:2000] = 0.80
        res = calibrate_from_audio(audio, sr=16000)

        # p99 ignores the top 1% (spike is 0.25%), so p99 ~ 0.0035
        # Formula: max(p99 * 4.0, 0.010) -> ~0.014 > 0.010 floor
        self.assertGreater(res["p99_silence"], 0.0028)
        self.assertLess(res["p99_silence"], 0.0050)
        self.assertGreater(res["calibrated_threshold"], 0.010)  # Exercised p99 * 4 branch
        self.assertAlmostEqual(res["calibrated_threshold"], res["p99_silence"] * 4.0, places=4)

    def test_fixture3_non_16k_and_stereo_handling(self):
        """
        Fixture 3: non-16-kHz or stereo input:
        documented reject-or-convert behavior, no silent wrong-Th.
        """
        # 1. Non-16-kHz sample rate must be rejected with ValueError
        audio_8k = np.zeros(8000 * 2, dtype=np.float32)
        with self.assertRaises(ValueError) as ctx:
            calibrate_from_audio(audio_8k, sr=8000)
        self.assertIn("16000 Hz", str(ctx.exception))

        # 2. Stereo input (2 channels) must be gracefully downmixed to mono
        stereo_audio = np.zeros((16000 * 2, 2), dtype=np.float32)
        res_stereo = calibrate_from_audio(stereo_audio, sr=16000)
        self.assertEqual(res_stereo["calibrated_threshold"], 0.010)

    def test_fixture4_endpoint_validation_v1_v2_v3(self):
        """
        Fixture 4: Step 5 validation on normative vectors V1/V2/V3:
        V1 -> START@2 END@8 | V2 -> none | V3 -> START@0 END@9.
        """
        # Testing at calibrated threshold 0.014
        val = validate_v1_v2_v3(0.014)
        self.assertTrue(val["all_pass"])
        self.assertEqual(val["V1"]["detected"], "START@2 END@8")
        self.assertEqual(val["V2"]["detected"], "none")
        self.assertEqual(val["V3"]["detected"], "START@0 END@9")


if __name__ == "__main__":
    unittest.main()

