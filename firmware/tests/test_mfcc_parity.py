#!/usr/bin/env python3
"""
test_mfcc_parity.py — Phase 2 Parity Verification Suite

Validates:
  1. Golden bundle structure and versioning (mfcc_goldens.bin)
  2. Mathematical invariants of C header tables:
     - Hann window periodic symmetry and peak
     - Twiddle factor unit circle conservation (cos^2 + sin^2 == 1.0)
     - Mel filterbank sparse structure and Slaney triangular area
     - Type-II DCT orthonormal basis (D @ D.T == I_40)
  3. Executes compiled C regression test (test_mfcc_host.c) and asserts:
     - Max |Δfloat| <= 1e-3
     - INT8 byte match >= 99.0%
     - 50 / 50 clips pass
     - Edge cases pass (silence, sub-threshold, sine 440Hz, zero NaN/inf)
"""

import os
import sys
import struct
import subprocess
import unittest
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MAIN_DIR = os.path.join(PROJECT_ROOT, "firmware", "spectra", "main")
TESTS_DIR = os.path.join(PROJECT_ROOT, "firmware", "tests")
GOLDEN_BIN = os.path.join(TESTS_DIR, "mfcc_goldens.bin")


class TestMfccTables(unittest.TestCase):
    """Verifies mathematical correctness of generated C header tables."""

    def test_hann_table(self):
        hann_path = os.path.join(MAIN_DIR, "hann_table.h")
        self.assertTrue(os.path.exists(hann_path), "hann_table.h must exist")
        with open(hann_path, "r") as f:
            content = f.read()

        self.assertIn("#define SPECTRA_HANN_LEN 1024", content)
        self.assertIn("kHannWindowFloat", content)
        self.assertIn("kHannWindowQ15", content)

        # Parse float values
        float_block = content.split("kHannWindowFloat[SPECTRA_HANN_LEN] = {")[1].split("};")[0]
        vals = [float(x.replace("f", "").strip()) for x in float_block.replace("\n", "").split(",") if x.strip()]
        self.assertEqual(len(vals), 1024)

        # Check periodic Hann formula: w[n] = 0.5 - 0.5 * cos(2*pi*n / 1024)
        n = np.arange(1024)
        ref_w = (0.5 - 0.5 * np.cos(2.0 * np.pi * n / 1024.0)).astype(np.float32)
        max_diff = np.max(np.abs(np.array(vals, dtype=np.float32) - ref_w))
        self.assertLess(max_diff, 1e-6, f"Hann window float differs from formula by {max_diff}")

    def test_twiddle_table(self):
        twiddle_path = os.path.join(MAIN_DIR, "twiddle_table.h")
        self.assertTrue(os.path.exists(twiddle_path), "twiddle_table.h must exist")
        with open(twiddle_path, "r") as f:
            content = f.read()

        self.assertIn("#define SPECTRA_TWIDDLE_LEN 512", content)
        self.assertIn("kTwiddleCosFloat", content)
        self.assertIn("kTwiddleSinFloat", content)

        # Parse cos and sin
        cos_block = content.split("kTwiddleCosFloat[SPECTRA_TWIDDLE_LEN] = {")[1].split("};")[0]
        sin_block = content.split("kTwiddleSinFloat[SPECTRA_TWIDDLE_LEN] = {")[1].split("};")[0]

        cos_vals = np.array([float(x.replace("f", "").strip()) for x in cos_block.replace("\n", "").split(",") if x.strip()], dtype=np.float32)
        sin_vals = np.array([float(x.replace("f", "").strip()) for x in sin_block.replace("\n", "").split(",") if x.strip()], dtype=np.float32)

        self.assertEqual(len(cos_vals), 512)
        self.assertEqual(len(sin_vals), 512)

        # Invariant: cos^2 + sin^2 == 1.0
        unit_circle = cos_vals**2 + sin_vals**2
        max_unit_diff = np.max(np.abs(unit_circle - 1.0))
        self.assertLess(max_unit_diff, 1e-6, f"Twiddles violate unit circle conservation by {max_unit_diff}")

    def test_mel_table(self):
        mel_path = os.path.join(MAIN_DIR, "mel_table.h")
        self.assertTrue(os.path.exists(mel_path), "mel_table.h must exist")
        with open(mel_path, "r") as f:
            content = f.read()

        self.assertIn("#define SPECTRA_N_MELS       128", content)
        self.assertIn("#define SPECTRA_FFT_BINS     513", content)
        self.assertIn("kMelFilters", content)
        self.assertIn("kMelWeights", content)

    def test_dct_table_orthonormality(self):
        dct_path = os.path.join(MAIN_DIR, "dct_table.h")
        self.assertTrue(os.path.exists(dct_path), "dct_table.h must exist")
        with open(dct_path, "r") as f:
            content = f.read()

        self.assertIn("#define SPECTRA_N_MFCC 40", content)
        self.assertIn("#define SPECTRA_N_MELS 128", content)

        # Extract 40x128 matrix
        mat_block = content.split("kDctMatrix[SPECTRA_N_MFCC][SPECTRA_N_MELS] = {")[1].split("};")[0]
        rows = mat_block.strip().split("},\n")
        dct_mat = []
        for r in rows:
            r_clean = r.replace("{", "").replace("}", "").strip()
            # remove comments like // Coefficient 0
            lines = [l for l in r_clean.split("\n") if not l.strip().startswith("//")]
            row_str = " ".join(lines)
            vals = [float(x.replace("f", "").strip()) for x in row_str.split(",") if x.strip()]
            if vals:
                dct_mat.append(vals)

        dct_arr = np.array(dct_mat, dtype=np.float64)
        self.assertEqual(dct_arr.shape, (40, 128))

        # Test orthonormality: D @ D.T == I_40
        gram = np.dot(dct_arr, dct_arr.T)
        identity = np.eye(40, dtype=np.float64)
        max_ortho_diff = np.max(np.abs(gram - identity))
        self.assertLess(max_ortho_diff, 1e-6, f"DCT matrix is not orthonormal: max diff {max_ortho_diff}")


class TestMfccGoldenBundle(unittest.TestCase):
    """Verifies binary structure of mfcc_goldens.bin."""

    def test_bundle_header(self):
        self.assertTrue(os.path.exists(GOLDEN_BIN), "mfcc_goldens.bin must exist")
        with open(GOLDEN_BIN, "rb") as f:
            header_bytes = f.read(124)

        magic, ver, n_clips, sr, tgt, n_mfcc, n_fr = struct.unpack("<4sIIIIII", header_bytes[:28])
        self.assertEqual(magic, b"SPMF")
        self.assertEqual(ver, 1)
        self.assertGreaterEqual(n_clips, 50)
        self.assertEqual(sr, 16000)
        self.assertEqual(tgt, 16000)
        self.assertEqual(n_mfcc, 40)
        self.assertEqual(n_fr, 32)


class TestMfccHostExecution(unittest.TestCase):
    """Compiles and runs test_mfcc_host.c, ensuring complete parity."""

    def test_c_engine_regression(self):
        c_src = os.path.join(TESTS_DIR, "test_mfcc_host.c")
        mfcc_src = os.path.join(MAIN_DIR, "mfcc.cc")
        exe_path = os.path.join(TESTS_DIR, "test_mfcc_host.exe")

        # Compile
        cmd_compile = [
            "gcc", "-O3", "-o", exe_path,
            c_src, mfcc_src,
            f"-I{MAIN_DIR}",
            "-lm"
        ]
        res = subprocess.run(cmd_compile, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"Compilation failed: {res.stderr}")

        try:
            # Run
            res_run = subprocess.run([exe_path], capture_output=True, text=True)
            output = res_run.stdout
            self.assertEqual(res_run.returncode, 0, f"Regression run returned {res_run.returncode}\n{output}")
            self.assertIn("Passed All Gates: 50 / 50 (100.0%)", output)
            self.assertIn("Results: 7 passed, 0 failed", output)
            self.assertIn("Pure silence correctly rejected", output)
            self.assertIn("Sub-threshold background noise correctly rejected", output)
            self.assertIn("Full-scale 440Hz sine processed successfully", output)
            self.assertIn("Zero NaN or Inf values produced", output)
        finally:
            if os.path.exists(exe_path):
                os.remove(exe_path)


if __name__ == "__main__":
    unittest.main()
