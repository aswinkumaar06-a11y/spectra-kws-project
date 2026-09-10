"""
test_model_contract.py — Host-side validation of the Spectra TFLite model contract.

Verifies that the INT8-quantized model has the exact tensor shapes,
dtypes, and quantization parameters that the firmware expects.
Runs on x86 (no ESP-IDF needed).

Exit criteria: All assertions pass → model is safe to embed in firmware.
"""

import os
import sys
import struct
import numpy as np

# Add project root to path for audio_utils import
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

# ─── Frozen Contract Constants (must match spectra_config.h) ─────────────────
EXPECTED_INPUT_SHAPE = [1, 40, 32, 1]
EXPECTED_INPUT_DTYPE = np.int8
EXPECTED_INPUT_SCALE = 3.9962144
EXPECTED_INPUT_ZERO_POINT = 60

EXPECTED_OUTPUT_SHAPE = [1, 2]
EXPECTED_OUTPUT_DTYPE = np.int8
EXPECTED_OUTPUT_SCALE = 0.00390625
EXPECTED_OUTPUT_ZERO_POINT = -128

EXPECTED_MODEL_SIZE = 39552
TFLITE_MAGIC = b'TFL3'


def test_model_contract():
    model_path = os.path.join(PROJECT_ROOT, 'models', 'spectra_model.tflite')
    assert os.path.exists(model_path), f"Model file not found: {model_path}"

    # ── File-level checks ────────────────────────────────────────────────
    file_size = os.path.getsize(model_path)
    print(f"[CHECK] Model file size: {file_size} bytes (expected {EXPECTED_MODEL_SIZE})")
    assert file_size == EXPECTED_MODEL_SIZE, \
        f"Model size mismatch: {file_size} != {EXPECTED_MODEL_SIZE}"

    with open(model_path, 'rb') as f:
        header = f.read(8)
    # TFLite FlatBuffer: bytes 4-7 are the magic "TFL3"
    magic = header[4:8]
    print(f"[CHECK] FlatBuffer magic: {magic} (expected {TFLITE_MAGIC})")
    assert magic == TFLITE_MAGIC, f"Bad FlatBuffer magic: {magic}"

    # ── Load interpreter ─────────────────────────────────────────────────
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    # ── Input tensor checks ──────────────────────────────────────────────
    in_shape = list(input_details['shape'])
    in_dtype = input_details['dtype']
    in_quant = input_details['quantization_parameters']
    in_scale = float(in_quant['scales'][0])
    in_zp = int(in_quant['zero_points'][0])

    print(f"[CHECK] Input shape: {in_shape} (expected {EXPECTED_INPUT_SHAPE})")
    assert in_shape == EXPECTED_INPUT_SHAPE, \
        f"Input shape mismatch: {in_shape} != {EXPECTED_INPUT_SHAPE}"

    print(f"[CHECK] Input dtype: {in_dtype} (expected {EXPECTED_INPUT_DTYPE})")
    assert in_dtype == EXPECTED_INPUT_DTYPE, \
        f"Input dtype mismatch: {in_dtype} != {EXPECTED_INPUT_DTYPE}"

    print(f"[CHECK] Input scale: {in_scale:.7f} (expected {EXPECTED_INPUT_SCALE:.7f})")
    assert abs(in_scale - EXPECTED_INPUT_SCALE) < 1e-4, \
        f"Input scale mismatch: {in_scale} != {EXPECTED_INPUT_SCALE}"

    print(f"[CHECK] Input zero_point: {in_zp} (expected {EXPECTED_INPUT_ZERO_POINT})")
    assert in_zp == EXPECTED_INPUT_ZERO_POINT, \
        f"Input zero_point mismatch: {in_zp} != {EXPECTED_INPUT_ZERO_POINT}"

    # ── Output tensor checks ─────────────────────────────────────────────
    out_shape = list(output_details['shape'])
    out_dtype = output_details['dtype']
    out_quant = output_details['quantization_parameters']
    out_scale = float(out_quant['scales'][0])
    out_zp = int(out_quant['zero_points'][0])

    print(f"[CHECK] Output shape: {out_shape} (expected {EXPECTED_OUTPUT_SHAPE})")
    assert out_shape == EXPECTED_OUTPUT_SHAPE, \
        f"Output shape mismatch: {out_shape} != {EXPECTED_OUTPUT_SHAPE}"

    print(f"[CHECK] Output dtype: {out_dtype} (expected {EXPECTED_OUTPUT_DTYPE})")
    assert out_dtype == EXPECTED_OUTPUT_DTYPE, \
        f"Output dtype mismatch: {out_dtype} != {EXPECTED_OUTPUT_DTYPE}"

    print(f"[CHECK] Output scale: {out_scale:.8f} (expected {EXPECTED_OUTPUT_SCALE:.8f})")
    assert abs(out_scale - EXPECTED_OUTPUT_SCALE) < 1e-6, \
        f"Output scale mismatch: {out_scale} != {EXPECTED_OUTPUT_SCALE}"

    print(f"[CHECK] Output zero_point: {out_zp} (expected {EXPECTED_OUTPUT_ZERO_POINT})")
    assert out_zp == EXPECTED_OUTPUT_ZERO_POINT, \
        f"Output zero_point mismatch: {out_zp} != {EXPECTED_OUTPUT_ZERO_POINT}"

    # ── Smoke test: zero-input inference ─────────────────────────────────
    input_data = np.zeros(EXPECTED_INPUT_SHAPE, dtype=np.int8)
    interpreter.set_tensor(input_details['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details['index'])

    # Dequantize output
    probs = (output_data.astype(np.float32) - EXPECTED_OUTPUT_ZERO_POINT) * EXPECTED_OUTPUT_SCALE
    print(f"[CHECK] Zero-input inference: neg={probs[0,0]:.4f} pos={probs[0,1]:.4f}")
    assert output_data.shape == tuple(EXPECTED_OUTPUT_SHAPE), \
        f"Output shape after invoke: {output_data.shape}"

    # Probabilities should sum to ~1.0 (softmax output)
    prob_sum = float(np.sum(probs))
    print(f"[CHECK] Probability sum: {prob_sum:.4f} (expected ~1.0)")
    assert 0.9 < prob_sum < 1.1, f"Probability sum out of range: {prob_sum}"

    print()
    print("=" * 60)
    print("  ALL CONTRACT CHECKS PASSED")
    print("  Model is safe to embed in XIAO ESP32-C5 firmware")
    print("=" * 60)


if __name__ == '__main__':
    test_model_contract()
