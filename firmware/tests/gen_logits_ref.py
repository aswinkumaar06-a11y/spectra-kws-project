#!/usr/bin/env python3
"""
gen_logits_ref.py - Generate Desktop TFLite Golden Logits for Phase 3 Parity

Reads 50 INT8 test inputs from firmware/tests/mfcc_goldens.bin,
evaluates them with desktop tflite on models/spectra_model.tflite,
and exports:
  1. firmware/tests/logits_goldens.json (full 50-clip reference dataset)
  2. firmware/spectra/main/test_clips_5.h (5 embedded clips for on-device boot self-test)
"""

import os
import sys
import json
import struct
import numpy as np

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "spectra_model.tflite")
GOLDEN_BIN = os.path.join(PROJECT_ROOT, "firmware", "tests", "mfcc_goldens.bin")
OUT_JSON = os.path.join(PROJECT_ROOT, "firmware", "tests", "logits_goldens.json")
OUT_HEADER = os.path.join(PROJECT_ROOT, "firmware", "spectra", "main", "test_clips_5.h")


def main():
    print("=== Generating Phase 3 TFLM Golden Reference Logits ===")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    if not os.path.exists(GOLDEN_BIN):
        raise FileNotFoundError(f"Golden bundle not found: {GOLDEN_BIN}")

    # Load desktop TFLite interpreter
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    in_shape = input_details["shape"]
    in_scale, in_zp = input_details["quantization"]
    out_scale, out_zp = output_details["quantization"]

    print(f"Model: {MODEL_PATH} ({os.path.getsize(MODEL_PATH)} bytes)")
    print(f"Input:  shape={in_shape}, scale={in_scale:.6f}, zp={in_zp}")
    print(f"Output: shape={output_details['shape']}, scale={out_scale:.6f}, zp={out_zp}")

    # Read 50 clips from mfcc_goldens.bin
    with open(GOLDEN_BIN, "rb") as f:
        bundle_bytes = f.read()

    magic, ver, n_clips, sr, tgt, n_mfcc, n_fr = struct.unpack("<4sIIIIII", bundle_bytes[:28])
    assert magic == b"SPMF", f"Invalid magic: {magic}"
    assert n_clips >= 50, f"Expected >= 50 clips, found {n_clips}"

    offset = 124
    goldens = []

    for c in range(n_clips):
        fname = bundle_bytes[offset:offset+64].split(b'\0')[0].decode('utf-8')
        lbl = bundle_bytes[offset+64:offset+80].split(b'\0')[0].decode('utf-8')
        is_hard, max_amp = struct.unpack('<If', bundle_bytes[offset+80:offset+88])
        ref_int8 = np.frombuffer(
            bundle_bytes[offset+88+32000+5120:offset+88+32000+5120+1280],
            dtype=np.int8
        ).copy()
        offset += 88 + 32000 + 5120 + 1280

        # Run desktop inference (shape [1, 40, 32, 1])
        model_in = ref_int8.reshape(1, 40, 32, 1)
        interpreter.set_tensor(input_details["index"], model_in)
        interpreter.invoke()
        raw_out = interpreter.get_tensor(output_details["index"])[0]

        # Dequantize: p[i] = (q[i] + 128) / 256.0
        p_neg = (float(raw_out[0]) - float(out_zp)) * float(out_scale)
        p_pos = (float(raw_out[1]) - float(out_zp)) * float(out_scale)
        argmax = int(np.argmax(raw_out))
        label_int = 1 if lbl == "positive" else 0

        goldens.append({
            "index": c,
            "filename": fname,
            "label_str": lbl,
            "label": label_int,
            "is_hard": int(is_hard),
            "max_amp": float(max_amp),
            "raw_int8": [int(raw_out[0]), int(raw_out[1])],
            "p_neg": float(p_neg),
            "p_pos": float(p_pos),
            "argmax": argmax,
            "features_int8": [int(b) for b in ref_int8],
            "int8_features_hex": ref_int8.tobytes().hex()
        })

    # Save JSON reference
    goldens_doc = {
        "description": "Spectra Phase 3 Desktop TFLite Golden Reference Logits",
        "model_file": "models/spectra_model.tflite",
        "total_clips": len(goldens),
        "clips": goldens
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(goldens_doc, f, indent=2)
    print(f"[EXPORT] {OUT_JSON}: 50 reference clips saved")

    # Select 5 representative clips for on-device boot self-test:
    # 2 positive, 1 standard negative, 2 synthetic hard negatives
    pos_idx = [i for i, g in enumerate(goldens) if g["label_str"] == "positive"][:2]
    neg_idx = [i for i, g in enumerate(goldens) if g["label_str"] == "negative" and g["is_hard"] == 0][:1]
    hard_idx = [i for i, g in enumerate(goldens) if g["is_hard"] == 1][:2]
    selected_5_idx = pos_idx + neg_idx + hard_idx
    assert len(selected_5_idx) == 5, f"Expected 5 clips, got {len(selected_5_idx)}"

    print(f"Selected 5 clips for on-device equivalence test: indices {selected_5_idx}")

    # Generate test_clips_5.h
    with open(OUT_HEADER, "w", encoding="utf-8") as f:
        f.write("/**\n * test_clips_5.h - 5 Embedded Golden Reference Clips for On-Device Equivalence Test\n")
        f.write(" * Generated by gen_logits_ref.py for Spectra Phase 3\n */\n\n")
        f.write("#ifndef TEST_CLIPS_5_H_\n#define TEST_CLIPS_5_H_\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write("#define SPECTRA_NUM_TEST_CLIPS_5 5\n")
        f.write("#define SPECTRA_TEST_CLIP_INT8_SIZE 1280\n\n")
        f.write("typedef struct {\n")
        f.write("    const char* filename;\n")
        f.write("    const char* label;\n")
        f.write("    uint32_t is_hard;\n")
        f.write("    int8_t raw_neg;\n")
        f.write("    int8_t raw_pos;\n")
        f.write("    float p_neg;\n")
        f.write("    float p_pos;\n")
        f.write("    int argmax;\n")
        f.write("    const int8_t* features;\n")
        f.write("} spectra_test_clip_t;\n\n")

        # Write each clip's 1280-byte feature array
        for rank, idx in enumerate(selected_5_idx):
            clip = goldens[idx]
            raw_bytes = bytes.fromhex(clip["int8_features_hex"])
            f.write(f"static const int8_t kTestClipFeatures_{rank}[1280] = {{\n")
            for b_idx, val in enumerate(raw_bytes):
                signed_val = struct.unpack("b", bytes([val]))[0]
                f.write(f"{signed_val:4d},")
                if (b_idx + 1) % 16 == 0:
                    f.write("\n")
            f.write("};\n\n")

        # Write struct array
        f.write("static const spectra_test_clip_t kTestClips5[SPECTRA_NUM_TEST_CLIPS_5] = {\n")
        for rank, idx in enumerate(selected_5_idx):
            clip = goldens[idx]
            f.write(f"    {{\n")
            f.write(f"        \"{clip['filename']}\",\n")
            f.write(f"        \"{clip['label_str']}\",\n")
            f.write(f"        {clip['is_hard']},\n")
            f.write(f"        {clip['raw_int8'][0]},\n")
            f.write(f"        {clip['raw_int8'][1]},\n")
            f.write(f"        {clip['p_neg']:.6f}f,\n")
            f.write(f"        {clip['p_pos']:.6f}f,\n")
            f.write(f"        {clip['argmax']},\n")
            f.write(f"        kTestClipFeatures_{rank}\n")
            f.write(f"    }},\n")
        f.write("};\n\n")
        f.write("#endif  // TEST_CLIPS_5_H_\n")

    print(f"[EXPORT] {OUT_HEADER}: 5 embedded clips written successfully")
    print("=== Golden Logits Reference Generation Complete ===")


if __name__ == "__main__":
    main()
