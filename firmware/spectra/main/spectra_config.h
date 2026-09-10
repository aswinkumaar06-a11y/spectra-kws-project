/**
 * spectra_config.h — Spectra KWS Frozen Contract
 *
 * Single source of truth for all DSP, model, and quantization parameters.
 * These values MUST match the desktop pipeline (audio_utils.py) exactly.
 * Any mismatch causes silent accuracy degradation.
 *
 * Target: XIAO ESP32-C5 (RISC-V, 384 KB SRAM, 8 MB PSRAM, 8 MB Flash)
 * Derived from: models/model_manifest.json + TFLite interpreter inspection
 * Commit ref: c3e24c1e (desktop pipeline v2)
 */

#ifndef SPECTRA_CONFIG_H_
#define SPECTRA_CONFIG_H_

#include <stdint.h>

// ─── Audio Capture ──────────────────────────────────────────────────────────
#define SPECTRA_SAMPLE_RATE       16000    // Hz
#define SPECTRA_CLIP_DURATION_MS  1000     // milliseconds
#define SPECTRA_CLIP_SAMPLES      16000    // SAMPLE_RATE * CLIP_DURATION_S
#define SPECTRA_AUDIO_BYTES       32000    // CLIP_SAMPLES * sizeof(int16_t)

// ─── Feature Extraction (MFCC) ─────────────────────────────────────────────
#define SPECTRA_N_MFCC            40       // number of MFCC coefficients
#define SPECTRA_N_FFT             1024     // FFT window size
#define SPECTRA_HOP_LENGTH        512      // FFT hop (stride)
#define SPECTRA_N_FRAMES          32       // fixed number of time frames
#define SPECTRA_N_MEL_BINS        40       // Mel filterbank bins (= N_MFCC)

// ─── Model I/O ──────────────────────────────────────────────────────────────
#define SPECTRA_INPUT_HEIGHT       40      // MFCC coefficients (rows)
#define SPECTRA_INPUT_WIDTH        32      // time frames (columns)
#define SPECTRA_INPUT_CHANNELS     1       // mono
#define SPECTRA_OUTPUT_CLASSES     2       // [negative, positive]

// ─── Quantization Parameters (INT8 affine) ──────────────────────────────────
// Input:  real_value = (int8_value - zero_point) * scale
//         int8_value = clip(round(real_value / scale) + zero_point, -128, 127)
#define SPECTRA_INPUT_SCALE        3.9962144f
#define SPECTRA_INPUT_ZERO_POINT   60

// Output: probability = (int8_value - zero_point) * scale
#define SPECTRA_OUTPUT_SCALE       0.00390625f
#define SPECTRA_OUTPUT_ZERO_POINT  (-128)

// ─── Energy Gate ────────────────────────────────────────────────────────────
// Minimum peak amplitude (0.0–1.0 float) to consider audio as speech.
// Below this threshold, skip inference entirely (saves power).
#define SPECTRA_ENERGY_GATE_THRESHOLD  0.03f

// ─── Inference ──────────────────────────────────────────────────────────────
// Tensor arena size for TFLM — must be tuned empirically per model.
// Start generous, then shrink after measuring actual usage (Phase 3).
// Placement: BSS segment (internal SRAM), NOT stack.
#define SPECTRA_TENSOR_ARENA_SIZE  (80 * 1024)  // 80 KB provisional

// Decision threshold for positive classification (post-dequantization).
#define SPECTRA_POSITIVE_THRESHOLD 0.5f

// ─── Model Metadata ────────────────────────────────────────────────────────
#define SPECTRA_MODEL_SIZE_BYTES   39552

// ─── Board: XIAO ESP32-C5 ──────────────────────────────────────────────────
#define SPECTRA_LED_GPIO           27      // Onboard LED (board-alive proof)

#endif  // SPECTRA_CONFIG_H_
