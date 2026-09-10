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

// ─── Phase 1: Capture & Ring Buffer ─────────────────────────────────────────
// R1: BCLK on D0 (GPIO 1), WS on D1 (GPIO 0), DIN on D4 (GPIO 23).
// Frees GPIO 11 (U0TXD boot rom conflict) and keeps I2C pins unburned.
#define SPECTRA_I2S_BCLK_GPIO     1        // D0 on XIAO ESP32-C5 (INMP441 SCK)
#define SPECTRA_I2S_WS_GPIO       0        // D1 on XIAO ESP32-C5 (INMP441 WS)
#define SPECTRA_I2S_DIN_GPIO      23       // D4 on XIAO ESP32-C5 (INMP441 SD)

#define SPECTRA_RING_SECONDS      3        // 3-second ring buffer capacity
#define SPECTRA_RING_CAPACITY     (SPECTRA_SAMPLE_RATE * SPECTRA_RING_SECONDS) // 48000 samples (96 KB)
#define SPECTRA_WINDOW_SAMPLES    16000    // 1.0s inference window
#define SPECTRA_HOP_SAMPLES       8000     // 0.5s hop (50% overlap)
#define SPECTRA_CAPTURE_BLOCK_SAMPLES 4000 // 0.25s DMA read block (8 KB)

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

// ─── Phase 3: Inference & Parity Constants ─────────────────────────────────
#define SPECTRA_RESOLVER_OPS       5        // Minimal unique op count: CONV, DW_CONV, MEAN, FC, SOFTMAX
#define SPECTRA_LOGITS_TOLERANCE   0.02f    // Gate: per-logit |Δ| <= 0.02 on all 50 clips
#define SPECTRA_MAX_INFERENCE_MS   200.0f   // Soft budget per invoke @ 240 MHz (hop budget < 250 ms)
#define SPECTRA_TARGET_SRAM_KB     256      // Internal SRAM budget discipline

// Decision threshold for positive classification (post-dequantization).
#define SPECTRA_POSITIVE_THRESHOLD 0.5f

// ─── Model Metadata ────────────────────────────────────────────────────────
#define SPECTRA_MODEL_SIZE_BYTES   39552

// ─── Board: XIAO ESP32-C5 ──────────────────────────────────────────────────
#define SPECTRA_LED_GPIO           27       // Active HIGH

#endif  // SPECTRA_CONFIG_H_
