/**
 * tflm.h — TensorFlow Lite for Microcontrollers (TFLM) Inference Engine
 *
 * Phase 3: Model Contract Verification, Minimal 5-Op Resolver,
 *          Internal BSS Arena Management, Input Ingestion & Dequantization.
 *
 * Target: Seeed Studio XIAO ESP32-C5 (RISC-V @ 240 MHz)
 */

#ifndef SPECTRA_TFLM_H_
#define SPECTRA_TFLM_H_

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Inference classification result.
 */
typedef struct {
    float p_neg;         // Class 0 probability (Negative / Other speech)
    float p_pos;         // Class 1 probability (Positive / "Spectra")
    int8_t raw_neg;      // Raw quantized output tensor element 0
    int8_t raw_pos;      // Raw quantized output tensor element 1
    int argmax;          // Predicted class index (0 or 1)
    uint32_t cycles;     // CPU cycle count for Invoke()
    float inference_ms;  // Duration in milliseconds @ 240 MHz
} tflm_result_t;

/**
 * TFLM engine status and memory metrics.
 */
typedef struct {
    size_t arena_capacity;
    size_t arena_used;
    size_t model_bytes;
    bool is_minimal_resolver;
} tflm_info_t;

/**
 * Initialize TFLM engine:
 * 1. Verifies model binary header ('TFL3', schema v3).
 * 2. Asserts input/output shapes, dtypes, scales, and zero-points against frozen contract.
 * 3. Builds Op Resolver:
 *      - Default: MicroMutableOpResolver<5> (CONV_2D, DEPTHWISE_CONV_2D, MEAN, FULLY_CONNECTED, SOFTMAX)
 *      - Fallback: AllOpsResolver (controlled via CONFIG_SPECTRA_USE_ALLOPS)
 * 4. Allocates tensors in internal SRAM BSS arena (16-byte aligned).
 * 5. Runs recording allocator analysis to log exact memory usage.
 *
 * Returns true if all contract assertions pass and tensors are allocated.
 */
bool tflm_init(void);

/**
 * Returns true if engine has been successfully initialized.
 */
bool tflm_is_initialized(void);

/**
 * Ingest 1,280-byte INT8 MFCC feature map into model input tensor.
 * Input format: [1, 40, 32, 1] NHWC contiguous layout.
 */
bool tflm_feed_input(const int8_t* mfcc_features_1280);

/**
 * Execute model inference and dequantize outputs.
 * Populates out_result with probabilities and execution timing.
 * Enforces health invariants: 0.99 <= p_neg + p_pos <= 1.01, zero NaN / Inf.
 */
bool tflm_invoke(tflm_result_t* out_result);

/**
 * Run zeros-smoke regression test:
 * Ingests 1,280 zeros, expects p_neg ≈ 0.9961 ± 0.01, p_pos ≈ 0.0.
 */
bool tflm_run_zeros_smoke(tflm_result_t* out_result);

/**
 * Retrieve memory allocation telemetry for budget accounting.
 */
void tflm_get_info(tflm_info_t* out_info);

#ifdef __cplusplus
}
#endif

#endif  // SPECTRA_TFLM_H_
