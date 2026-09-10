/**
 * tflm.cc — TensorFlow Lite for Microcontrollers (TFLM) Inference Engine
 *
 * Phase 3: Model Contract Verification, Minimal 5-Op Resolver,
 *          Internal BSS Arena Management, Input Ingestion & Dequantization.
 *
 * Target: Seeed Studio XIAO ESP32-C5 (RISC-V @ 240 MHz)
 */

#include "tflm.h"
#include "model_data.h"
#include "spectra_config.h"

#include <cmath>
#include <cstdio>
#include <cstring>

#ifdef ESP_PLATFORM
#include "esp_log.h"
#include "esp_cpu.h"
#include "esp_timer.h"
static const char* TAG = "TFLM";
#else
#include <cassert>
#define ESP_LOGI(tag, fmt, ...) printf("[%s] " fmt "\n", tag, ##__VA_ARGS__)
#define ESP_LOGW(tag, fmt, ...) printf("[WARN][%s] " fmt "\n", tag, ##__VA_ARGS__)
#define ESP_LOGE(tag, fmt, ...) printf("[ERR][%s] " fmt "\n", tag, ##__VA_ARGS__)
static const char* TAG = "TFLM";
#endif

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

/*
 * Tensor arena — strictly placed in internal SRAM BSS segment (16-byte aligned).
 * NEVER allocated in external PSRAM (which causes high cache miss latency on every MAC).
 * 80 KB is the provisional size; recording allocator measures exact used bytes.
 */
static uint8_t s_tensor_arena[SPECTRA_TENSOR_ARENA_SIZE]
    __attribute__((aligned(16), section(".bss")));

// TFLM engine static handles
static const tflite::Model* s_model = nullptr;
static tflite::MicroInterpreter* s_interpreter = nullptr;
static TfLiteTensor* s_input_tensor = nullptr;
static TfLiteTensor* s_output_tensor = nullptr;

static bool s_initialized = false;
static size_t s_arena_used_bytes = 0;
static bool s_is_minimal_resolver = true;

// Static op resolvers to avoid heap fragmentation
static tflite::MicroMutableOpResolver<5> s_minimal_resolver;
static tflite::AllOpsResolver s_all_ops_resolver;

/**
 * Helper to assert and log tensor dimension mismatches.
 */
static bool CheckDimension(const char* name, int dim_idx, int actual, int expected) {
    if (actual != expected) {
        ESP_LOGE(TAG, "%s dim[%d] mismatch: expected %d, got %d", name, dim_idx, expected, actual);
        return false;
    }
    return true;
}

bool tflm_init(void) {
    if (s_initialized) {
        return true;
    }

    ESP_LOGI(TAG, "========================================");
    ESP_LOGI(TAG, "  Spectra KWS — Phase 3: TFLM Engine");
    ESP_LOGI(TAG, "  Target: XIAO ESP32-C5 (RISC-V @ 240 MHz)");
    ESP_LOGI(TAG, "========================================");

    // ── 1. Model Header & Schema Verification ───────────────────────────
    if (spectra_model == nullptr || spectra_model_len < 16) {
        ESP_LOGE(TAG, "FATAL: Model buffer is null or truncated (%u bytes)", (unsigned)spectra_model_len);
        return false;
    }

    // Verify 'TFL3' magic at offset 4
    if (memcmp(&spectra_model[4], "TFL3", 4) != 0) {
        char magic[5] = {0};
        memcpy(magic, &spectra_model[4], 4);
        ESP_LOGE(TAG, "FATAL: Invalid TFLite magic identifier: expected 'TFL3', got '%s'", magic);
        return false;
    }

    s_model = tflite::GetModel(spectra_model);
    if (s_model == nullptr) {
        ESP_LOGE(TAG, "FATAL: tflite::GetModel() failed");
        return false;
    }

    if (s_model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "FATAL: Model schema version %lu != expected %d",
                 s_model->version(), TFLITE_SCHEMA_VERSION);
        return false;
    }

    ESP_LOGI(TAG, "Model verified: %u bytes, magic 'TFL3', schema v%lu",
             (unsigned)spectra_model_len, s_model->version());

    // ── 2. Op Resolver Configuration ────────────────────────────────────
    tflite::MicroOpResolver* active_resolver = nullptr;

#if defined(CONFIG_SPECTRA_USE_ALLOPS) && CONFIG_SPECTRA_USE_ALLOPS
    ESP_LOGW(TAG, "Op Resolver: AllOpsResolver enabled via Kconfig (DEBUG MODE)");
    s_is_minimal_resolver = false;
    active_resolver = &s_all_ops_resolver;
#else
    ESP_LOGI(TAG, "Op Resolver: MicroMutableOpResolver<5> (Minimal Production)");
    s_minimal_resolver.AddConv2D();
    s_minimal_resolver.AddDepthwiseConv2D();
    s_minimal_resolver.AddMean();
    s_minimal_resolver.AddFullyConnected();
    s_minimal_resolver.AddSoftmax();
    s_is_minimal_resolver = true;
    active_resolver = &s_minimal_resolver;
#endif

    // ── 3. Create Interpreter & Allocate Tensors ────────────────────────
    static tflite::MicroInterpreter static_interpreter(
        s_model, *active_resolver, s_tensor_arena, SPECTRA_TENSOR_ARENA_SIZE
    );
    s_interpreter = &static_interpreter;

    TfLiteStatus alloc_status = s_interpreter->AllocateTensors();
    if (alloc_status != kTfLiteOk) {
        ESP_LOGE(TAG, "FATAL: AllocateTensors() failed (status=%d). Check arena size / missing ops!", alloc_status);
        return false;
    }

    s_arena_used_bytes = s_interpreter->arena_used_bytes();
    size_t recommended_arena = s_arena_used_bytes + (s_arena_used_bytes / 10) + 2048;

    ESP_LOGI(TAG, "Tensor Arena: %u / %u bytes used (%.1f%%) [Internal SRAM BSS]",
             (unsigned)s_arena_used_bytes, (unsigned)SPECTRA_TENSOR_ARENA_SIZE,
             100.0f * (float)s_arena_used_bytes / (float)SPECTRA_TENSOR_ARENA_SIZE);
    ESP_LOGI(TAG, "Arena Sizing: Exact used=%u B, Recommended budget (used+10%%+2KB)=%u B (~%u KB)",
             (unsigned)s_arena_used_bytes, (unsigned)recommended_arena,
             (unsigned)((recommended_arena + 1023) / 1024));

    // ── 4. Verify Input Tensor Metadata ─────────────────────────────────
    s_input_tensor = s_interpreter->input(0);
    if (s_input_tensor == nullptr) {
        ESP_LOGE(TAG, "FATAL: Input tensor (index 0) is null");
        return false;
    }

    bool contract_ok = true;
    contract_ok &= CheckDimension("Input", 0, s_input_tensor->dims->data[0], 1);
    contract_ok &= CheckDimension("Input", 1, s_input_tensor->dims->data[1], SPECTRA_INPUT_HEIGHT);
    contract_ok &= CheckDimension("Input", 2, s_input_tensor->dims->data[2], SPECTRA_INPUT_WIDTH);
    contract_ok &= CheckDimension("Input", 3, s_input_tensor->dims->data[3], SPECTRA_INPUT_CHANNELS);

    if (s_input_tensor->type != kTfLiteInt8) {
        ESP_LOGE(TAG, "Input dtype mismatch: expected INT8, got %s", TfLiteTypeGetName(s_input_tensor->type));
        contract_ok = false;
    }

    float in_scale_err = fabsf(s_input_tensor->params.scale - SPECTRA_INPUT_SCALE);
    if (in_scale_err > 1e-4f) {
        ESP_LOGE(TAG, "Input scale mismatch: expected %.6f, got %.6f",
                 SPECTRA_INPUT_SCALE, s_input_tensor->params.scale);
        contract_ok = false;
    }

    if (s_input_tensor->params.zero_point != SPECTRA_INPUT_ZERO_POINT) {
        ESP_LOGE(TAG, "Input zp mismatch: expected %d, got %d",
                 SPECTRA_INPUT_ZERO_POINT, s_input_tensor->params.zero_point);
        contract_ok = false;
    }

    // ── 5. Verify Output Tensor Metadata ────────────────────────────────
    s_output_tensor = s_interpreter->output(0);
    if (s_output_tensor == nullptr) {
        ESP_LOGE(TAG, "FATAL: Output tensor (index 27) is null");
        return false;
    }

    contract_ok &= CheckDimension("Output", 0, s_output_tensor->dims->data[0], 1);
    contract_ok &= CheckDimension("Output", 1, s_output_tensor->dims->data[1], SPECTRA_OUTPUT_CLASSES);

    if (s_output_tensor->type != kTfLiteInt8) {
        ESP_LOGE(TAG, "Output dtype mismatch: expected INT8, got %s", TfLiteTypeGetName(s_output_tensor->type));
        contract_ok = false;
    }

    float out_scale_err = fabsf(s_output_tensor->params.scale - SPECTRA_OUTPUT_SCALE);
    if (out_scale_err > 1e-6f) {
        ESP_LOGE(TAG, "Output scale mismatch: expected %.6f, got %.6f",
                 SPECTRA_OUTPUT_SCALE, s_output_tensor->params.scale);
        contract_ok = false;
    }

    if (s_output_tensor->params.zero_point != SPECTRA_OUTPUT_ZERO_POINT) {
        ESP_LOGE(TAG, "Output zp mismatch: expected %d, got %d",
                 SPECTRA_OUTPUT_ZERO_POINT, s_output_tensor->params.zero_point);
        contract_ok = false;
    }

    if (!contract_ok) {
        ESP_LOGE(TAG, "FATAL: Contract assertions failed. Halting TFLM initialization.");
        return false;
    }

    ESP_LOGI(TAG, "Input:  shape=[1,%d,%d,%d] INT8 | scale=%.6f zp=%d (1280 bytes)",
             SPECTRA_INPUT_HEIGHT, SPECTRA_INPUT_WIDTH, SPECTRA_INPUT_CHANNELS,
             s_input_tensor->params.scale, s_input_tensor->params.zero_point);
    ESP_LOGI(TAG, "Output: shape=[1,%d] INT8 | scale=%.6f zp=%d (2 classes)",
             SPECTRA_OUTPUT_CLASSES, s_output_tensor->params.scale, s_output_tensor->params.zero_point);

    s_initialized = true;
    return true;
}

bool tflm_is_initialized(void) {
    return s_initialized;
}

bool tflm_feed_input(const int8_t* mfcc_features_1280) {
    if (!s_initialized || s_input_tensor == nullptr || mfcc_features_1280 == nullptr) {
        return false;
    }

    // Direct contiguous memcpy: [1, 40, 32, 1] NHWC matching P2's mel-major [m*32 + t] layout
    memcpy(s_input_tensor->data.int8, mfcc_features_1280, 1280);
    return true;
}

bool tflm_invoke(tflm_result_t* out_result) {
    if (!s_initialized || s_interpreter == nullptr || s_output_tensor == nullptr) {
        return false;
    }

    // Timing measurement around Invoke()
#ifdef ESP_PLATFORM
    uint32_t cycle_start = esp_cpu_get_cycle_count();
#endif

    TfLiteStatus invoke_status = s_interpreter->Invoke();

#ifdef ESP_PLATFORM
    uint32_t cycle_end = esp_cpu_get_cycle_count();
    uint32_t cycles = cycle_end - cycle_start;
#else
    uint32_t cycles = 0;
#endif

    if (invoke_status != kTfLiteOk) {
        ESP_LOGE(TAG, "Invoke() failed with status=%d", invoke_status);
        return false;
    }

    // Dequantize outputs: p[i] = (q[i] - zp) * scale
    // With scale = 1/256 and zp = -128: p[i] = (q[i] + 128) / 256.0f
    int8_t raw_0 = s_output_tensor->data.int8[0];
    int8_t raw_1 = s_output_tensor->data.int8[1];

    float p_0 = (float)(raw_0 + 128) / 256.0f;
    float p_1 = (float)(raw_1 + 128) / 256.0f;

    // Health Invariants Check
    float sum_p = p_0 + p_1;
    if (sum_p < 0.99f || sum_p > 1.01f || std::isnan(p_0) || std::isnan(p_1) || std::isinf(p_0) || std::isinf(p_1)) {
        ESP_LOGE(TAG, "HEALTH ASSERT FAIL: Invalid probabilities p0=%.4f, p1=%.4f (sum=%.4f)", p_0, p_1, sum_p);
        return false;
    }

    if (out_result != nullptr) {
        out_result->p_neg = p_0;
        out_result->p_pos = p_1;
        out_result->raw_neg = raw_0;
        out_result->raw_pos = raw_1;
        out_result->argmax = (p_1 > p_0) ? 1 : 0;
        out_result->cycles = cycles;
        out_result->inference_ms = (float)cycles / 240000.0f;  // 240 MHz clock
    }

    return true;
}

bool tflm_run_zeros_smoke(tflm_result_t* out_result) {
    if (!s_initialized) {
        if (!tflm_init()) return false;
    }

    // Feed 1,280 all-zeros
    memset(s_input_tensor->data.int8, 0, 1280);

    tflm_result_t res;
    if (!tflm_invoke(&res)) {
        ESP_LOGE(TAG, "Smoke test failed during invoke");
        return false;
    }

    // Assert zeros-smoke anchor: neg ≈ 0.9961 ± 0.01, pos ≈ 0.0
    float neg_err = fabsf(res.p_neg - 0.9961f);
    bool pass = (neg_err <= 0.01f) && (res.argmax == 0);

    ESP_LOGI(TAG, "Zeros-Smoke Test: neg=%.4f, pos=%.4f (raw: %d, %d) | %s (err=%.4f <= 0.01)",
             res.p_neg, res.p_pos, res.raw_neg, res.raw_pos,
             pass ? "PASS" : "FAIL", neg_err);

    if (out_result != nullptr) {
        *out_result = res;
    }

    return pass;
}

void tflm_get_info(tflm_info_t* out_info) {
    if (out_info == nullptr) return;
    out_info->arena_capacity = SPECTRA_TENSOR_ARENA_SIZE;
    out_info->arena_used = s_arena_used_bytes;
    out_info->model_bytes = spectra_model_len;
    out_info->is_minimal_resolver = s_is_minimal_resolver;
}
