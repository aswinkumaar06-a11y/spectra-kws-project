/**
 * main.cc — Spectra KWS Phase 0: Model Load + Contract Verification
 *
 * This is the firmware entry point. It initializes the TFLM interpreter,
 * loads the INT8 model, and prints I/O tensor metadata to UART.
 *
 * Phase 0 exit criteria:
 *   - Model loads without error
 *   - Input tensor:  shape=[1,40,32,1], dtype=INT8, scale=3.996214, zp=60
 *   - Output tensor: shape=[1,2],       dtype=INT8, scale=0.003906, zp=-128
 *   - Tensor arena usage is printed
 *   - "Phase 0 PASS" is printed
 */

#include <cstdio>
#include <cstdint>
#include <cmath>

#include "esp_log.h"
#include "esp_system.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model_data.h"
#include "spectra_config.h"

static const char* TAG = "SPECTRA";

// Static tensor arena — no heap allocation for TFLM
static uint8_t tensor_arena[SPECTRA_TENSOR_ARENA_SIZE]
    __attribute__((aligned(16)));

/**
 * Register only the operators used by the Spectra DS-CNN model.
 * This keeps the binary small — full AllOpsResolver would add ~200 KB.
 *
 * Operators in the model (from TFLite FlatBuffer inspection):
 *   Conv2D, DepthwiseConv2D, Reshape, Softmax,
 *   Add (residual), Mean (GlobalAveragePooling2D),
 *   FullyConnected (Dense), Quantize, Dequantize
 */
static tflite::MicroMutableOpResolver<9>& GetOpResolver() {
    static tflite::MicroMutableOpResolver<9> resolver;
    static bool initialized = false;
    if (!initialized) {
        resolver.AddConv2D();
        resolver.AddDepthwiseConv2D();
        resolver.AddReshape();
        resolver.AddSoftmax();
        resolver.AddAdd();
        resolver.AddMean();
        resolver.AddFullyConnected();
        resolver.AddQuantize();
        resolver.AddDequantize();
        initialized = true;
    }
    return resolver;
}

/**
 * Verify a single tensor dimension matches expected value.
 * Returns true if match, false + logs error if mismatch.
 */
static bool CheckDim(const char* name, int dim_idx, int actual, int expected) {
    if (actual != expected) {
        ESP_LOGE(TAG, "%s dim[%d]: expected %d, got %d", name, dim_idx, expected, actual);
        return false;
    }
    return true;
}

extern "C" void app_main(void) {
    ESP_LOGI(TAG, "========================================");
    ESP_LOGI(TAG, "  Spectra KWS — Phase 0: Target Lock");
    ESP_LOGI(TAG, "========================================");

    // ── Step 1: Load model ──────────────────────────────────────────────
    const tflite::Model* model = tflite::GetModel(spectra_model);
    if (model == nullptr) {
        ESP_LOGE(TAG, "FATAL: Failed to load model from spectra_model[]");
        return;
    }

    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "FATAL: Model schema version %lu != expected %d",
                 model->version(), TFLITE_SCHEMA_VERSION);
        return;
    }

    ESP_LOGI(TAG, "Model loaded: %u bytes (schema v%lu)",
             spectra_model_len, model->version());

    // ── Step 2: Create interpreter ──────────────────────────────────────
    tflite::MicroMutableOpResolver<9>& resolver = GetOpResolver();

    tflite::MicroInterpreter interpreter(
        model, resolver, tensor_arena, SPECTRA_TENSOR_ARENA_SIZE);

    TfLiteStatus allocate_status = interpreter.AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        ESP_LOGE(TAG, "FATAL: AllocateTensors() failed (status=%d)", allocate_status);
        return;
    }

    size_t arena_used = interpreter.arena_used_bytes();
    ESP_LOGI(TAG, "Tensor arena: %u / %u bytes used (%.1f%%)",
             (unsigned)arena_used, SPECTRA_TENSOR_ARENA_SIZE,
             100.0f * arena_used / SPECTRA_TENSOR_ARENA_SIZE);

    // ── Step 3: Verify input tensor ─────────────────────────────────────
    TfLiteTensor* input = interpreter.input(0);
    bool pass = true;

    ESP_LOGI(TAG, "Input:  shape=[%d,%d,%d,%d] dtype=%s scale=%.6f zp=%d",
             input->dims->data[0], input->dims->data[1],
             input->dims->data[2], input->dims->data[3],
             TfLiteTypeGetName(input->type),
             input->params.scale, input->params.zero_point);

    pass &= CheckDim("Input", 0, input->dims->data[0], 1);
    pass &= CheckDim("Input", 1, input->dims->data[1], SPECTRA_INPUT_HEIGHT);
    pass &= CheckDim("Input", 2, input->dims->data[2], SPECTRA_INPUT_WIDTH);
    pass &= CheckDim("Input", 3, input->dims->data[3], SPECTRA_INPUT_CHANNELS);

    if (input->type != kTfLiteInt8) {
        ESP_LOGE(TAG, "Input dtype: expected INT8, got %s",
                 TfLiteTypeGetName(input->type));
        pass = false;
    }

    float scale_diff = fabsf(input->params.scale - SPECTRA_INPUT_SCALE);
    if (scale_diff > 1e-4f) {
        ESP_LOGE(TAG, "Input scale: expected %.6f, got %.6f",
                 SPECTRA_INPUT_SCALE, input->params.scale);
        pass = false;
    }

    if (input->params.zero_point != SPECTRA_INPUT_ZERO_POINT) {
        ESP_LOGE(TAG, "Input zp: expected %d, got %d",
                 SPECTRA_INPUT_ZERO_POINT, input->params.zero_point);
        pass = false;
    }

    // ── Step 4: Verify output tensor ────────────────────────────────────
    TfLiteTensor* output = interpreter.output(0);

    ESP_LOGI(TAG, "Output: shape=[%d,%d] dtype=%s scale=%.6f zp=%d",
             output->dims->data[0], output->dims->data[1],
             TfLiteTypeGetName(output->type),
             output->params.scale, output->params.zero_point);

    pass &= CheckDim("Output", 0, output->dims->data[0], 1);
    pass &= CheckDim("Output", 1, output->dims->data[1], SPECTRA_OUTPUT_CLASSES);

    if (output->type != kTfLiteInt8) {
        ESP_LOGE(TAG, "Output dtype: expected INT8, got %s",
                 TfLiteTypeGetName(output->type));
        pass = false;
    }

    float out_scale_diff = fabsf(output->params.scale - SPECTRA_OUTPUT_SCALE);
    if (out_scale_diff > 1e-6f) {
        ESP_LOGE(TAG, "Output scale: expected %.6f, got %.6f",
                 SPECTRA_OUTPUT_SCALE, output->params.scale);
        pass = false;
    }

    if (output->params.zero_point != SPECTRA_OUTPUT_ZERO_POINT) {
        ESP_LOGE(TAG, "Output zp: expected %d, got %d",
                 SPECTRA_OUTPUT_ZERO_POINT, output->params.zero_point);
        pass = false;
    }

    // ── Step 5: Run a zero-input inference (smoke test) ─────────────────
    memset(input->data.int8, 0, input->bytes);
    TfLiteStatus invoke_status = interpreter.Invoke();
    if (invoke_status != kTfLiteOk) {
        ESP_LOGE(TAG, "FATAL: Invoke() failed on zero input (status=%d)", invoke_status);
        pass = false;
    } else {
        int8_t out_neg = output->data.int8[0];
        int8_t out_pos = output->data.int8[1];
        float prob_neg = (out_neg - SPECTRA_OUTPUT_ZERO_POINT) * SPECTRA_OUTPUT_SCALE;
        float prob_pos = (out_pos - SPECTRA_OUTPUT_ZERO_POINT) * SPECTRA_OUTPUT_SCALE;
        ESP_LOGI(TAG, "Zero-input inference: neg=%.4f pos=%.4f (raw: %d, %d)",
                 prob_neg, prob_pos, out_neg, out_pos);
    }

    // ── Result ──────────────────────────────────────────────────────────
    ESP_LOGI(TAG, "========================================");
    if (pass) {
        ESP_LOGI(TAG, "Phase 0 PASS — model contract verified");
    } else {
        ESP_LOGE(TAG, "Phase 0 FAIL — contract mismatch detected");
    }
    ESP_LOGI(TAG, "========================================");

    // Keep alive (FreeRTOS requires app_main to not return or to idle)
    while (true) {
        vTaskDelay(pdMS_TO_TICKS(10000));
    }
}
