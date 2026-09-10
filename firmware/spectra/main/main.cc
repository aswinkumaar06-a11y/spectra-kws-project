/**
 * main.cc — Spectra KWS Phase 0b: Target Lock + Contract Verification
 *
 * Target: XIAO ESP32-C5 (RISC-V, ESP-IDF v5.5.2+)
 *
 * This is the firmware entry point. It initializes the TFLM interpreter,
 * loads the INT8 model, prints I/O tensor metadata + heap stats to UART,
 * and blinks the onboard LED as board-alive proof.
 *
 * Phase 0b exit criteria:
 *   - Model loads without error
 *   - Input tensor:  shape=[1,40,32,1], dtype=INT8, scale=3.996214, zp=60
 *   - Output tensor: shape=[1,2],       dtype=INT8, scale=0.003906, zp=-128
 *   - Tensor arena usage printed (BSS placement, not stack)
 *   - Heap report: internal free + PSRAM free
 *   - LED blinks 3× on PASS, steady on FAIL
 *   - "Phase 0b PASS" printed
 */

#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstring>

#include "esp_log.h"
#include "esp_system.h"
#include "esp_heap_caps.h"
#include "driver/gpio.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model_data.h"
#include "spectra_config.h"

static const char* TAG = "SPECTRA";

/*
 * Tensor arena — placed in BSS (file-scope static), NOT on the stack.
 * 80 KB is provisional; Phase 3 will finalize via TFLM recording allocator.
 *
 * Placement: internal SRAM BSS segment (ESP32-C5 has 384 KB internal).
 * 16-byte alignment for efficient RISC-V vector access.
 *
 * WARNING: Do NOT move this inside a function — that would put it on the
 * stack and cause an immediate stack overflow on first inference.
 */
static uint8_t tensor_arena[SPECTRA_TENSOR_ARENA_SIZE]
    __attribute__((aligned(16), section(".bss")));

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

/**
 * Initialize onboard LED GPIO for board-alive proof.
 * XIAO ESP32-C5 onboard LED is on GPIO27.
 */
static void led_init(void) {
    gpio_config_t io_conf = {};
    io_conf.pin_bit_mask = (1ULL << SPECTRA_LED_GPIO);
    io_conf.mode = GPIO_MODE_OUTPUT;
    io_conf.pull_up_en = GPIO_PULLUP_DISABLE;
    io_conf.pull_down_en = GPIO_PULLDOWN_DISABLE;
    io_conf.intr_type = GPIO_INTR_DISABLE;
    gpio_config(&io_conf);
    gpio_set_level((gpio_num_t)SPECTRA_LED_GPIO, 0);
}

/**
 * Blink LED n times (200ms on, 200ms off each).
 */
static void led_blink(int count) {
    for (int i = 0; i < count; i++) {
        gpio_set_level((gpio_num_t)SPECTRA_LED_GPIO, 1);
        vTaskDelay(pdMS_TO_TICKS(200));
        gpio_set_level((gpio_num_t)SPECTRA_LED_GPIO, 0);
        vTaskDelay(pdMS_TO_TICKS(200));
    }
}

/**
 * Set LED steady on (for FAIL indication).
 */
static void led_steady_on(void) {
    gpio_set_level((gpio_num_t)SPECTRA_LED_GPIO, 1);
}

/**
 * Print heap statistics: internal SRAM free + PSRAM free.
 */
static void print_heap_report(void) {
    size_t internal_free = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
    size_t internal_total = heap_caps_get_total_size(MALLOC_CAP_INTERNAL);
    size_t psram_free = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    size_t psram_total = heap_caps_get_total_size(MALLOC_CAP_SPIRAM);

    ESP_LOGI(TAG, "Heap internal: %u / %u bytes free (%.1f%%)",
             (unsigned)internal_free, (unsigned)internal_total,
             internal_total > 0 ? 100.0f * internal_free / internal_total : 0.0f);
    ESP_LOGI(TAG, "Heap PSRAM:    %u / %u bytes free (%.1f%%)",
             (unsigned)psram_free, (unsigned)psram_total,
             psram_total > 0 ? 100.0f * psram_free / psram_total : 0.0f);
}

extern "C" void app_main(void) {
    // ── LED init (board-alive proof) ────────────────────────────────────
    led_init();

    ESP_LOGI(TAG, "========================================");
    ESP_LOGI(TAG, "  Spectra KWS — Phase 0b: Target Lock");
    ESP_LOGI(TAG, "  Board: XIAO ESP32-C5");
    ESP_LOGI(TAG, "========================================");

    // ── Heap report (before model load) ─────────────────────────────────
    ESP_LOGI(TAG, "--- Heap (pre-model) ---");
    print_heap_report();

    // ── Step 1: Load model ──────────────────────────────────────────────
    const tflite::Model* model = tflite::GetModel(spectra_model);
    if (model == nullptr) {
        ESP_LOGE(TAG, "FATAL: Failed to load model from spectra_model[]");
        led_steady_on();
        return;
    }

    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "FATAL: Model schema version %lu != expected %d",
                 model->version(), TFLITE_SCHEMA_VERSION);
        led_steady_on();
        return;
    }

    ESP_LOGI(TAG, "Model loaded: %u bytes (schema v%lu)",
             spectra_model_len, model->version());

    // ── Step 2: Create interpreter (AllOpsResolver — P3 will minimize) ──
    tflite::AllOpsResolver resolver;

    tflite::MicroInterpreter interpreter(
        model, resolver, tensor_arena, SPECTRA_TENSOR_ARENA_SIZE);

    TfLiteStatus allocate_status = interpreter.AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        ESP_LOGE(TAG, "FATAL: AllocateTensors() failed (status=%d)", allocate_status);
        led_steady_on();
        return;
    }

    size_t arena_used = interpreter.arena_used_bytes();
    ESP_LOGI(TAG, "Tensor arena: %u / %u bytes used (%.1f%%) [BSS placement]",
             (unsigned)arena_used, SPECTRA_TENSOR_ARENA_SIZE,
             100.0f * arena_used / SPECTRA_TENSOR_ARENA_SIZE);

    // ── Heap report (after model load) ──────────────────────────────────
    ESP_LOGI(TAG, "--- Heap (post-model) ---");
    print_heap_report();

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

    // ── Result + LED proof ──────────────────────────────────────────────
    ESP_LOGI(TAG, "========================================");
    if (pass) {
        ESP_LOGI(TAG, "Phase 0b PASS — model contract verified on XIAO ESP32-C5");
        led_blink(3);  // 3 blinks = PASS (board-alive proof)
    } else {
        ESP_LOGE(TAG, "Phase 0b FAIL — contract mismatch detected");
        led_steady_on();  // Steady = FAIL
    }
    ESP_LOGI(TAG, "========================================");

    // Keep alive
    while (true) {
        vTaskDelay(pdMS_TO_TICKS(10000));
    }
}
