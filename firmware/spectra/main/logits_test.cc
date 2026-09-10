/**
 * logits_test.cc — Phase 3 Equivalence Proof & UART Parity Protocol
 *
 * Target: Seeed Studio XIAO ESP32-C5 (RISC-V @ 240 MHz)
 */

#include "logits_test.h"
#include "tflm.h"
#include "test_clips_5.h"
#include "spectra_config.h"

#include <cstdio>
#include <cstring>
#include <cmath>

#ifdef ESP_PLATFORM
#include "esp_log.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "driver/uart.h"
static const char* TAG = "LOGITS_TEST";
#else
#define ESP_LOGI(tag, fmt, ...) printf("[%s] " fmt "\n", tag, ##__VA_ARGS__)
#define ESP_LOGW(tag, fmt, ...) printf("[WARN][%s] " fmt "\n", tag, ##__VA_ARGS__)
#define ESP_LOGE(tag, fmt, ...) printf("[ERR][%s] " fmt "\n", tag, ##__VA_ARGS__)
static const char* TAG = "LOGITS_TEST";
#endif

bool logits_test_run_equivalence_5(void) {
    ESP_LOGI(TAG, "==================================================");
    ESP_LOGI(TAG, "  Phase 3: 5-Clip On-Device Equivalence Proof");
    ESP_LOGI(TAG, "  Testing Minimal Resolver <5> vs Ground Truth");
    ESP_LOGI(TAG, "==================================================");

    if (!tflm_is_initialized()) {
        if (!tflm_init()) {
            ESP_LOGE(TAG, "Failed to initialize TFLM engine");
            return false;
        }
    }

    uint32_t passed_count = 0;
    uint32_t total_cycles = 0;

    for (int i = 0; i < SPECTRA_NUM_TEST_CLIPS_5; i++) {
        const spectra_test_clip_t* clip = &kTestClips5[i];

        if (!tflm_feed_input(clip->features)) {
            ESP_LOGE(TAG, "Failed to feed clip %d (%s)", i, clip->filename);
            continue;
        }

        tflm_result_t res;
        if (!tflm_invoke(&res)) {
            ESP_LOGE(TAG, "Inference failed for clip %d (%s)", i, clip->filename);
            continue;
        }

        total_cycles += res.cycles;

        // Bit-identical raw integer comparison
        bool bit_identical = (res.raw_neg == clip->raw_neg) && (res.raw_pos == clip->raw_pos);
        bool argmax_match = (res.argmax == clip->argmax);
        bool pass = bit_identical && argmax_match;

        if (pass) {
            passed_count++;
        }

        ESP_LOGI(TAG, " [%d/5] %-30s | C: [%4d, %4d] (pos=%.4f) | Ref: [%4d, %4d] | %s | %u cyc (%.2f ms)",
                 i + 1, clip->filename,
                 res.raw_neg, res.raw_pos, res.p_pos,
                 clip->raw_neg, clip->raw_pos,
                 pass ? "BIT-IDENTICAL" : "MISMATCH",
                 (unsigned)res.cycles, res.inference_ms);
    }

    float avg_ms = (float)total_cycles / (SPECTRA_NUM_TEST_CLIPS_5 * 240000.0f);
    ESP_LOGI(TAG, "--------------------------------------------------");
    ESP_LOGI(TAG, "Equivalence Proof Result: %u / %d Bit-Identical", passed_count, SPECTRA_NUM_TEST_CLIPS_5);
    ESP_LOGI(TAG, "Average Inference Latency: %.2f ms @ 240 MHz", avg_ms);
    ESP_LOGI(TAG, "==================================================");

    return (passed_count == SPECTRA_NUM_TEST_CLIPS_5);
}

void logits_test_print_memory_budget(void) {
    ESP_LOGI(TAG, "==================================================");
    ESP_LOGI(TAG, "  Spectra KWS: Phase 3 Measured Memory Budget");
    ESP_LOGI(TAG, "  Discipline: Internal SRAM < 256 KB (PSRAM Ring)");
    ESP_LOGI(TAG, "==================================================");

    tflm_info_t info;
    tflm_get_info(&info);

#ifdef ESP_PLATFORM
    size_t internal_free = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
    size_t internal_total = heap_caps_get_total_size(MALLOC_CAP_INTERNAL);
    size_t internal_used = (internal_total > internal_free) ? (internal_total - internal_free) : 0;

    size_t psram_free = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    size_t psram_total = heap_caps_get_total_size(MALLOC_CAP_SPIRAM);
    size_t psram_used = (psram_total > psram_free) ? (psram_total - psram_free) : 0;
#else
    size_t internal_total = 384 * 1024;
    size_t internal_used = 184 * 1024;
    size_t internal_free = internal_total - internal_used;
    size_t psram_total = 8 * 1024 * 1024;
    size_t psram_used = 96 * 1024;
    size_t psram_free = psram_total - psram_used;
#endif

    // Static footprint accounting
    size_t arena_sz = info.arena_capacity;
    size_t arena_used = info.arena_used;
    size_t ring_sz = SPECTRA_RING_CAPACITY * sizeof(int16_t);  // 96,000 bytes
    size_t model_sz = info.model_bytes;                        // 39,552 bytes
    size_t dsp_tables_sz = (1024 * 4) + (512 * 8) + (1009 * 4 + 128 * 4) + (40 * 128 * 4); // ~33 KB

    ESP_LOGI(TAG, "| Component                     | Placement     | Size (Bytes) | Size (KB)  |");
    ESP_LOGI(TAG, "|-------------------------------|---------------|--------------|------------|");
    ESP_LOGI(TAG, "| TFLM Tensor Arena (BSS)       | Internal SRAM | %12u | %8.1f KB |", (unsigned)arena_sz, (float)arena_sz / 1024.0f);
    ESP_LOGI(TAG, "|   └─ Exact Arena Used         | Internal SRAM | %12u | %8.1f KB |", (unsigned)arena_used, (float)arena_used / 1024.0f);
    ESP_LOGI(TAG, "| MFCC Padded Audio + FFT (BSS) | Internal SRAM | %12u | %8.1f KB |", 17024 * 4 + 1024 * 8, (float)(17024 * 4 + 1024 * 8) / 1024.0f);
    ESP_LOGI(TAG, "| DMA Block Buffer              | Internal SRAM | %12u | %8.1f KB |", 4000 * 2, 8.0f);
    ESP_LOGI(TAG, "| Total Internal SRAM Allocated | Internal SRAM | %12u | %8.1f KB |", (unsigned)internal_used, (float)internal_used / 1024.0f);
    ESP_LOGI(TAG, "| Internal Free Headroom        | Internal SRAM | %12u | %8.1f KB |", (unsigned)internal_free, (float)internal_free / 1024.0f);
    ESP_LOGI(TAG, "|-------------------------------|---------------|--------------|------------|");
    ESP_LOGI(TAG, "| Audio Ring Buffer (3.0s)      | External PSRAM| %12u | %8.1f KB |", (unsigned)ring_sz, (float)ring_sz / 1024.0f);
    ESP_LOGI(TAG, "| PSRAM Free Remaining          | External PSRAM| %12u | %8.1f MB |", (unsigned)psram_free, (float)psram_free / (1024.0f * 1024.0f));
    ESP_LOGI(TAG, "|-------------------------------|---------------|--------------|------------|");
    ESP_LOGI(TAG, "| TFLite Model Data             | Flash (RO)    | %12u | %8.1f KB |", (unsigned)model_sz, (float)model_sz / 1024.0f);
    ESP_LOGI(TAG, "| DSP Precomputed Tables        | Flash (RO)    | %12u | %8.1f KB |", (unsigned)dsp_tables_sz, (float)dsp_tables_sz / 1024.0f);
    ESP_LOGI(TAG, "==================================================");
    ESP_LOGI(TAG, "Discipline Check: Internal SRAM Total %u KB <= 256 KB Target -> %s",
             (unsigned)(internal_used / 1024),
             (internal_used <= 256 * 1024) ? "COMPLIANT" : "EXCEEDED");
    ESP_LOGI(TAG, "==================================================");
}

bool logits_test_poll_uart(void) {
    // UART Packet Framing Protocol:
    // Request:  'SPLG' (4B) + uint16_t clip_idx (2B) + int8_t features[1280] (1280B)
    // Response: 'SPLR' (4B) + uint16_t clip_idx (2B) + float p_neg (4B) + float p_pos (4B)
    //           + int8_t raw_neg (1B) + int8_t raw_pos (1B) + uint32_t cycles (4B) + uint8_t status (1B)

#ifdef ESP_PLATFORM
    uint8_t header[4];
    int read_bytes = uart_read_bytes(UART_NUM_0, header, 4, 10 / portTICK_PERIOD_MS);
    if (read_bytes != 4) return false;

    if (memcmp(header, "SPLG", 4) != 0) {
        return false;
    }

    uint16_t clip_idx = 0;
    if (uart_read_bytes(UART_NUM_0, (uint8_t*)&clip_idx, 2, 50 / portTICK_PERIOD_MS) != 2) {
        return false;
    }

    int8_t features[1280];
    int remaining = 1280;
    uint8_t* ptr = (uint8_t*)features;
    while (remaining > 0) {
        int r = uart_read_bytes(UART_NUM_0, ptr, remaining, 100 / portTICK_PERIOD_MS);
        if (r <= 0) break;
        ptr += r;
        remaining -= r;
    }
    if (remaining > 0) return false;

    tflm_feed_input(features);
    tflm_result_t res;
    bool ok = tflm_invoke(&res);

    // Send response packet
    uint8_t resp[22];
    memcpy(&resp[0], "SPLR", 4);
    memcpy(&resp[4], &clip_idx, 2);
    memcpy(&resp[6], &res.p_neg, 4);
    memcpy(&resp[10], &res.p_pos, 4);
    resp[14] = (uint8_t)res.raw_neg;
    resp[15] = (uint8_t)res.raw_pos;
    memcpy(&resp[16], &res.cycles, 4);
    resp[20] = ok ? 0 : 1;
    resp[21] = '\n';

    uart_write_bytes(UART_NUM_0, (const char*)resp, 22);
    return true;
#else
    return false;
#endif
}
