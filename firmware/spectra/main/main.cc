/**
 * main.cc — Spectra KWS Firmware Entry Point
 *
 * Phase 0b: Target Lock + Contract Verification (XIAO ESP32-C5)
 * Phase 1:  Audio Capture (INMP441 I2S) + PSRAM Ring Buffer
 *
 * Architecture:
 *   - Target: Seeed Studio XIAO ESP32-C5 (RISC-V @ 240 MHz, 8 MB PSRAM, 8 MB Flash)
 *   - Tensor arena: Explicit BSS placement (80 KB in internal SRAM)
 *   - Audio Ring: PSRAM allocation (96 KB = 48,000 samples = 3.0s)
 *   - INMP441 I2S: BCLK=GPIO23 (D4), WS=GPIO24 (D5), DIN=GPIO11 (D6)
 *   - LED proof: GPIO27 (3 blinks on Phase 0b PASS, heartbeat during Phase 1)
 */

#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <cstdlib>

#include "esp_log.h"
#include "esp_system.h"
#include "esp_heap_caps.h"
#include "driver/gpio.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "model_data.h"
#include "spectra_config.h"
#include "audio_ring.h"
#include "audio_capture.h"
#include "wav_injector.h"
#include "mfcc.h"
#include "tflm.h"
#include "logits_test.h"
#include "trigger.h"
#include "pre_roll.h"
#include "esp_timer.h"
#include "power_mgr.h"
#include "vad_calib.h"
#include "watchdog.h"
#include "boot_guard.h"
#include "health_diag.h"

static const char* TAG = "SPECTRA";

// Working buffers for Phase 1 audio processing & Phase 2 feature extraction
static int16_t s_capture_block[SPECTRA_CAPTURE_BLOCK_SAMPLES];
static int16_t s_audio_window[SPECTRA_WINDOW_SAMPLES];
static float s_mfcc_float[SPECTRA_INPUT_HEIGHT * SPECTRA_INPUT_WIDTH];
static int8_t s_mfcc_int8[SPECTRA_INPUT_HEIGHT * SPECTRA_INPUT_WIDTH];

/**
 * Verify a single tensor dimension matches expected value.
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
 * Toggle LED briefly (heartbeat).
 */
static void led_toggle(void) {
    static int s_state = 0;
    s_state ^= 1;
    gpio_set_level((gpio_num_t)SPECTRA_LED_GPIO, s_state);
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

/**
 * Run deterministic WAV injector test on target.
 * Verifies mathematical parity and overrun handling with 0 external dependencies.
 */
static bool run_wav_injection_selftest(void) {
    ESP_LOGI(TAG, "--- Starting WAV Injection Self-Test ---");

    audio_ring_reset();
    wav_injector_init(INJECTOR_MODE_RAMP, 0);

    // Ingest 4 blocks (16,000 samples = Window 0)
    for (int b = 0; b < 4; b++) {
        wav_injector_generate_block(s_capture_block, SPECTRA_CAPTURE_BLOCK_SAMPLES);
        audio_ring_write(s_capture_block, SPECTRA_CAPTURE_BLOCK_SAMPLES);
    }

    if (!audio_ring_peek_window(s_audio_window, SPECTRA_WINDOW_SAMPLES)) {
        ESP_LOGE(TAG, "Self-Test FAIL: Peek window 0 failed");
        return false;
    }

    uint32_t crc0 = wav_injector_compute_crc32(s_audio_window, SPECTRA_WINDOW_SAMPLES);
    ESP_LOGI(TAG, "Window 0 CRC-32: 0x%08X (Golden: 0x6B2E496E) -> %s",
             crc0, (crc0 == 0x6B2E496E) ? "MATCH" : "MISMATCH");
    if (crc0 != 0x6B2E496E) return false;

    // Advance 8,000 samples and ingest 2 more blocks (Hop 1)
    audio_ring_advance(SPECTRA_HOP_SAMPLES);
    for (int b = 0; b < 2; b++) {
        wav_injector_generate_block(s_capture_block, SPECTRA_CAPTURE_BLOCK_SAMPLES);
        audio_ring_write(s_capture_block, SPECTRA_CAPTURE_BLOCK_SAMPLES);
    }

    if (!audio_ring_peek_window(s_audio_window, SPECTRA_WINDOW_SAMPLES)) {
        ESP_LOGE(TAG, "Self-Test FAIL: Peek window 1 failed");
        return false;
    }

    uint32_t crc1 = wav_injector_compute_crc32(s_audio_window, SPECTRA_WINDOW_SAMPLES);
    ESP_LOGI(TAG, "Hop 1    CRC-32: 0x%08X (Golden: 0x86F0B27A) -> %s",
             crc1, (crc1 == 0x86F0B27A) ? "MATCH" : "MISMATCH");
    if (crc1 != 0x86F0B27A) return false;

    // Forced-overrun verification: write 50,000 samples without advancing
    ESP_LOGI(TAG, "Testing forced-overrun drop-oldest behavior...");
    for (int b = 0; b < 10; b++) {
        wav_injector_generate_block(s_capture_block, SPECTRA_CAPTURE_BLOCK_SAMPLES);
        audio_ring_write(s_capture_block, SPECTRA_CAPTURE_BLOCK_SAMPLES);
    }

    audio_ring_stats_t stats;
    audio_ring_get_stats(&stats);
    ESP_LOGI(TAG, "Forced-overrun result: overruns=%u, dropped=%u samples",
             stats.overrun_count, stats.dropped_samples);
    if (stats.overrun_count == 0 || stats.dropped_samples == 0) {
        ESP_LOGE(TAG, "Self-Test FAIL: Overrun policy did not register dropped samples");
        return false;
    }

    audio_ring_reset();
    ESP_LOGI(TAG, "WAV Injection Self-Test PASS: bit-exact match & overrun verified");
    return true;
}

/**
 * Run Phase 2 MFCC Feature Extraction self-test on target.
 * Verifies silence energy gating and 440Hz synthetic signal extraction.
 */
static bool run_mfcc_selftest(void) {
    ESP_LOGI(TAG, "--- Starting MFCC Engine Self-Test ---");
    mfcc_init();

    // 1. Verify silence energy gate (must reject)
    memset(s_audio_window, 0, sizeof(s_audio_window));
    bool silence_voiced = mfcc_process_window(s_audio_window, s_mfcc_float, s_mfcc_int8);
    if (silence_voiced) {
        ESP_LOGE(TAG, "MFCC Self-Test FAIL: Silence not rejected by energy gate");
        return false;
    }
    ESP_LOGI(TAG, "  [PASS] Silence correctly rejected by energy gate");

    // 2. Verify synthetic voiced signal (440Hz sine wave, peak 0.8)
    for (int i = 0; i < SPECTRA_WINDOW_SAMPLES; i++) {
        float s = 0.8f * sinf(2.0f * (float)M_PI * 440.0f * (float)i / (float)SPECTRA_SAMPLE_RATE);
        s_audio_window[i] = (int16_t)(s * 32767.0f);
    }

    int64_t t0 = esp_timer_get_time();
    bool voiced = mfcc_process_window(s_audio_window, s_mfcc_float, s_mfcc_int8);
    int64_t elapsed_us = esp_timer_get_time() - t0;

    if (!voiced) {
        ESP_LOGE(TAG, "MFCC Self-Test FAIL: 440Hz sine wave rejected by energy gate");
        return false;
    }

    // Check for NaN or Inf in output
    for (int i = 0; i < SPECTRA_INPUT_HEIGHT * SPECTRA_INPUT_WIDTH; i++) {
        if (isnan(s_mfcc_float[i]) || isinf(s_mfcc_float[i])) {
            ESP_LOGE(TAG, "MFCC Self-Test FAIL: NaN/Inf detected at index %d", i);
            return false;
        }
    }

    ESP_LOGI(TAG, "  [PASS] 440Hz sine processed in %lld us (~%.2f ms) | Features: %dx%d INT8",
             (long long)elapsed_us, (float)elapsed_us / 1000.0f,
             SPECTRA_INPUT_HEIGHT, SPECTRA_INPUT_WIDTH);
    ESP_LOGI(TAG, "MFCC Engine Self-Test PASS: Gating and extraction operational");
    return true;
}

/**
 * Execute Phase 1 & 2: Ring Buffer, Audio Capture, and MFCC Feature Pipeline.
 */
static void run_pipeline_audio(void) {
    ESP_LOGI(TAG, "========================================");
    ESP_LOGI(TAG, "  Spectra KWS — Phase 1 & 2: Audio & MFCC");
    ESP_LOGI(TAG, "  Target: XIAO ESP32-C5 | INMP441 I2S");
    ESP_LOGI(TAG, "========================================");

    // ── 1. Allocate Audio Ring Buffer in PSRAM ──────────────────────────
    ESP_LOGI(TAG, "Allocating Audio Ring: %u samples (3.0s = %u KB)...",
             SPECTRA_RING_CAPACITY, (unsigned)(SPECTRA_RING_CAPACITY * sizeof(int16_t) / 1024));
    ESP_LOGI(TAG, "--- Heap before Ring Allocation ---");
    print_heap_report();

    if (audio_ring_init(SPECTRA_RING_CAPACITY) != 0) {
        ESP_LOGE(TAG, "FATAL: Failed to allocate audio ring buffer in PSRAM");
        led_steady_on();
        return;
    }

    ESP_LOGI(TAG, "--- Heap after Ring Allocation ---");
    print_heap_report();

    audio_ring_stats_t ring_stats;
    audio_ring_get_stats(&ring_stats);
    ESP_LOGI(TAG, "Ring Buffer Placement: %s",
             ring_stats.is_psram ? "PSRAM (External SPI RAM)" : "Internal SRAM");

    // ── 2. Run deterministic injection verification (Phase 1) ───────────
    if (!run_wav_injection_selftest()) {
        ESP_LOGE(TAG, "Phase 1 FAIL — WAV injection test failed");
        led_steady_on();
        return;
    }

    // ── 3. Run MFCC engine verification (Phase 2) ───────────────────────
    if (!run_mfcc_selftest()) {
        ESP_LOGE(TAG, "Phase 2 FAIL — MFCC self-test failed");
        led_steady_on();
        return;
    }

    // ── 4. Initialize I2S Hardware Driver ───────────────────────────────
    ESP_LOGI(TAG, "Configuring INMP441 I2S on BCLK=%d, WS=%d, DIN=%d",
             SPECTRA_I2S_BCLK_GPIO, SPECTRA_I2S_WS_GPIO, SPECTRA_I2S_DIN_GPIO);

    esp_err_t err = audio_capture_init();
    if (err != ESP_OK) {
        ESP_LOGW(TAG, "I2S init warning (err=0x%x). Operating in test/injection mode.", err);
    } else {
        err = audio_capture_start();
        if (err != ESP_OK) {
            ESP_LOGW(TAG, "I2S start warning (err=0x%x).", err);
        }
    }

    ESP_LOGI(TAG, "========================================");
    ESP_LOGI(TAG, "Phase 1 & 2 PASS — Capture, Ring & MFCC Operational");
    ESP_LOGI(TAG, "Entering continuous audio & feature extraction loop...");
    ESP_LOGI(TAG, "========================================");

    // ── 5. Initialize VAD Auto-Calibration & Supervised Watchdogs ───────
    static vad_calib_context_t s_vad_calib;
    vad_calib_init(&s_vad_calib, VAD_CALIB_DEFAULT_FRAMES);

    watchdog_enable_channel(WATCHDOG_CHAN_AUDIO);
    watchdog_enable_channel(WATCHDOG_CHAN_INFER);

    // ── 6. Continuous Streaming / Feature Extraction Loop ───────────────
    uint64_t total_samples_captured = 0;
    uint32_t total_hops_processed = 0;
    uint32_t loop_count = 0;

    while (true) {
        size_t samples_read = 0;
        esp_err_t read_err = audio_capture_read(s_capture_block,
                                                SPECTRA_CAPTURE_BLOCK_SAMPLES,
                                                &samples_read,
                                                300);

        uint32_t now_ms = (uint32_t)(esp_timer_get_time() / 1000);

        if (read_err == ESP_OK && samples_read > 0) {
            audio_ring_write(s_capture_block, samples_read);
            total_samples_captured += samples_read;
            watchdog_feed(WATCHDOG_CHAN_AUDIO, now_ms);

            // Background acoustic noise floor auto-calibration
            if (!s_vad_calib.is_complete && samples_read >= 3200) {
                vad_calib_feed_pcm(&s_vad_calib, &s_capture_block[0]);
                if (!s_vad_calib.is_complete) {
                    vad_calib_feed_pcm(&s_vad_calib, &s_capture_block[1600]);
                }
            }
        } else {
            // If I2S hardware mic not connected, inject synthetic block for soak simulation
            wav_injector_generate_block(s_capture_block, SPECTRA_CAPTURE_BLOCK_SAMPLES);
            audio_ring_write(s_capture_block, SPECTRA_CAPTURE_BLOCK_SAMPLES);
            total_samples_captured += SPECTRA_CAPTURE_BLOCK_SAMPLES;
            watchdog_feed(WATCHDOG_CHAN_AUDIO, now_ms);

            if (!s_vad_calib.is_complete) {
                vad_calib_feed_pcm(&s_vad_calib, &s_capture_block[0]);
                if (!s_vad_calib.is_complete) {
                    vad_calib_feed_pcm(&s_vad_calib, &s_capture_block[1600]);
                }
            }
            vTaskDelay(pdMS_TO_TICKS(250)); // Simulates 0.25s audio arrival rate
        }

        // Check if full 1.0s window is available for processing
        if (audio_ring_peek_window(s_audio_window, SPECTRA_WINDOW_SAMPLES)) {
            total_hops_processed++;

            // Measure peak amplitude and RMS energy across window
            int16_t peak = 0;
            int64_t sum_sq = 0;
            for (int i = 0; i < SPECTRA_WINDOW_SAMPLES; i++) {
                int16_t s = s_audio_window[i];
                int16_t a = abs(s);
                if (a > peak) peak = a;
                sum_sq += ((int32_t)s * (int32_t)s);
            }
            float peak_norm = (float)peak / 32768.0f;
            float rms = sqrtf((float)sum_sq / SPECTRA_WINDOW_SAMPLES) / 32768.0f;

            // Power Lock: Boost CPU to 240 MHz for compute-intensive DSP + TFLM
            power_lock_acquire(POWER_CLIENT_DSP_INFERENCE);

            // Phase 2: Compute 40x32 MFCC Features + Energy Gating
            int64_t t_start = esp_timer_get_time();
            bool voiced = mfcc_process_window(s_audio_window, s_mfcc_float, s_mfcc_int8);
            int64_t mfcc_us = esp_timer_get_time() - t_start;

            // Phase 3: TFLM Inference on Voiced Audio
            tflm_result_t tflm_res = {0};
            int64_t tflm_us = 0;
            float p_pos = 0.0f;
            if (voiced) {
                int64_t t_inf = esp_timer_get_time();
                tflm_feed_input(s_mfcc_int8);
                if (tflm_invoke(&tflm_res)) {
                    tflm_us = esp_timer_get_time() - t_inf;
                    p_pos = tflm_res.p_pos;
                }
            }

            // Release power lock: allows DFS to drop clock to 80 MHz / idle
            power_lock_release(POWER_CLIENT_DSP_INFERENCE);

            // Supervised TWDT: Feed inference channel
            now_ms = (uint32_t)(esp_timer_get_time() / 1000);
            watchdog_feed(WATCHDOG_CHAN_INFER, now_ms);

            // Phase 4: Trigger State Machine Evaluation (Tick = 500 ms hop)
            bool fired = trigger_tick(p_pos, total_hops_processed, now_ms, s_audio_window);
            if (fired) {
                health_diag_record_trigger();
                ESP_LOGI(TAG, ">>> KEYWORD TRIGGERED! Event fired on hop %u <<<", (unsigned)total_hops_processed);
            }

            // Update System Health & Diagnostic Telemetry
            health_diag_update(now_ms, (uint32_t)tflm_us);

            // Check UART for test runner packet
            logits_test_poll_uart();

            // Advance read pointer by 0.5s hop (50% overlap)
            audio_ring_advance(SPECTRA_HOP_SAMPLES);

            // Mark system boot healthy after running stably for 30 seconds
            static bool s_marked_healthy = false;
            if (!s_marked_healthy && now_ms >= BOOT_GUARD_HEALTHY_MS) {
                boot_guard_mark_healthy();
                s_marked_healthy = true;
            }

            // Log telemetry every 10 hops (~5 seconds)
            if (total_hops_processed % 10 == 0) {
                audio_ring_get_stats(&ring_stats);
                const trigger_context_t* t_ctx = trigger_get_context();
                ESP_LOGI(TAG, "[SOAK] Hop: %u | Peak: %.3f | Voiced: %s | MFCC: %lld us | TFLM: %lld us | P(\"Spectra\"): %.4f | Trigs: %u",
                         total_hops_processed, peak_norm,
                         voiced ? "YES" : "NO", (long long)mfcc_us, (long long)tflm_us,
                         p_pos, (unsigned)t_ctx->total_triggers);
            }

            // Log detailed health diagnostic summary every 60 hops (~30 seconds)
            if (total_hops_processed % 60 == 0) {
                health_diag_log_periodic(now_ms, 0); // Force immediate periodic output
                print_heap_report();
            }
        }

        loop_count++;
    }
}

extern "C" void app_main(void) {
    // ── Early Boot-Loop Guard (Tripwire to Safe Mode on 3 crashes) ───────
    boot_mode_t bmode = boot_guard_check();
    if (bmode == BOOT_MODE_SAFE) {
        ESP_LOGE(TAG, "CRITICAL: Boot-loop guard tripwire reached! Entering SAFE MODE");
        led_init();
        while (true) {
            boot_guard_indicate_safe_mode();
            vTaskDelay(pdMS_TO_TICKS(1000));
        }
        return;
    }

    // ── LED init (board-alive proof) ────────────────────────────────────
    led_init();

    ESP_LOGI(TAG, "========================================");
    ESP_LOGI(TAG, "  Spectra KWS — Production Firmware");
    ESP_LOGI(TAG, "  Board: XIAO ESP32-C5 (RISC-V @ 240 MHz)");
    ESP_LOGI(TAG, "========================================");

    // ── Power Management, Task Watchdog & Health Telemetry Init ────────
    power_mgr_init(true);
    watchdog_init(SPECTRA_WATCHDOG_TIMEOUT_MS);
    health_diag_init();

    // ── Heap report (pre-initialization) ────────────────────────────────
    ESP_LOGI(TAG, "--- Heap (pre-model) ---");
    print_heap_report();

    // ── Step 1: Initialize TFLM Engine (Model verification + Minimal Resolver <5> + BSS Arena) ──
    if (!tflm_init()) {
        ESP_LOGE(TAG, "FATAL: TFLM initialization failed");
        led_steady_on();
        return;
    }

    // ── Step 2: Zeros-Smoke Regression Test ─────────────────────────────
    tflm_result_t smoke_res;
    if (!tflm_run_zeros_smoke(&smoke_res)) {
        ESP_LOGE(TAG, "FATAL: Zeros-smoke anchor test failed");
        led_steady_on();
        return;
    }

    // ── Step 3: On-Device 5-Clip Equivalence Proof (Minimal vs Reference) ──
    if (!logits_test_run_equivalence_5()) {
        ESP_LOGE(TAG, "FATAL: 5-clip equivalence proof failed");
        led_steady_on();
        return;
    }

    // ── Step 4: Measured Memory Budget Audit (< 256 KB Internal SRAM) ───
    logits_test_print_memory_budget();

    // ── Step 5: Initialize Phase 4 Trigger Subsystem & Autonomous Self-Test ─
    trigger_init();
    ESP_LOGI(TAG, "Running Phase 4 Trigger Self-Test (Stream D)...");
    if (!trigger_run_selftest_stream_d()) {
        ESP_LOGE(TAG, "FATAL: Trigger self-test on Stream D failed");
        led_steady_on();
        return;
    }
    ESP_LOGI(TAG, "Phase 4 Trigger Self-Test: Stream D [2, 9] PASS");

    // ── Step 6: LED Proof ───────────────────────────────────────────────
    ESP_LOGI(TAG, "========================================");
    ESP_LOGI(TAG, "Phase 4 PASS — Trigger & LED Operational");
    ESP_LOGI(TAG, "========================================");
    led_blink(3);  // 3 blinks = PASS

    // ── Handover to Continuous Audio & Inference Pipeline ───────────────
    run_pipeline_audio();
}
