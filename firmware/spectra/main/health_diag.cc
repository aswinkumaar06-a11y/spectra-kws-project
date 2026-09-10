/**
 * health_diag.cc — Spectra Production Health & Diagnostic Implementation
 */

#include "health_diag.h"
#include "audio_ring.h"
#include "power_mgr.h"
#include <cstdio>
#include <cstring>

#ifdef ESP_PLATFORM
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
static const char* TAG = "HEALTH_DIAG";
#else
#define ESP_LOGI(tag, fmt, ...) printf("[%s] " fmt "\n", tag, ##__VA_ARGS__)
static const char* TAG = "HEALTH_DIAG";
#endif

static health_stats_t s_stats;
static uint32_t s_start_time_ms = 0;
static uint32_t s_last_periodic_log_ms = 0;

void health_diag_init(void) {
    memset(&s_stats, 0, sizeof(s_stats));

#ifdef ESP_PLATFORM
    s_start_time_ms = (uint32_t)(esp_timer_get_time() / 1000);
    s_stats.min_free_sram = heap_caps_get_minimum_free_size(MALLOC_CAP_INTERNAL);
    s_stats.current_free_sram = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
    s_stats.min_free_psram = heap_caps_get_minimum_free_size(MALLOC_CAP_SPIRAM);
    s_stats.current_free_psram = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
#else
    s_start_time_ms = 0;
    s_stats.min_free_sram = 172000;
    s_stats.current_free_sram = 172000;
    s_stats.min_free_psram = 8000000;
    s_stats.current_free_psram = 8000000;
#endif

    s_stats.cpu_freq_mhz = power_mgr_get_freq_mhz();
    ESP_LOGI(TAG, "Health diagnostics initialized. Initial SRAM Free: %u B, PSRAM Free: %u B",
             (unsigned)s_stats.current_free_sram, (unsigned)s_stats.current_free_psram);
}

void health_diag_update(uint32_t now_ms, uint32_t last_inference_us) {
    s_stats.uptime_s = (now_ms >= s_start_time_ms) ? ((now_ms - s_start_time_ms) / 1000) : 0;
    s_stats.total_inferences++;

    // Exponential moving average for inference latency
    if (s_stats.avg_inference_us == 0) {
        s_stats.avg_inference_us = last_inference_us;
    } else {
        s_stats.avg_inference_us = (uint32_t)((s_stats.avg_inference_us * 7 + last_inference_us) / 8);
    }

#ifdef ESP_PLATFORM
    size_t sram = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
    size_t psram = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    s_stats.current_free_sram = sram;
    s_stats.current_free_psram = psram;
    if (sram < s_stats.min_free_sram) s_stats.min_free_sram = sram;
    if (psram < s_stats.min_free_psram) s_stats.min_free_psram = psram;
#endif

    s_stats.ring_overruns = audio_ring_overruns();
    s_stats.cpu_freq_mhz = power_mgr_get_freq_mhz();
}

void health_diag_record_trigger(void) {
    s_stats.total_triggers++;
}

void health_diag_format_summary(char* out_str, size_t max_len) {
    if (out_str == NULL || max_len == 0) return;

    snprintf(out_str, max_len,
             "HEALTH uptime=%us sram_min=%uB sram_cur=%uB psram_min=%uB "
             "inf_count=%u inf_avg_us=%u trigs=%u overruns=%u freq=%uMHz",
             (unsigned)s_stats.uptime_s,
             (unsigned)s_stats.min_free_sram,
             (unsigned)s_stats.current_free_sram,
             (unsigned)s_stats.min_free_psram,
             (unsigned)s_stats.total_inferences,
             (unsigned)s_stats.avg_inference_us,
             (unsigned)s_stats.total_triggers,
             (unsigned)s_stats.ring_overruns,
             (unsigned)s_stats.cpu_freq_mhz);
}

void health_diag_log_periodic(uint32_t now_ms, uint32_t interval_ms) {
    if (now_ms - s_last_periodic_log_ms >= interval_ms) {
        char buf[256];
        health_diag_format_summary(buf, sizeof(buf));
        ESP_LOGI(TAG, "%s", buf);
        s_last_periodic_log_ms = now_ms;
    }
}

void health_diag_get_stats(health_stats_t* stats) {
    if (stats == NULL) return;
    *stats = s_stats;
}
