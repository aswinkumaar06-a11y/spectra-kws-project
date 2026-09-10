/**
 * health_diag.h — Spectra Production Health & Diagnostic Telemetry
 *
 * Collects runtime health watermarks and performance metrics:
 *   - Internal SRAM & PSRAM free memory watermarks
 *   - Cumulative inference count & moving average latency
 *   - Ring buffer drop/overrun occurrences
 *   - Watchdog heartbeat status
 *   - Formats compact diagnostic telemetry lines for UART logging
 */

#ifndef SPECTRA_HEALTH_DIAG_H_
#define SPECTRA_HEALTH_DIAG_H_

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint32_t uptime_s;
    size_t   min_free_sram;
    size_t   current_free_sram;
    size_t   min_free_psram;
    size_t   current_free_psram;
    uint32_t total_inferences;
    uint32_t avg_inference_us;
    uint32_t total_triggers;
    uint32_t ring_overruns;
    uint32_t cpu_freq_mhz;
} health_stats_t;

/**
 * Initialize health diagnostic subsystem and capture memory baselines.
 */
void health_diag_init(void);

/**
 * Update health metrics (called after each inference hop).
 *
 * @param now_ms Current timestamp in milliseconds
 * @param last_inference_us Latency of the most recent model invoke in microseconds
 */
void health_diag_update(uint32_t now_ms, uint32_t last_inference_us);

/**
 * Record a keyword detection trigger event.
 */
void health_diag_record_trigger(void);

/**
 * Format a compact key-value diagnostic summary string.
 */
void health_diag_format_summary(char* out_str, size_t max_len);

/**
 * Periodic UART log helper (e.g. logs every 60 seconds).
 */
void health_diag_log_periodic(uint32_t now_ms, uint32_t interval_ms);

/**
 * Get snapshot of current health statistics.
 */
void health_diag_get_stats(health_stats_t* stats);

#ifdef __cplusplus
}
#endif

#endif  // SPECTRA_HEALTH_DIAG_H_
