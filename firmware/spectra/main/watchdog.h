/**
 * watchdog.h — Spectra Task Watchdog Timer (TWDT) Hardening
 *
 * Provides task supervision across core firmware threads:
 *   - Audio Capture Task (DMA ingestion)
 *   - Main Inference Loop (Hop processing & TFLM)
 *   - Network Streamer Task (Socket I/O)
 *
 * If any supervised thread fails to feed its watchdog channel within
 * the timeout period (default: 3000 ms), the TWDT triggers a system reset.
 */

#ifndef SPECTRA_WATCHDOG_H_
#define SPECTRA_WATCHDOG_H_

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    WATCHDOG_CHAN_AUDIO   = 0,  // Fed on each DMA block (every 250 ms)
    WATCHDOG_CHAN_INFER   = 1,  // Fed on each inference hop (every 500 ms)
    WATCHDOG_CHAN_STREAM  = 2,  // Fed on socket loop activity
    WATCHDOG_NUM_CHANNELS = 3
} watchdog_channel_t;

typedef struct {
    uint32_t timeout_ms;
    uint32_t feeds_count[WATCHDOG_NUM_CHANNELS];
    uint32_t last_feed_ms[WATCHDOG_NUM_CHANNELS];
    bool     channel_enabled[WATCHDOG_NUM_CHANNELS];
} watchdog_stats_t;

/**
 * Initialize Task Watchdog Timer subsystem.
 *
 * @param timeout_ms Timeout threshold in milliseconds (default: 3000 ms)
 * @return 0 on success, negative on error.
 */
int watchdog_init(uint32_t timeout_ms);

/**
 * Enable monitoring on a specific channel.
 */
void watchdog_enable_channel(watchdog_channel_t ch);

/**
 * Feed watchdog channel. Must be called periodically before timeout expires.
 */
void watchdog_feed(watchdog_channel_t ch, uint32_t current_time_ms);

/**
 * Check if any enabled channel has expired (useful on host or periodic supervisor).
 * Returns true if all channels healthy, false if any channel timed out.
 */
bool watchdog_check_all(uint32_t current_time_ms, watchdog_channel_t* out_expired_ch);

/**
 * Get watchdog operational statistics.
 */
void watchdog_get_stats(watchdog_stats_t* stats);

#ifdef __cplusplus
}
#endif

#endif  // SPECTRA_WATCHDOG_H_
