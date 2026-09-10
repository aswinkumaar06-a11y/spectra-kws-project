/**
 * power_mgr.h — Spectra Power Optimization & Dynamic Frequency Scaling (DFS)
 *
 * Coordinates power modes on XIAO ESP32-C5:
 *   - High Performance (240 MHz): Active MFCC DSP, TFLM inference, Wi-Fi streaming
 *   - Balanced / Low Power (80 MHz): Idle listening, waiting for DMA audio blocks
 *   - Light Sleep: Automatic CPU clock gating between 500 ms inference hops
 *
 * Implements power client locks to ensure frequency scaling does not throttle
 * latency-sensitive DSP or streaming loops.
 */

#ifndef SPECTRA_POWER_MGR_H_
#define SPECTRA_POWER_MGR_H_

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Power Lock Client Flags
typedef enum {
    POWER_CLIENT_DSP_INFERENCE = (1 << 0),  // 240 MHz required for MFCC & TFLM
    POWER_CLIENT_STREAMING     = (1 << 1),  // 240 MHz / Active Wi-Fi
    POWER_CLIENT_CALIBRATION   = (1 << 2),  // On-boot acoustic calibration
    POWER_CLIENT_MANUAL_HOLD   = (1 << 3)   // Dev / Benchmarking hold
} power_client_t;

typedef struct {
    uint32_t max_freq_mhz;      // 240 MHz
    uint32_t min_freq_mhz;      // 80 MHz (or 40 MHz)
    bool     light_sleep_enable;// True: allow tickless light sleep between hops
    uint32_t active_locks;      // Bitmask of currently active power clients
} power_mgr_stats_t;

/**
 * Initialize power management subsystem.
 * On ESP32-C5: configures esp_pm_configure with DFS and dynamic sleep.
 * On Host: initializes mock power tracker.
 */
int power_mgr_init(bool light_sleep_enable);

/**
 * Acquire high-performance power lock (forces 240 MHz max clock).
 */
void power_lock_acquire(power_client_t client);

/**
 * Release power lock. When all locks are released, DFS drops frequency
 * to 80 MHz to preserve energy during idle windows.
 */
void power_lock_release(power_client_t client);

/**
 * Get current effective CPU frequency in MHz.
 */
uint32_t power_mgr_get_freq_mhz(void);

/**
 * Returns true if high-performance lock (240 MHz) is active.
 */
bool power_mgr_is_high_perf(void);

/**
 * Idle sleep helper: blocks for sleep_ms while allowing CPU clock gating / light sleep.
 */
void power_mgr_idle_sleep(uint32_t sleep_ms);

/**
 * Get power subsystem telemetry and lock status.
 */
void power_mgr_get_stats(power_mgr_stats_t* stats);

#ifdef __cplusplus
}
#endif

#endif  // SPECTRA_POWER_MGR_H_
