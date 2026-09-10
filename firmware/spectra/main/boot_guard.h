/**
 * boot_guard.h — Spectra NVS Boot-Loop Guard & Safe Mode Protection
 *
 * Protects firmware against persistent reboot loops:
 *   1. Increments an 'unclean_crashes' counter in NVS on early boot
 *   2. Clears the counter once the system runs healthy for >= 30 seconds
 *   3. If 3 consecutive crashes occur without clearing, activates SAFE_MODE:
 *      - Bypasses Wi-Fi radio & socket streaming
 *      - Flashes LED in a distinct 5x rapid alert pattern
 *      - Dumps error diagnostics to UART for developer recovery
 */

#ifndef SPECTRA_BOOT_GUARD_H_
#define SPECTRA_BOOT_GUARD_H_

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define BOOT_GUARD_MAX_CRASHES  3       // Tripwire threshold
#define BOOT_GUARD_HEALTHY_MS   30000   // 30 seconds of runtime marks boot clean

typedef enum {
    BOOT_MODE_NORMAL = 0,
    BOOT_MODE_SAFE   = 1
} boot_mode_t;

typedef struct {
    uint32_t total_boots;
    uint32_t consecutive_crashes;
    boot_mode_t current_mode;
    bool     is_marked_healthy;
} boot_guard_stats_t;

/**
 * Initialize boot guard on startup.
 * Reads crash counter, determines boot mode (NORMAL or SAFE),
 * and increments crash count.
 */
boot_mode_t boot_guard_check(void);

/**
 * Mark current boot session as healthy.
 * Resets consecutive crash counter to 0 in NVS.
 */
void boot_guard_mark_healthy(void);

/**
 * Manual reset of crash counters (e.g. from UART command or self-test).
 */
void boot_guard_reset_counter(void);

/**
 * Visual LED indicator for Safe Mode (5 rapid pulses).
 */
void boot_guard_indicate_safe_mode(void);

/**
 * Get current boot guard statistics.
 */
void boot_guard_get_stats(boot_guard_stats_t* stats);

#ifdef __cplusplus
}
#endif

#endif  // SPECTRA_BOOT_GUARD_H_
