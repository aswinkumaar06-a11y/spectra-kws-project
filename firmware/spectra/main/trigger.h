/**
 * trigger.h - Spectra KWS Confidence Trigger & Action State Machine
 *
 * Implements deterministic integer-tick trigger logic:
 *   - Tick duration = 1 hop (500 ms)
 *   - Threshold: P(pos) > 0.50 (strictly greater)
 *   - N_FIRE = 3 consecutive ticks
 *   - COOLDOWN = 4 ticks (2.0s suppression)
 *
 * States:
 *   LISTEN    -- p > 0.5 --> CANDIDATE (count=1)
 *   CANDIDATE -- p > 0.5 --> count++; count==3 --> TRIGGERED (fire!)
 *             -- p <= 0.5 --> LISTEN (count reset: expire path)
 *   TRIGGERED -- fire actions --> COOLDOWN (ticks_left=4)
 *   COOLDOWN  -- ticks_left--; 0 --> LISTEN fresh (p ignored)
 *
 * LED Semantics:
 *   ON from the fire tick through cooldown end; OFF otherwise.
 */

#ifndef TRIGGER_H_
#define TRIGGER_H_

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include "spectra_config.h"
#include "pre_roll.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    TRIGGER_STATE_LISTEN = 0,
    TRIGGER_STATE_CANDIDATE,
    TRIGGER_STATE_TRIGGERED,
    TRIGGER_STATE_COOLDOWN,
} trigger_state_t;

typedef struct {
    trigger_state_t state;
    uint32_t candidate_count;      // Consecutive ticks above threshold (0..N_FIRE)
    uint32_t cooldown_ticks_left;  // Ticks remaining in cooldown (COOLDOWN_TICKS..0)
    uint32_t total_triggers;       // Cumulative firing count
    uint32_t boot_count;          // NVS boot counter
    uint32_t last_trigger_ts_ms;   // Timestamp of last fire
    bool led_state;                // Current LED output level (true=ON, false=OFF)
    float p_history[3];            // Rolling buffer of the last 3 probabilities
    char last_log_line[128];       // Normative UART string of last trigger event
} trigger_context_t;

// Callback signatures for hardware / mock abstraction
typedef void (*trigger_led_cb_t)(bool on);
typedef void (*trigger_log_cb_t)(const char* line);
typedef void (*trigger_nvs_cb_t)(uint32_t count, uint32_t ts_ms);
typedef void (*trigger_freeze_cb_t)(uint32_t fire_tick, uint32_t t_ms, const float p_history[3], uint32_t pre_ts);

/**
 * Initialize trigger state machine and persistent NVS storage.
 * Reads boot_count and trigger_count from NVS and increments boot_count.
 */
void trigger_init(void);

/**
 * Reset trigger state machine to clean initial state (LISTEN, LED OFF, count=0).
 */
void trigger_reset(void);

/**
 * Process one discrete hop tick (500 ms).
 *
 * @param p_pos Model probability for "Spectra" (0.0 .. 1.0)
 * @param tick Monotonic integer tick index
 * @param t_ms Timestamp of this tick in milliseconds
 * @param src_window_16k Optional pointer to 16,000 PCM samples for pre-roll freeze (can be NULL in mocks)
 * @return true if a FIRE event occurred on this tick, false otherwise
 */
bool trigger_tick(float p_pos, uint32_t tick, uint32_t t_ms, const int16_t* src_window_16k);

/**
 * Get pointer to trigger context (for telemetry, status inspection & tests).
 */
const trigger_context_t* trigger_get_context(void);

/**
 * Register custom callbacks (used by host unit test for mocks).
 */
void trigger_set_callbacks(trigger_led_cb_t led_cb,
                           trigger_log_cb_t log_cb,
                           trigger_nvs_cb_t nvs_cb,
                           trigger_freeze_cb_t freeze_cb);

/**
 * Autonomous self-test on Stream D vectors asserting events [2, 9].
 * @return true if Stream D evaluates exactly to events [2, 9], false otherwise.
 */
bool trigger_run_selftest_stream_d(void);

#ifdef __cplusplus
}
#endif

#endif  // TRIGGER_H_
