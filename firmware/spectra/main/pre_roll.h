/**
 * pre_roll.h - Pre-roll audio window freeze buffer for keyword verification
 *
 * Captures the exact 1.0s (16,000 INT16 samples = 32 KB) audio window
 * that triggered the keyword detection event, along with trigger metadata.
 *
 * Placed in internal SRAM (.bss). Live capture MUST NOT stall during freeze.
 */

#ifndef PRE_ROLL_H_
#define PRE_ROLL_H_

#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include "spectra_config.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint32_t fire_tick;
    uint32_t fire_time_ms;
    uint32_t pre_ts;
    float p[3];
    int16_t samples[SPECTRA_PRE_ROLL_SAMPLES]; // 16000 int16 samples (32,000 bytes)
    bool is_frozen;
} spectra_pre_roll_t;

/**
 * Initialize pre-roll buffer.
 */
void pre_roll_init(void);

/**
 * Freeze 16,000 samples from the audio ring buffer into the pre-roll snapshot.
 *
 * @param fire_tick Discrete hop tick of the firing event
 * @param fire_time_ms Timestamp of firing event in ms
 * @param p_history Array of 3 probabilities [p_N-3, p_N-2, p_N-1]
 * @param src_16k Pointer to 16,000 int16 samples
 * @param pre_ts Timestamp of pre-roll window start (ms)
 */
void pre_roll_freeze(uint32_t fire_tick,
                     uint32_t fire_time_ms,
                     const float p_history[3],
                     const int16_t* src_16k,
                     uint32_t pre_ts);

/**
 * Access the currently frozen pre-roll buffer.
 */
const spectra_pre_roll_t* pre_roll_get(void);

#ifdef __cplusplus
}
#endif

#endif  // PRE_ROLL_H_
