/**
 * audio_ring.h — Circular Audio Buffer for Spectra KWS
 *
 * Target: XIAO ESP32-C5 (PSRAM allocation: 8 MB external SPI RAM)
 *
 * Implements a thread-safe circular ring buffer for streaming 16-bit PCM audio.
 * Manages continuous ingestion with a strict drop-oldest overrun policy,
 * providing zero-copy window peek and hop advancement for DSP / inference.
 */

#ifndef SPECTRA_AUDIO_RING_H_
#define SPECTRA_AUDIO_RING_H_

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include "spectra_config.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Ring buffer telemetry and state metrics.
 */
typedef struct {
    size_t capacity_samples;     // Total capacity in samples (e.g. 48,000)
    size_t available_samples;    // Current unread samples
    uint32_t overrun_count;      // Total number of overrun occurrences
    uint32_t dropped_samples;    // Cumulative samples dropped due to overrun
    bool is_psram;               // True if allocated in external PSRAM
} audio_ring_stats_t;

/**
 * Initialize the audio ring buffer in PSRAM (or standard RAM on host/fallback).
 *
 * @param capacity_samples Buffer capacity in samples (e.g. SPECTRA_RING_CAPACITY)
 * @return 0 on success, negative error code on failure.
 */
int audio_ring_init(size_t capacity_samples);

/**
 * Free memory and release resources used by the ring buffer.
 */
void audio_ring_deinit(void);

/**
 * Write samples into the ring buffer.
 * If incoming samples exceed remaining space, the oldest unread samples
 * are discarded to accommodate the new audio (drop-oldest policy), and an
 * overrun event is recorded and logged.
 *
 * @param src Pointer to int16_t PCM source samples
 * @param n_samples Number of samples to write
 * @return Number of samples written
 */
size_t audio_ring_write(const int16_t* src, size_t n_samples);

/**
 * Peek at a window of samples without advancing the read pointer.
 *
 * @param dst Destination buffer (must hold at least window_samples)
 * @param window_samples Number of samples to copy (e.g. SPECTRA_WINDOW_SAMPLES = 16,000)
 * @return true if window_samples were available and copied; false if buffer has insufficient data.
 */
bool audio_ring_peek_window(int16_t* dst, size_t window_samples);

/**
 * Advance the read pointer by hop_samples.
 *
 * @param hop_samples Number of samples to consume (e.g. SPECTRA_HOP_SAMPLES = 8,000)
 * @return true if advanced successfully; false if fewer than hop_samples were available.
 */
bool audio_ring_advance(size_t hop_samples);

/**
 * Get number of currently available unread samples.
 */
size_t audio_ring_available(void);

/**
 * Get cumulative overrun event count.
 */
uint32_t audio_ring_overruns(void);

/**
 * Retrieve full telemetry and state snapshot.
 */
void audio_ring_get_stats(audio_ring_stats_t* stats);

/**
 * Reset ring buffer pointers and overrun counters to initial empty state.
 */
void audio_ring_reset(void);

#ifdef __cplusplus
}
#endif

#endif  // SPECTRA_AUDIO_RING_H_
