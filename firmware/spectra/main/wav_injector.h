/**
 * wav_injector.h — Deterministic Test Audio Injector for Spectra KWS
 *
 * Provides synthetic, mathematically exact audio streams for testing the
 * ring buffer, windowing, and feature extraction pipelines without requiring
 * live audio input or physical microphones.
 */

#ifndef SPECTRA_WAV_INJECTOR_H_
#define SPECTRA_WAV_INJECTOR_H_

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    INJECTOR_MODE_RAMP = 0,   // Linear 16-bit integer counter (0, 1, 2, 3...)
    INJECTOR_MODE_SINE = 1,   // Deterministic sine wave (e.g. 1000 Hz tone)
    INJECTOR_MODE_BUFFER = 2  // Replay from preloaded PCM buffer
} injector_mode_t;

/**
 * Initialize the synthetic test injector.
 *
 * @param mode Pattern generation mode
 * @param frequency_hz Frequency in Hz if mode is INJECTOR_MODE_SINE (e.g. 1000.0f)
 */
void wav_injector_init(injector_mode_t mode, float frequency_hz);

/**
 * Fill destination buffer with n_samples of synthetic audio.
 *
 * @param dest Output buffer
 * @param n_samples Number of samples to generate
 * @return Number of samples written
 */
size_t wav_injector_generate_block(int16_t* dest, size_t n_samples);

/**
 * Set an external int16 PCM buffer for replay in INJECTOR_MODE_BUFFER.
 *
 * @param buffer Pointer to PCM buffer
 * @param total_samples Total samples in the buffer
 */
void wav_injector_set_buffer(const int16_t* buffer, size_t total_samples);

/**
 * Calculate standard CRC-32 checksum across int16 PCM buffer for bit-exact validation.
 *
 * @param data Pointer to PCM samples
 * @param n_samples Number of samples
 * @return 32-bit CRC checksum
 */
uint32_t wav_injector_compute_crc32(const int16_t* data, size_t n_samples);

/**
 * Reset generator internal phase / sample counter to zero.
 */
void wav_injector_reset(void);

#ifdef __cplusplus
}
#endif

#endif  // SPECTRA_WAV_INJECTOR_H_
