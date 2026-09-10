/**
 * vad_calib.h — Spectra VAD Acoustic Noise Floor Auto-Calibration
 *
 * Automatically calibrates the VAD RMS threshold from ambient mic telemetry:
 *   1. Accumulates K ambient 100 ms frames (default: 20 frames = 2.0 s)
 *   2. Computes mean (mu) and standard deviation (sigma) of background RMS
 *   3. Sets TH = clamp(mu + 3.0 * sigma, 0.012, 0.060)
 *   4. Applies tuned threshold to VAD engine and logs baseline
 */

#ifndef SPECTRA_VAD_CALIB_H_
#define SPECTRA_VAD_CALIB_H_

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include "vad.h"

#ifdef __cplusplus
extern "C" {
#endif

#define VAD_CALIB_MIN_THRESHOLD    0.012f   // Minimum lower safety clamp
#define VAD_CALIB_MAX_THRESHOLD    0.060f   // Maximum upper safety clamp
#define VAD_CALIB_DEFAULT_FRAMES   20       // 20 x 100 ms = 2.0 seconds

typedef struct {
    float    noise_mean;        // Estimated mean background RMS
    float    noise_std;         // Estimated standard deviation
    float    calibrated_th;     // Resulting calibrated threshold
    uint32_t frames_collected;  // Current frame count
    uint32_t target_frames;     // Target frame count (e.g. 20)
    bool     is_complete;       // True when calibration finished
    float    frame_energies[64];// Sampled frame energies
} vad_calib_context_t;

/**
 * Initialize calibration context for K target frames (pass 0 for default 20).
 */
void vad_calib_init(vad_calib_context_t* ctx, uint32_t target_frames);

/**
 * Feed a 100 ms PCM chunk (1600 samples) to the calibrator during boot.
 * Returns true when calibration finishes.
 */
bool vad_calib_feed_pcm(vad_calib_context_t* ctx, const int16_t* pcm_100ms);

/**
 * Feed a precomputed 100 ms RMS energy value to the calibrator.
 */
bool vad_calib_feed_energy(vad_calib_context_t* ctx, float energy);

/**
 * Apply calibrated threshold to an active VAD engine.
 */
void vad_calib_apply(const vad_calib_context_t* ctx, vad_config_t* vad);

#ifdef __cplusplus
}
#endif

#endif  // SPECTRA_VAD_CALIB_H_
