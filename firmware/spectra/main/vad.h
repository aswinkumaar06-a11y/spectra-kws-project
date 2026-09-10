/**
 * vad.h — Spectra Endpoint Voice Activity Detection (VAD)
 *
 * Implements 100 ms RMS energy framing and deterministic state machine:
 *   IDLE --(E > TH x2)--> SPEECH --(E < TH x3)--> END
 *
 * Invariants:
 *   - Blip (< 2 frames > TH) never leaves IDLE.
 *   - Hangover absorbs <= 2 frames pause during speech.
 *   - END trigger requires >= 3-frame speech duration.
 *   - Vectors V1, V2, V3 validated deterministically in test_vad_host.c.
 */

#ifndef SPECTRA_VAD_H_
#define SPECTRA_VAD_H_

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include "spectra_config.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    VAD_STATE_IDLE = 0,
    VAD_STATE_SPEECH = 1,
    VAD_STATE_END = 2
} vad_state_t;

typedef struct {
    float threshold;             // RMS energy threshold (default: SPECTRA_VAD_THRESHOLD = 0.02)
    int speech_trigger_frames;   // Consecutive frames > TH to confirm speech (default: 2)
    int silence_trigger_frames;  // Consecutive frames < TH to trigger END (default: 3)
    int min_speech_frames;       // Minimum speech frames required for valid utterance (default: 3)
    int max_speech_frames;       // Maximum speech duration frames (default: 100 = 10.0s)

    // Dynamic state
    vad_state_t state;
    int current_frame_idx;
    int speech_candidate_frame;  // First candidate frame index when entering speech
    int consecutive_speech;      // Current run of speech frames
    int consecutive_silence;     // Current run of silence frames
    int total_speech_frames;     // Total speech frames accumulated in this utterance
    int start_frame;             // Start frame index of active/completed utterance (-1 if none)
    int end_frame;               // End frame index of completed utterance (-1 if none)
} vad_config_t;

/**
 * Initialize VAD state machine with given threshold.
 * Pass negative or zero threshold to use default SPECTRA_VAD_THRESHOLD (0.02).
 */
void vad_init(vad_config_t* vad, float threshold);

/**
 * Reset VAD runtime state machine back to IDLE without modifying thresholds.
 */
void vad_reset(vad_config_t* vad);

/**
 * Compute root-mean-square (RMS) energy normalized to [0.0, 1.0] from 16-bit PCM.
 */
float vad_compute_rms(const int16_t* pcm_samples, size_t num_samples);

/**
 * Process a single precomputed frame energy value through the state machine.
 * Returns the state after this frame (VAD_STATE_IDLE, VAD_STATE_SPEECH, VAD_STATE_END).
 */
vad_state_t vad_process_energy(vad_config_t* vad, float energy);

/**
 * Process a 100 ms PCM frame (SPECTRA_VAD_FRAME_SAMPLES = 1600 samples @ 16 kHz).
 * Computes RMS and feeds through state machine.
 */
vad_state_t vad_process_frame(vad_config_t* vad, const int16_t* pcm_100ms);

#ifdef __cplusplus
}
#endif

#endif  // SPECTRA_VAD_H_
