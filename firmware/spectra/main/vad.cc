/**
 * vad.cc — Spectra Endpoint Voice Activity Detection (VAD) Implementation
 */

#include "vad.h"
#include <cmath>
#include <cstring>

#ifdef ESP_PLATFORM
#include "esp_log.h"
static const char* TAG = "VAD";
#else
#include <cstdio>
#define ESP_LOGI(tag, fmt, ...) printf("[%s] " fmt "\n", tag, ##__VA_ARGS__)
#define ESP_LOGD(tag, fmt, ...)
static const char* TAG = "VAD";
#endif

void vad_init(vad_config_t* vad, float threshold) {
    if (vad == NULL) return;

    vad->threshold = (threshold > 0.0f) ? threshold : SPECTRA_VAD_THRESHOLD;
    vad->speech_trigger_frames = SPECTRA_VAD_SPEECH_TRIGGER;
    vad->silence_trigger_frames = SPECTRA_VAD_SILENCE_TRIGGER;
    vad->min_speech_frames = SPECTRA_VAD_MIN_SPEECH_FRAMES;
    vad->max_speech_frames = SPECTRA_STREAM_MAX_FRAMES;

    vad_reset(vad);
}

void vad_reset(vad_config_t* vad) {
    if (vad == NULL) return;

    vad->state = VAD_STATE_IDLE;
    vad->current_frame_idx = 0;
    vad->speech_candidate_frame = -1;
    vad->consecutive_speech = 0;
    vad->consecutive_silence = 0;
    vad->total_speech_frames = 0;
    vad->start_frame = -1;
    vad->end_frame = -1;
}

float vad_compute_rms(const int16_t* pcm_samples, size_t num_samples) {
    if (pcm_samples == NULL || num_samples == 0) {
        return 0.0f;
    }

    double sum_sq = 0.0;
    for (size_t i = 0; i < num_samples; i++) {
        double norm = (double)pcm_samples[i] / 32768.0;
        sum_sq += norm * norm;
    }

    return (float)sqrt(sum_sq / (double)num_samples);
}

vad_state_t vad_process_energy(vad_config_t* vad, float energy) {
    if (vad == NULL) return VAD_STATE_IDLE;

    int frame_idx = vad->current_frame_idx++;
    bool is_speech = (energy > vad->threshold);

    switch (vad->state) {
        case VAD_STATE_IDLE:
            if (is_speech) {
                vad->consecutive_speech++;
                if (vad->consecutive_speech == 1) {
                    vad->speech_candidate_frame = frame_idx;
                }
                if (vad->consecutive_speech >= vad->speech_trigger_frames) {
                    vad->state = VAD_STATE_SPEECH;
                    vad->start_frame = vad->speech_candidate_frame;
                    vad->total_speech_frames = vad->consecutive_speech;
                    vad->consecutive_silence = 0;
                }
            } else {
                vad->consecutive_speech = 0;
                vad->speech_candidate_frame = -1;
            }
            break;

        case VAD_STATE_SPEECH:
            if (is_speech) {
                vad->total_speech_frames++;
                vad->consecutive_speech++;
                vad->consecutive_silence = 0;

                // Max duration cap
                if (frame_idx - vad->start_frame + 1 >= vad->max_speech_frames) {
                    vad->state = VAD_STATE_END;
                    vad->end_frame = frame_idx;
                }
            } else {
                vad->consecutive_speech = 0;
                vad->consecutive_silence++;

                if (vad->consecutive_silence >= vad->silence_trigger_frames) {
                    if (vad->total_speech_frames >= vad->min_speech_frames) {
                        vad->state = VAD_STATE_END;
                        vad->end_frame = frame_idx;
                    } else {
                        // Utterance too short — discard as noise/blip and return to IDLE
                        vad->state = VAD_STATE_IDLE;
                        vad->start_frame = -1;
                        vad->consecutive_silence = 0;
                        vad->total_speech_frames = 0;
                    }
                } else if (frame_idx - vad->start_frame + 1 >= vad->max_speech_frames) {
                    vad->state = VAD_STATE_END;
                    vad->end_frame = frame_idx;
                }
            }
            break;

        case VAD_STATE_END:
            // Terminal state until explicit vad_reset()
            break;
    }

    return vad->state;
}

vad_state_t vad_process_frame(vad_config_t* vad, const int16_t* pcm_100ms) {
    if (vad == NULL || pcm_100ms == NULL) return VAD_STATE_IDLE;

    float rms = vad_compute_rms(pcm_100ms, SPECTRA_VAD_FRAME_SAMPLES);
    return vad_process_energy(vad, rms);
}
