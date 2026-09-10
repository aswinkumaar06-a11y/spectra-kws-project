/**
 * vad_calib.cc — Spectra VAD Acoustic Noise Floor Auto-Calibration Implementation
 */

#include "vad_calib.h"
#include <cmath>
#include <cstring>
#include <cstdio>

#ifdef ESP_PLATFORM
#include "esp_log.h"
static const char* TAG = "VAD_CALIB";
#else
#define ESP_LOGI(tag, fmt, ...) printf("[%s] " fmt "\n", tag, ##__VA_ARGS__)
#define ESP_LOGW(tag, fmt, ...) printf("[WARN][%s] " fmt "\n", tag, ##__VA_ARGS__)
static const char* TAG = "VAD_CALIB";
#endif

void vad_calib_init(vad_calib_context_t* ctx, uint32_t target_frames) {
    if (ctx == NULL) return;

    memset(ctx, 0, sizeof(vad_calib_context_t));
    ctx->target_frames = (target_frames > 0 && target_frames <= 64)
                         ? target_frames : VAD_CALIB_DEFAULT_FRAMES;
    ctx->calibrated_th = SPECTRA_VAD_THRESHOLD; // fallback default
}

bool vad_calib_feed_energy(vad_calib_context_t* ctx, float energy) {
    if (ctx == NULL) return false;
    if (ctx->is_complete) return true;

    ctx->frame_energies[ctx->frames_collected++] = energy;

    if (ctx->frames_collected >= ctx->target_frames) {
        // Compute mean
        double sum = 0.0;
        for (uint32_t i = 0; i < ctx->target_frames; i++) {
            sum += (double)ctx->frame_energies[i];
        }
        double mu = sum / (double)ctx->target_frames;

        // Compute standard deviation
        double sum_sq_diff = 0.0;
        for (uint32_t i = 0; i < ctx->target_frames; i++) {
            double diff = (double)ctx->frame_energies[i] - mu;
            sum_sq_diff += diff * diff;
        }
        double sigma = sqrt(sum_sq_diff / (double)ctx->target_frames);

        // Calibrated threshold = mu + 3 * sigma, clamped to [0.012, 0.060]
        float raw_th = (float)(mu + 3.0 * sigma);
        float clamped_th = raw_th;
        if (clamped_th < VAD_CALIB_MIN_THRESHOLD) clamped_th = VAD_CALIB_MIN_THRESHOLD;
        if (clamped_th > VAD_CALIB_MAX_THRESHOLD) clamped_th = VAD_CALIB_MAX_THRESHOLD;

        ctx->noise_mean = (float)mu;
        ctx->noise_std = (float)sigma;
        ctx->calibrated_th = clamped_th;
        ctx->is_complete = true;

        ESP_LOGI(TAG, "Calibration complete (%u frames): Noise mu=%.4f, sigma=%.4f -> TH=%.4f (raw=%.4f)",
                 (unsigned)ctx->target_frames, (float)mu, (float)sigma, clamped_th, raw_th);

        return true;
    }

    return false;
}

bool vad_calib_feed_pcm(vad_calib_context_t* ctx, const int16_t* pcm_100ms) {
    if (ctx == NULL || pcm_100ms == NULL) return false;
    float rms = vad_compute_rms(pcm_100ms, SPECTRA_VAD_FRAME_SAMPLES);
    return vad_calib_feed_energy(ctx, rms);
}

void vad_calib_apply(const vad_calib_context_t* ctx, vad_config_t* vad) {
    if (ctx == NULL || vad == NULL) return;
    if (ctx->is_complete) {
        vad->threshold = ctx->calibrated_th;
    }
}
