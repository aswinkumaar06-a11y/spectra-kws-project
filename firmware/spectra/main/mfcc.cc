/**
 * mfcc.cc — Spectra KWS MFCC Feature Extraction Implementation
 *
 * Implements exact desktop pipeline parity (scripts/audio_utils.py):
 *   1. Energy gate check (threshold >= 0.03)
 *   2. Peak normalization to [-1.0, 1.0] (guard max > 1e-6)
 *   3. 512 zero-padding on both edges (pad_mode='constant', 17,024 samples)
 *   4. 32 frames x 1024-point periodic Hann window
 *   5. Radix-2 DIT FFT -> 513 power spectrum bins (|X|^2)
 *   6. Sparse Mel filterbank multiplication (128 mel bins, Slaney norm)
 *   7. Log power (10*log10(max(S, 1e-10))) with top_db=80 clip relative to clip-global max
 *   8. DCT-II orthonormal projection -> 40 MFCCs x 32 time frames
 *   9. Banker's round-half-even INT8 quantization (scale=3.9962144, zp=60)
 */

#include "mfcc.h"
#include "spectra_config.h"

#include "hann_table.h"
#include "twiddle_table.h"
#include "mel_table.h"
#include "dct_table.h"

#include <cmath>
#include <cstring>
#include <cstdlib>

#define N_FFT           SPECTRA_N_FFT
#define TARGET_SAMPLES  SPECTRA_CLIP_SAMPLES
#define HOP_LENGTH      SPECTRA_HOP_LENGTH

#ifdef ESP_PLATFORM
#include "esp_log.h"
static const char* TAG = "MFCC";
#else
#include <cstdio>
#define ESP_LOGI(tag, fmt, ...) printf("[%s] " fmt "\n", tag, ##__VA_ARGS__)
#define ESP_LOGW(tag, fmt, ...) printf("[WARN][%s] " fmt "\n", tag, ##__VA_ARGS__)
#define ESP_LOGE(tag, fmt, ...) printf("[ERR][%s] " fmt "\n", tag, ##__VA_ARGS__)
static const char* TAG = "MFCC";
#endif

// Static lookup table for 1024-point bit-reversal
static uint16_t s_bit_rev[N_FFT];
static bool s_initialized = false;

// Working buffers (kept static to avoid stack overflow)
static float s_padded_audio[TARGET_SAMPLES + N_FFT];  // 17,024 samples
static float s_fft_re[N_FFT];
static float s_fft_im[N_FFT];
static float s_power[SPECTRA_FFT_BINS];
static float s_log_mel[SPECTRA_N_MELS][SPECTRA_N_FRAMES];

void mfcc_init(void) {
    if (s_initialized) return;

    // Precompute bit-reversal table for 10-bit indices (0..1023)
    for (uint16_t i = 0; i < N_FFT; i++) {
        uint16_t rev = 0;
        uint16_t val = i;
        for (int b = 0; b < 10; b++) {
            rev = (rev << 1) | (val & 1);
            val >>= 1;
        }
        s_bit_rev[i] = rev;
    }

    s_initialized = true;
    ESP_LOGI(TAG, "MFCC engine initialized (1024-pt FFT, 128 Mel bins, 40 MFCCs, 32 frames)");
}

bool mfcc_check_energy_gate(const int16_t* window_16k, float* out_peak_amp) {
    if (window_16k == NULL) return false;

    int16_t max_abs_int = 0;
    for (size_t i = 0; i < TARGET_SAMPLES; i++) {
        int16_t val = window_16k[i];
        int16_t a = (val < 0) ? (int16_t)(-val) : val;
        if (a > max_abs_int) {
            max_abs_int = a;
        }
    }

    float max_amp = (float)max_abs_int / 32768.0f;
    if (out_peak_amp != NULL) {
        *out_peak_amp = max_amp;
    }

    return (max_amp >= SPECTRA_ENERGY_GATE_THRESHOLD);
}

int8_t mfcc_quantize_sample(float val, float scale, int zero_point) {
    float q = val / scale;
    float rounded;

    // Strict banker's rounding (round-half-even)
    float fl = floorf(q);
    float diff = q - fl;
    if (diff < 0.5f) {
        rounded = fl;
    } else if (diff > 0.5f) {
        rounded = fl + 1.0f;
    } else {
        // Halfway case: round to nearest even integer
        rounded = (fmodf(fl, 2.0f) == 0.0f) ? fl : (fl + 1.0f);
    }

    int32_t quantized = (int32_t)rounded + zero_point;
    if (quantized < -128) quantized = -128;
    if (quantized > 127) quantized = 127;
    return (int8_t)quantized;
}

/**
 * 1024-point Radix-2 DIT FFT in float32.
 * Input in s_fft_re, s_fft_im is zeroed.
 * Output in s_fft_re, s_fft_im.
 */
static void run_fft_1024(void) {
    // Stage-by-stage Cooley-Tukey Radix-2 DIT
    for (int s = 1; s <= 10; s++) {
        int half_len = 1 << (s - 1);
        int len = 1 << s;
        int step = 512 / half_len;

        for (int k = 0; k < half_len; k++) {
            float tw_cos = kTwiddleCosFloat[k * step];
            float tw_sin = kTwiddleSinFloat[k * step];

            for (int i = k; i < N_FFT; i += len) {
                int j = i + half_len;
                float tre = tw_cos * s_fft_re[j] - tw_sin * s_fft_im[j];
                float tim = tw_cos * s_fft_im[j] + tw_sin * s_fft_re[j];

                float ure = s_fft_re[i];
                float uim = s_fft_im[i];

                s_fft_re[i] = ure + tre;
                s_fft_im[i] = uim + tim;
                s_fft_re[j] = ure - tre;
                s_fft_im[j] = uim - tim;
            }
        }
    }
}

bool mfcc_process_window(const int16_t* window_16k,
                         float* out_mfcc_float,
                         int8_t* out_mfcc_int8) {
    if (!s_initialized) {
        mfcc_init();
    }

    if (window_16k == NULL) {
        return false;
    }

    // ── Stage 1: Energy Gate Check ──────────────────────────────────────
    float max_amp = 0.0f;
    if (!mfcc_check_energy_gate(window_16k, &max_amp)) {
        // Gated as silence / background noise
        if (out_mfcc_float != NULL) {
            memset(out_mfcc_float, 0, SPECTRA_INPUT_HEIGHT * SPECTRA_INPUT_WIDTH * sizeof(float));
        }
        if (out_mfcc_int8 != NULL) {
            // Fill with zero_point (dequantizes to ~0.0)
            memset(out_mfcc_int8, (int8_t)SPECTRA_INPUT_ZERO_POINT,
                   SPECTRA_INPUT_HEIGHT * SPECTRA_INPUT_WIDTH);
        }
        return false;
    }

    // ── Stage 2: Peak Normalization & Centered Zero-Padding ──────────────
    // Pad 512 zeros at beginning and end (matching librosa melspectrogram constant pad)
    memset(&s_padded_audio[0], 0, 512 * sizeof(float));
    memset(&s_padded_audio[512 + TARGET_SAMPLES], 0, 512 * sizeof(float));

    int16_t max_abs_int = 0;
    for (size_t i = 0; i < TARGET_SAMPLES; i++) {
        int16_t val = window_16k[i];
        int16_t a = (val < 0) ? (int16_t)(-val) : val;
        if (a > max_abs_int) max_abs_int = a;
    }

    float norm_factor = (max_abs_int > 0) ? (1.0f / (float)max_abs_int) : 1.0f;
    for (size_t i = 0; i < TARGET_SAMPLES; i++) {
        s_padded_audio[512 + i] = (float)window_16k[i] * norm_factor;
    }

    // ── Stage 3-5: 32 Frames x (Window + FFT + Power + Sparse Mel) ──────
    float global_max_log = -1e30f;

    for (int t = 0; t < SPECTRA_N_FRAMES; t++) {
        const float* frame = &s_padded_audio[t * HOP_LENGTH];

        // Apply periodic Hann window and bit-reversal reordering
        for (int i = 0; i < N_FFT; i++) {
            uint16_t rev_idx = s_bit_rev[i];
            s_fft_re[i] = frame[rev_idx] * kHannWindowFloat[rev_idx];
            s_fft_im[i] = 0.0f;
        }

        // 1024-point FFT
        run_fft_1024();

        // Power spectrum: |X|^2 = re^2 + im^2 (first 513 bins)
        for (int k = 0; k < SPECTRA_FFT_BINS; k++) {
            s_power[k] = s_fft_re[k] * s_fft_re[k] + s_fft_im[k] * s_fft_im[k];
        }

        // Sparse Mel filterbank multiplication
        const float* weight_ptr = kMelWeights;
        for (int m = 0; m < SPECTRA_N_MELS; m++) {
            uint16_t start = kMelFilters[m].start_bin;
            uint16_t count = kMelFilters[m].count;
            double mel_energy = 0.0;

            for (uint16_t b = 0; b < count; b++) {
                mel_energy += (double)s_power[start + b] * (double)(*weight_ptr++);
            }

            // Power-to-dB floor: 10 * log10(max(S, 1e-10))
            float log_val = (float)(10.0 * log10(fmax(mel_energy, 1e-10)));
            s_log_mel[m][t] = log_val;
            if (log_val > global_max_log) {
                global_max_log = log_val;
            }
        }
    }

    // ── Stage 6: Clip-Global Top-dB Floor (80 dB below peak) ────────────
    float clip_threshold = global_max_log - 80.0f;
    for (int m = 0; m < SPECTRA_N_MELS; m++) {
        for (int t = 0; t < SPECTRA_N_FRAMES; t++) {
            if (s_log_mel[m][t] < clip_threshold) {
                s_log_mel[m][t] = clip_threshold;
            }
        }
    }

    // ── Stage 7: DCT-II Orthonormal Projection & Banker's Quantization ──
    for (int k = 0; k < SPECTRA_N_MFCC; k++) {
        for (int t = 0; t < SPECTRA_N_FRAMES; t++) {
            double mfcc_accum = 0.0;
            for (int m = 0; m < SPECTRA_N_MELS; m++) {
                mfcc_accum += (double)kDctMatrix[k][m] * (double)s_log_mel[m][t];
            }
            float mfcc_val = (float)mfcc_accum;

            // Output format: [40, 32] (rows = 40 MFCCs, cols = 32 frames)
            size_t out_idx = k * SPECTRA_N_FRAMES + t;

            if (out_mfcc_float != NULL) {
                out_mfcc_float[out_idx] = mfcc_val;
            }

            if (out_mfcc_int8 != NULL) {
                out_mfcc_int8[out_idx] = mfcc_quantize_sample(
                    mfcc_val, SPECTRA_INPUT_SCALE, SPECTRA_INPUT_ZERO_POINT
                );
            }
        }
    }

    return true;
}
