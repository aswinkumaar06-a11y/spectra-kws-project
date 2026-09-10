/**
 * test_calib_host.c — Spectra VAD Acoustic Calibration Host Unit Test
 *
 * Verifies:
 *   1. Quiet environment lower-bound clamping (raw < 0.012 -> 0.012)
 *   2. Normal acoustic environment tuning (0.012 <= TH <= 0.060)
 *   3. High noise environment upper-bound clamping (raw > 0.060 -> 0.060)
 *   4. Application of calibrated threshold to active VAD engine
 */

#include "vad_calib.h"
#include "vad.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#define COLOR_GREEN "\033[32m"
#define COLOR_RED   "\033[31m"
#define COLOR_RESET "\033[0m"

static int g_passed = 0;
static int g_failed = 0;

#define CHECK(cond, msg) do { \
    if (cond) { \
        printf("  [" COLOR_GREEN "PASS" COLOR_RESET "] %s\n", msg); \
        g_passed++; \
    } else { \
        printf("  [" COLOR_RED "FAIL" COLOR_RESET "] %s (line %d)\n", msg, __LINE__); \
        g_failed++; \
    } \
} while(0)

int main(void) {
    printf("====================================================\n");
    printf("  Spectra VAD Auto-Calibration Host Unit Test       \n");
    printf("====================================================\n");

    // Case 1: Quiet acoustic floor (lower bound test)
    printf("[1] Testing Quiet Environment (Lower Bound Clamp)...\n");
    vad_calib_context_t calib_quiet;
    vad_calib_init(&calib_quiet, 20);

    for (int i = 0; i < 20; i++) {
        float e = 0.002f + (float)(i % 3) * 0.0005f; // ~0.002 - 0.003
        vad_calib_feed_energy(&calib_quiet, e);
    }
    CHECK(calib_quiet.is_complete, "Calibration completed after 20 frames");
    printf("  Quiet raw TH: %.4f -> Clamped TH: %.4f\n",
           calib_quiet.noise_mean + 3.0f * calib_quiet.noise_std, calib_quiet.calibrated_th);
    CHECK(calib_quiet.calibrated_th == VAD_CALIB_MIN_THRESHOLD, "Clamped to lower safety bound 0.012");

    // Case 2: Moderate office noise floor (linear region test)
    printf("\n[2] Testing Moderate Noise Floor (Nominal Range)...\n");
    vad_calib_context_t calib_mod;
    vad_calib_init(&calib_mod, 20);

    for (int i = 0; i < 20; i++) {
        float e = 0.008f + (float)(i % 5) * 0.001f; // ~0.008 - 0.012
        vad_calib_feed_energy(&calib_mod, e);
    }
    CHECK(calib_mod.is_complete, "Calibration completed after 20 frames");
    printf("  Moderate mu: %.4f, sigma: %.4f -> TH: %.4f\n",
           calib_mod.noise_mean, calib_mod.noise_std, calib_mod.calibrated_th);
    CHECK(calib_mod.calibrated_th >= VAD_CALIB_MIN_THRESHOLD &&
          calib_mod.calibrated_th <= VAD_CALIB_MAX_THRESHOLD,
          "Threshold lies within nominal [0.012, 0.060] range");

    // Case 3: High background noise floor (upper bound test)
    printf("\n[3] Testing High Noise Floor (Upper Bound Clamp)...\n");
    vad_calib_context_t calib_loud;
    vad_calib_init(&calib_loud, 20);

    for (int i = 0; i < 20; i++) {
        float e = 0.035f + (float)(i % 7) * 0.005f; // ~0.035 - 0.065
        vad_calib_feed_energy(&calib_loud, e);
    }
    CHECK(calib_loud.is_complete, "Calibration completed after 20 frames");
    printf("  Loud raw TH: %.4f -> Clamped TH: %.4f\n",
           calib_loud.noise_mean + 3.0f * calib_loud.noise_std, calib_loud.calibrated_th);
    CHECK(calib_loud.calibrated_th == VAD_CALIB_MAX_THRESHOLD, "Clamped to upper safety bound 0.060");

    // Case 4: Application to VAD engine
    printf("\n[4] Testing Application to Active VAD Engine...\n");
    vad_config_t vad;
    vad_init(&vad, 0.02f);
    CHECK(vad.threshold == 0.02f, "Initial default threshold is 0.020");

    vad_calib_apply(&calib_mod, &vad);
    CHECK(vad.threshold == calib_mod.calibrated_th, "VAD threshold updated to calibrated value");

    printf("\n====================================================\n");
    printf("  Results: %d passed, %d failed\n", g_passed, g_failed);
    printf("====================================================\n");

    return (g_failed == 0) ? 0 : 1;
}
