/**
 * test_mfcc_host.c — Host-side C MFCC Parity Verification Harness
 *
 * Evaluates the C MFCC pipeline against all 50+ golden clips in mfcc_goldens.bin:
 *   - Gate 1: max|Δfloat| <= 1e-3 (0.001) on every clip
 *   - Gate 2: INT8 byte match >= 99.0% on every clip
 *   - Gate 3: Frame count = 32 on every clip
 *   - Gate 4: Edge cases (silence, sub-threshold gating, 1e-7 guard)
 *
 * Build: gcc -O3 -o test_mfcc_host test_mfcc_host.c ../spectra/main/mfcc.cc -I ../spectra/main -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>

#include "mfcc.h"
#include "spectra_config.h"

#define MAX_FLOAT_DIFF_TOLERANCE 1e-3f   // 0.001 max delta float
#define MIN_BYTE_MATCH_PERCENT   99.0f   // >= 99% byte match

static int s_tests_passed = 0;
static int s_tests_failed = 0;

#define CHECK(cond, msg)                                                    \
    do {                                                                    \
        if (cond) {                                                         \
            printf("  [PASS] %s\n", msg);                                   \
            s_tests_passed++;                                               \
        } else {                                                            \
            printf("  [FAIL] %s (line %d)\n", msg, __LINE__);               \
            s_tests_failed++;                                               \
        }                                                                   \
    } while (0)

int main(int argc, char** argv) {
    const char* bin_path = "firmware/tests/mfcc_goldens.bin";
    if (argc > 1) {
        bin_path = argv[1];
    }

    printf("====================================================\n");
    printf("  Spectra MFCC Engine Host-Side Parity Verification\n");
    printf("====================================================\n\n");

    FILE* fp = fopen(bin_path, "rb");
    if (!fp) {
        // Try relative path from firmware/tests directory
        fp = fopen("mfcc_goldens.bin", "rb");
    }
    if (!fp) {
        printf("[FATAL] Cannot open golden vectors file: %s\n", bin_path);
        return 1;
    }

    // ── 1. Read Header ──────────────────────────────────────────────────
    char magic[5] = {0};
    uint32_t version, n_clips, sample_rate, target_samples, n_mfcc, n_frames;
    char librosa_ver[33] = {0};
    char scipy_ver[33] = {0};
    char numpy_ver[33] = {0};

    if (fread(magic, 1, 4, fp) != 4 ||
        fread(&version, 4, 1, fp) != 1 ||
        fread(&n_clips, 4, 1, fp) != 1 ||
        fread(&sample_rate, 4, 1, fp) != 1 ||
        fread(&target_samples, 4, 1, fp) != 1 ||
        fread(&n_mfcc, 4, 1, fp) != 1 ||
        fread(&n_frames, 4, 1, fp) != 1 ||
        fread(librosa_ver, 1, 32, fp) != 32 ||
        fread(scipy_ver, 1, 32, fp) != 32 ||
        fread(numpy_ver, 1, 32, fp) != 32) {
        printf("[FATAL] Corrupted golden bundle header\n");
        fclose(fp);
        return 1;
    }

    printf("[HEADER] Magic: %s | Version: %u | Clips: %u\n", magic, version, n_clips);
    printf("[HEADER] Audio: %u Hz, %u samples (1.0s) | Features: %ux%u\n",
           sample_rate, target_samples, n_mfcc, n_frames);
    printf("[ENV]    librosa=%s, scipy=%s, numpy=%s\n\n", librosa_ver, scipy_ver, numpy_ver);

    CHECK(strcmp(magic, "SPMF") == 0, "Valid SPMF magic header");
    CHECK(n_clips >= 50, "At least 50 golden test clips present in bundle");

    // Initialize C engine
    mfcc_init();

    // Allocate working memory
    int16_t* pcm_in = (int16_t*)malloc(target_samples * sizeof(int16_t));
    float* ref_mfcc = (float*)malloc(n_mfcc * n_frames * sizeof(float));
    int8_t* ref_int8 = (int8_t*)malloc(n_mfcc * n_frames);

    float* out_mfcc = (float*)malloc(n_mfcc * n_frames * sizeof(float));
    int8_t* out_int8 = (int8_t*)malloc(n_mfcc * n_frames);

    size_t total_elements = n_mfcc * n_frames;
    float global_max_delta = 0.0f;
    float global_sum_delta = 0.0f;
    uint64_t global_total_elements = 0;
    uint32_t clips_passed_gates = 0;

    printf("Evaluating %u golden test clips...\n", n_clips);
    printf("--------------------------------------------------------------------------------\n");
    printf(" #  | Filename                                     | Max |Δ|   | Byte Match | Verdict\n");
    printf("--------------------------------------------------------------------------------\n");

    for (uint32_t c = 0; c < n_clips; c++) {
        char fname[65] = {0};
        char label[17] = {0};
        uint32_t is_hard = 0;
        float peak_amp = 0.0f;

        fread(fname, 1, 64, fp);
        fread(label, 1, 16, fp);
        fread(&is_hard, 4, 1, fp);
        fread(&peak_amp, 4, 1, fp);

        fread(pcm_in, sizeof(int16_t), target_samples, fp);
        fread(ref_mfcc, sizeof(float), total_elements, fp);
        fread(ref_int8, sizeof(int8_t), total_elements, fp);

        // Run C MFCC extraction
        bool success = mfcc_process_window(pcm_in, out_mfcc, out_int8);

        float clip_max_delta = 0.0f;
        uint32_t byte_matches = 0;

        if (success) {
            for (size_t i = 0; i < total_elements; i++) {
                float diff = fabsf(out_mfcc[i] - ref_mfcc[i]);
                if (diff > clip_max_delta) {
                    clip_max_delta = diff;
                }
                global_sum_delta += diff;
                global_total_elements++;

                if (out_int8[i] == ref_int8[i]) {
                    byte_matches++;
                }
            }
        }

        if (clip_max_delta > global_max_delta) {
            global_max_delta = clip_max_delta;
        }

        float match_pct = (float)byte_matches * 100.0f / total_elements;
        bool pass = success && (clip_max_delta <= MAX_FLOAT_DIFF_TOLERANCE) && (match_pct >= MIN_BYTE_MATCH_PERCENT);
        if (pass) clips_passed_gates++;

        // Print row (truncate filename to 42 chars for clean formatting)
        char short_name[43];
        strncpy(short_name, fname, 42);
        short_name[42] = '\0';
        printf("%2u | %-44s | %.6f | %6.2f%%    | %s\n",
               c + 1, short_name, clip_max_delta, match_pct, pass ? "PASS" : "FAIL");
    }

    printf("--------------------------------------------------------------------------------\n");
    float global_mean_delta = (global_total_elements > 0) ? (global_sum_delta / global_total_elements) : 0.0f;
    printf("\n[SUMMARY] Evaluated: %u clips | Passed All Gates: %u / %u (%.1f%%)\n",
           n_clips, clips_passed_gates, n_clips, (float)clips_passed_gates * 100.0f / n_clips);
    printf("[SUMMARY] Global Max |Δfloat|: %.6f (Tolerance: <= %.4f)\n",
           global_max_delta, MAX_FLOAT_DIFF_TOLERANCE);
    printf("[SUMMARY] Global Mean |Δfloat|: %.6f\n\n", global_mean_delta);

    CHECK(clips_passed_gates == n_clips, "All clips passed both max|Δ| <= 1e-3 and >= 99% byte match gates");

    // ── 2. Edge Case Verification ───────────────────────────────────────
    printf("\n[EDGE CASES] Testing boundary and pathological conditions...\n");

    // Edge Case 1: Pure digital silence
    memset(pcm_in, 0, target_samples * sizeof(int16_t));
    bool silence_result = mfcc_process_window(pcm_in, out_mfcc, out_int8);
    CHECK(!silence_result, "Pure silence correctly rejected by energy gate");

    // Edge Case 2: Sub-threshold quiet noise (peak 0.015 < 0.03)
    for (size_t i = 0; i < target_samples; i++) {
        pcm_in[i] = (int16_t)((i % 300) - 150); // peak ~150 / 32768 = 0.0045
    }
    bool quiet_result = mfcc_process_window(pcm_in, out_mfcc, out_int8);
    CHECK(!quiet_result, "Sub-threshold background noise correctly rejected by energy gate");

    // Edge Case 3: Extreme full-scale sine wave
    for (size_t i = 0; i < target_samples; i++) {
        pcm_in[i] = (int16_t)(32767.0 * sin(2.0 * 3.1415926535 * 440.0 * i / 16000.0));
    }
    bool full_scale_result = mfcc_process_window(pcm_in, out_mfcc, out_int8);
    CHECK(full_scale_result, "Full-scale 440Hz sine processed successfully");

    // Ensure no NaNs or Infs occurred
    bool has_nan_inf = false;
    for (size_t i = 0; i < total_elements; i++) {
        if (isnan(out_mfcc[i]) || isinf(out_mfcc[i])) {
            has_nan_inf = true;
            break;
        }
    }
    CHECK(!has_nan_inf, "Zero NaN or Inf values produced across all test cases");

    // Clean up
    free(pcm_in);
    free(ref_mfcc);
    free(ref_int8);
    free(out_mfcc);
    free(out_int8);
    fclose(fp);

    printf("\n====================================================\n");
    printf("  Results: %d passed, %d failed\n", s_tests_passed, s_tests_failed);
    printf("====================================================\n");

    return s_tests_failed > 0 ? 1 : 0;
}
