/**
 * test_vad_host.c — Spectra Phase 5 Host-Side VAD & U2 Boundary Verification
 *
 * Verifies:
 *   1. Vector V1 (burst): START@2, END@8
 *   2. Vector V2 (blip): No START, remains IDLE
 *   3. Vector V3 (pause): START@0, END@9 (<=2 frame pause absorbed)
 *   4. U2 Boundary Test: MFCC energy threshold 0.029 rejected, 0.031 accepted
 *   5. RMS frame computation accuracy on PCM samples
 *   6. 10-second cap (100 frames) force endpointing
 */

#include "vad.h"
#include "mfcc.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <string.h>

#define COLOR_GREEN "\033[32m"
#define COLOR_RED   "\033[31m"
#define COLOR_BLUE  "\033[34m"
#define COLOR_RESET "\033[0m"

static int g_tests_passed = 0;
static int g_tests_failed = 0;

#define TEST_ASSERT(cond, desc) do { \
    if (cond) { \
        printf("  [" COLOR_GREEN "PASS" COLOR_RESET "] %s\n", desc); \
        g_tests_passed++; \
    } else { \
        printf("  [" COLOR_RED "FAIL" COLOR_RESET "] %s (line %d)\n", desc, __LINE__); \
        g_tests_failed++; \
    } \
} while(0)

void test_vector_v1(void) {
    printf("[TEST 1] Vector V1 (burst utterance)...\n");
    float v1[] = {0.005f, 0.008f, 0.05f, 0.09f, 0.12f, 0.10f, 0.004f, 0.003f, 0.002f, 0.001f};
    size_t n = sizeof(v1) / sizeof(v1[0]);

    vad_config_t vad;
    vad_init(&vad, 0.02f);

    for (size_t i = 0; i < n; i++) {
        vad_process_energy(&vad, v1[i]);
    }

    TEST_ASSERT(vad.start_frame == 2, "V1 start frame must be exactly 2 (START@2)");
    TEST_ASSERT(vad.end_frame == 8, "V1 end frame must be exactly 8 (END@8)");
    TEST_ASSERT(vad.state == VAD_STATE_END, "V1 terminal state must be VAD_STATE_END");
}

void test_vector_v2(void) {
    printf("[TEST 2] Vector V2 (blip noise, single frame)...\n");
    float v2[] = {0.005f, 0.06f, 0.004f, 0.003f, 0.002f, 0.001f};
    size_t n = sizeof(v2) / sizeof(v2[0]);

    vad_config_t vad;
    vad_init(&vad, 0.02f);

    for (size_t i = 0; i < n; i++) {
        vad_state_t st = vad_process_energy(&vad, v2[i]);
        TEST_ASSERT(st == VAD_STATE_IDLE, "V2 must never leave VAD_STATE_IDLE");
    }

    TEST_ASSERT(vad.start_frame == -1, "V2 start frame must remain -1 (no start)");
    TEST_ASSERT(vad.end_frame == -1, "V2 end frame must remain -1 (no end)");
}

void test_vector_v3(void) {
    printf("[TEST 3] Vector V3 (speech with 2-frame pause absorbed)...\n");
    float v3[] = {0.06f, 0.07f, 0.004f, 0.005f, 0.08f, 0.09f, 0.10f, 0.003f, 0.002f, 0.001f};
    size_t n = sizeof(v3) / sizeof(v3[0]);

    vad_config_t vad;
    vad_init(&vad, 0.02f);

    for (size_t i = 0; i < n; i++) {
        vad_process_energy(&vad, v3[i]);
    }

    TEST_ASSERT(vad.start_frame == 0, "V3 start frame must be exactly 0 (START@0)");
    TEST_ASSERT(vad.end_frame == 9, "V3 end frame must be exactly 9 (END@9)");
    TEST_ASSERT(vad.state == VAD_STATE_END, "V3 terminal state must be VAD_STATE_END");
}

void test_u2_boundary(void) {
    printf("[TEST 4] U2 Boundary Proof (Energy Threshold 0.029 vs 0.031)...\n");

    // Allocate 1.0s clip (16,000 samples)
    int16_t* pcm = (int16_t*)calloc(SPECTRA_CLIP_SAMPLES, sizeof(int16_t));
    assert(pcm != NULL);

    // Case A: 0.029 peak amplitude (round(0.029 * 32768) = 950)
    pcm[100] = 950;
    float peak_amp = 0.0f;
    bool gate_a = mfcc_check_energy_gate(pcm, &peak_amp);
    printf("  Sub-boundary 0.029: peak_amp = %.5f, gate = %s\n", peak_amp, gate_a ? "TRUE" : "FALSE");
    TEST_ASSERT(!gate_a, "Peak 0.029 (< 0.030) must be rejected by energy gate");
    TEST_ASSERT(peak_amp < 0.030f, "Peak amplitude must be strictly < 0.030");

    // Case B: 0.031 peak amplitude (round(0.031 * 32768) = 1016)
    pcm[100] = 1016;
    bool gate_b = mfcc_check_energy_gate(pcm, &peak_amp);
    printf("  Super-boundary 0.031: peak_amp = %.5f, gate = %s\n", peak_amp, gate_b ? "TRUE" : "FALSE");
    TEST_ASSERT(gate_b, "Peak 0.031 (>= 0.030) must be accepted by energy gate");
    TEST_ASSERT(peak_amp >= 0.030f, "Peak amplitude must be >= 0.030");

    free(pcm);
}

void test_rms_pcm(void) {
    printf("[TEST 5] 100 ms RMS Frame PCM Computation...\n");

    int16_t frame[SPECTRA_VAD_FRAME_SAMPLES];
    // DC signal at amplitude 16384 (0.5 normalized)
    for (size_t i = 0; i < SPECTRA_VAD_FRAME_SAMPLES; i++) {
        frame[i] = 16384;
    }

    float rms = vad_compute_rms(frame, SPECTRA_VAD_FRAME_SAMPLES);
    printf("  DC 0.5 RMS computed: %.4f (expected: 0.5000)\n", rms);
    TEST_ASSERT(fabsf(rms - 0.5f) < 1e-4f, "DC 0.5 signal RMS must equal 0.5000");

    // Silence
    memset(frame, 0, sizeof(frame));
    rms = vad_compute_rms(frame, SPECTRA_VAD_FRAME_SAMPLES);
    TEST_ASSERT(rms == 0.0f, "Silence frame RMS must equal 0.0000");
}

void test_max_duration_cap(void) {
    printf("[TEST 6] Utterance 10-Second Hard Cap (100 frames)...\n");

    vad_config_t vad;
    vad_init(&vad, 0.02f);

    // Feed continuous speech frames (energy = 0.05)
    for (int i = 0; i < 99; i++) {
        vad_state_t st = vad_process_energy(&vad, 0.05f);
        if (i >= 1) {
            TEST_ASSERT(st == VAD_STATE_SPEECH, "Frames 1..98 must be in SPEECH state");
        }
    }

    // 100th frame (index 99) reaches the 100-frame cap
    vad_state_t st_cap = vad_process_energy(&vad, 0.05f);
    TEST_ASSERT(st_cap == VAD_STATE_END, "100th frame must trigger VAD_STATE_END on 10s cap");
    TEST_ASSERT(vad.end_frame == 99, "End frame must equal 99 (100 frames total)");
}

int main(void) {
    printf("====================================================\n");
    printf("  Spectra Phase 5 Endpoint VAD & U2 Boundary Tests  \n");
    printf("====================================================\n");

    test_vector_v1();
    test_vector_v2();
    test_vector_v3();
    test_u2_boundary();
    test_rms_pcm();
    test_max_duration_cap();

    printf("\n----------------------------------------------------\n");
    printf("Summary: %d passed, %d failed\n", g_tests_passed, g_tests_failed);
    printf("====================================================\n");

    return (g_tests_failed == 0) ? 0 : 1;
}
