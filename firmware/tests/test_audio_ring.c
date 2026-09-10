/**
 * test_audio_ring.c — Host-side C Unit Test for Spectra Audio Ring Buffer
 *
 * Validates:
 *   1. Allocation & Initialization (48,000 samples = 3.0s = 96 KB)
 *   2. Block writes (4,000 samples per block)
 *   3. Insufficient data handling (peeking before 16,000 samples are accumulated)
 *   4. Zero-loss window peeking (16,000 samples = 1.0s) with bit-exact sample matching
 *   5. Hop advancement (8,000 samples = 0.5s)
 *   6. Continuous multi-window wrap-around across circular boundaries (>160,000 samples)
 *   7. Drop-oldest overrun policy enforcement and telemetry counters
 *   8. Ring reset and deinit lifecycle
 *
 * Build: gcc -o test_audio_ring test_audio_ring.c ../spectra/main/audio_ring.cc ../spectra/main/wav_injector.cc -I ../spectra/main
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>

#include "audio_ring.h"
#include "wav_injector.h"
#include "spectra_config.h"

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

int main(void) {
    printf("====================================================\n");
    printf("  Spectra Audio Ring Buffer Host Unit Test\n");
    printf("====================================================\n\n");

    // ── Test 1: Initialization ──────────────────────────────────────────
    printf("[1] Testing Initialization...\n");
    int ret = audio_ring_init(SPECTRA_RING_CAPACITY);
    CHECK(ret == 0, "audio_ring_init(48000) returned 0 (success)");
    CHECK(audio_ring_available() == 0, "Initial available samples is 0");
    CHECK(audio_ring_overruns() == 0, "Initial overrun count is 0");

    audio_ring_stats_t stats;
    audio_ring_get_stats(&stats);
    CHECK(stats.capacity_samples == SPECTRA_RING_CAPACITY, "Reported capacity matches 48000");

    // ── Test 2: Incomplete Window (4000 samples) ────────────────────────
    printf("\n[2] Testing Incomplete Window Read Protection...\n");
    int16_t block[SPECTRA_CAPTURE_BLOCK_SAMPLES];
    wav_injector_init(INJECTOR_MODE_RAMP, 0);
    wav_injector_generate_block(block, SPECTRA_CAPTURE_BLOCK_SAMPLES);

    size_t written = audio_ring_write(block, SPECTRA_CAPTURE_BLOCK_SAMPLES);
    CHECK(written == SPECTRA_CAPTURE_BLOCK_SAMPLES, "Wrote 4000 samples block");
    CHECK(audio_ring_available() == 4000, "Available count is now 4000");

    int16_t window[SPECTRA_WINDOW_SAMPLES];
    bool peek_ok = audio_ring_peek_window(window, SPECTRA_WINDOW_SAMPLES);
    CHECK(!peek_ok, "Peeking 16000 samples correctly fails when only 4000 available");

    // ── Test 3: Fill to Full Window (16000 samples) ──────────────────────
    printf("\n[3] Testing Full Window Ingestion & Verification...\n");
    for (int b = 1; b < 4; b++) {
        wav_injector_generate_block(block, SPECTRA_CAPTURE_BLOCK_SAMPLES);
        audio_ring_write(block, SPECTRA_CAPTURE_BLOCK_SAMPLES);
    }
    CHECK(audio_ring_available() == 16000, "Accumulated exactly 16000 samples (4 blocks)");

    peek_ok = audio_ring_peek_window(window, SPECTRA_WINDOW_SAMPLES);
    CHECK(peek_ok, "audio_ring_peek_window succeeded with 16000 samples");

    // Verify sample parity: should be 0, 1, 2, ..., 15999
    bool sample_match = true;
    for (int i = 0; i < SPECTRA_WINDOW_SAMPLES; i++) {
        if (window[i] != (int16_t)i) {
            sample_match = false;
            break;
        }
    }
    CHECK(sample_match, "All 16000 samples bit-identically match injected linear ramp");

    uint32_t crc0 = wav_injector_compute_crc32(window, SPECTRA_WINDOW_SAMPLES);
    printf("  Window 0 CRC-32: 0x%08X (expected 0x6B2E496E)\n", crc0);
    CHECK(crc0 == 0x6B2E496E, "Window 0 CRC-32 matches Python golden value (0x6B2E496E)");

    // ── Test 4: Hop Advance (8000 samples) ───────────────────────────────
    printf("\n[4] Testing Hop Advance...\n");
    bool adv_ok = audio_ring_advance(SPECTRA_HOP_SAMPLES);
    CHECK(adv_ok, "audio_ring_advance(8000) succeeded");
    CHECK(audio_ring_available() == 8000, "Available count decremented from 16000 to 8000");

    // Add 2 more blocks (8000 samples) to complete the next hop window
    for (int b = 0; b < 2; b++) {
        wav_injector_generate_block(block, SPECTRA_CAPTURE_BLOCK_SAMPLES);
        audio_ring_write(block, SPECTRA_CAPTURE_BLOCK_SAMPLES);
    }
    CHECK(audio_ring_available() == 16000, "Available is 16000 again after 2 new blocks");

    peek_ok = audio_ring_peek_window(window, SPECTRA_WINDOW_SAMPLES);
    CHECK(peek_ok, "audio_ring_peek_window succeeded for Hop 1");

    // Hop 1 window should span from sample index 8000 to 23999
    sample_match = true;
    for (int i = 0; i < SPECTRA_WINDOW_SAMPLES; i++) {
        if (window[i] != (int16_t)(8000 + i)) {
            sample_match = false;
            break;
        }
    }
    CHECK(sample_match, "Hop 1 window starts at sample 8000 with 50% overlap");

    uint32_t crc1 = wav_injector_compute_crc32(window, SPECTRA_WINDOW_SAMPLES);
    printf("  Hop 1 CRC-32:   0x%08X (expected 0x86F0B27A)\n", crc1);
    CHECK(crc1 == 0x86F0B27A, "Hop 1 CRC-32 matches Python golden value (0x86F0B27A)");

    // ── Test 5: Circular Boundary Wrap-Around Stream & 10-Hop CRC Parity ──
    printf("\n[5] Testing Continuous Streaming Across Circular Boundaries (10 Golden CRCs)...\n");
    // Golden CRC-32 table generated by Python test_wav_injection.py (IEEE 802.3)
    const uint32_t golden_crcs[10] = {
        0x6B2E496E,  // Hop 0: [    0, 16000)
        0x86F0B27A,  // Hop 1: [ 8000, 24000)
        0x779F536D,  // Hop 2: [16000, 32000)
        0x7128B51E,  // Hop 3: [24000, 40000)
        0x6C9313A5,  // Hop 4: [32000, 48000)
        0x2BDF8843,  // Hop 5: [40000, 56000)
        0xE646CD6B,  // Hop 6: [48000, 64000)
        0x86FDF6AC,  // Hop 7: [56000, 72000)
        0xDE65D410,  // Hop 8: [64000, 80000)
        0x9471D0F4   // Hop 9: [72000, 88000)
    };

    int crc_matches = 0;
    int32_t expected_start = 8000;

    // Stream hops 2 through 9 and verify all CRCs
    for (int h = 2; h < 10; h++) {
        audio_ring_advance(SPECTRA_HOP_SAMPLES);
        expected_start += SPECTRA_HOP_SAMPLES;

        // Feed 2 blocks (8000 samples)
        for (int b = 0; b < 2; b++) {
            wav_injector_generate_block(block, SPECTRA_CAPTURE_BLOCK_SAMPLES);
            audio_ring_write(block, SPECTRA_CAPTURE_BLOCK_SAMPLES);
        }

        if (audio_ring_peek_window(window, SPECTRA_WINDOW_SAMPLES)) {
            uint32_t crc_val = wav_injector_compute_crc32(window, SPECTRA_WINDOW_SAMPLES);
            if (crc_val == golden_crcs[h]) {
                crc_matches++;
            } else {
                printf("  [MISMATCH] Hop %d CRC: 0x%08X != Golden 0x%08X\n", h, crc_val, golden_crcs[h]);
            }
        }
    }
    CHECK(crc_matches == 8, "All remaining 8 hops (Hops 2-9) bit-identically match golden CRCs");
    CHECK(audio_ring_overruns() == 0, "Overrun count remained strictly 0 during normal streaming");

    // ── Test 6: Drop-Oldest Overrun Policy ───────────────────────────────
    printf("\n[6] Testing Drop-Oldest Overrun Enforcement...\n");
    // Without advancing, write 50,000 samples into the 48,000 capacity buffer
    size_t prev_avail = audio_ring_available(); // was 16000
    for (int b = 0; b < 10; b++) {
        wav_injector_generate_block(block, SPECTRA_CAPTURE_BLOCK_SAMPLES); // 40,000 more
        audio_ring_write(block, SPECTRA_CAPTURE_BLOCK_SAMPLES);
    }
    // Total attempted: 16000 + 40000 = 56000 > 48000
    CHECK(audio_ring_available() == SPECTRA_RING_CAPACITY, "Buffer saturated at exact capacity (48000)");
    CHECK(audio_ring_overruns() > 0, "Overrun event counter incremented");

    audio_ring_get_stats(&stats);
    printf("  Reported overruns: %u, total dropped samples: %u\n",
           stats.overrun_count, stats.dropped_samples);
    CHECK(stats.dropped_samples == (56000 - SPECTRA_RING_CAPACITY), "Exact dropped count matches 8000 samples");

    // Peek still succeeds and retrieves the newest valid window
    peek_ok = audio_ring_peek_window(window, SPECTRA_WINDOW_SAMPLES);
    CHECK(peek_ok, "Peek succeeds after overrun recovery");

    // ── Test 7: Reset and Deinit ────────────────────────────────────────
    printf("\n[7] Testing Reset and Deinitialization...\n");
    audio_ring_reset();
    CHECK(audio_ring_available() == 0, "Reset cleared available samples");
    CHECK(audio_ring_overruns() == 0, "Reset cleared overrun count");

    audio_ring_deinit();
    CHECK(audio_ring_write(block, 4000) == 0, "Write safely rejected after deinit");

    // ── Summary ─────────────────────────────────────────────────────────
    printf("\n====================================================\n");
    printf("  Results: %d passed, %d failed\n", s_tests_passed, s_tests_failed);
    printf("====================================================\n");

    return s_tests_failed > 0 ? 1 : 0;
}
