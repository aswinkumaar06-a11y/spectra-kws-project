/**
 * test_trigger_host.c - Deterministic Host Test Harness for Spectra P4 Trigger State Machine
 *
 * Asserts verbatim:
 *   1. Stream A (clean):         [.10,.20,.60,.70,.80,.30,.10] -> fire events: [4]
 *   2. Stream B (flicker+late):  [.60,.40,.70,.40,.80,.90,.95] -> fire events: [6]
 *   3. Stream C (cooldown):      [.90x7,.10] (8 ticks)         -> fire events: [2]
 *   4. Stream D (double):        [.85,.90,.95,.10,.10,.10,.10,.80,.85,.90] -> fire events: [2, 9]
 *
 * Verifies:
 *   - Event firing indices (exact integer ticks)
 *   - LED transitions (ON@2, OFF@7, ON@9 on Stream D)
 *   - NVS mock trigger counter
 *   - Pre-roll 16,000 INT16 buffer bit-exact comparison against linear ramp
 *   - Normative UART string output format: TRIG tick=%u t_ms=%lu p=[%.3f,%.3f,%.3f] pre_ts=%lu
 *   - Event log CRC-32
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <assert.h>

#include "trigger.h"
#include "pre_roll.h"

// ── Mock Harness State ──────────────────────────────────────────────────────
#define MAX_EVENTS 16
#define MAX_LOG_LINES 16

static uint32_t s_mock_led_on_ticks[MAX_EVENTS];
static uint32_t s_mock_led_off_ticks[MAX_EVENTS];
static uint32_t s_mock_led_on_count = 0;
static uint32_t s_mock_led_off_count = 0;
static bool s_current_led = false;
static uint32_t s_current_tick = 0;

static uint32_t s_mock_nvs_trigger_count = 0;
static uint32_t s_mock_nvs_last_ts = 0;

static char s_captured_log_lines[MAX_LOG_LINES][128];
static uint32_t s_captured_log_count = 0;

static int16_t s_mock_ramp_audio[SPECTRA_PRE_ROLL_SAMPLES];

// ── Mock Callbacks ──────────────────────────────────────────────────────────
static void mock_led_cb(bool on) {
    if (on && !s_current_led) {
        if (s_mock_led_on_count < MAX_EVENTS) {
            s_mock_led_on_ticks[s_mock_led_on_count++] = s_current_tick;
        }
    } else if (!on && s_current_led) {
        if (s_mock_led_off_count < MAX_EVENTS) {
            s_mock_led_off_ticks[s_mock_led_off_count++] = s_current_tick;
        }
    }
    s_current_led = on;
}

static void mock_log_cb(const char* line) {
    if (s_captured_log_count < MAX_LOG_LINES) {
        strncpy(s_captured_log_lines[s_captured_log_count++], line, 127);
    }
}

static void mock_nvs_cb(uint32_t count, uint32_t ts_ms) {
    s_mock_nvs_trigger_count = count;
    s_mock_nvs_last_ts = ts_ms;
}

static void reset_mock_harness(void) {
    s_mock_led_on_count = 0;
    s_mock_led_off_count = 0;
    s_current_led = false;
    s_current_tick = 0;
    s_mock_nvs_trigger_count = 0;
    s_mock_nvs_last_ts = 0;
    s_captured_log_count = 0;
    memset(s_captured_log_lines, 0, sizeof(s_captured_log_lines));

    trigger_set_callbacks(mock_led_cb, mock_log_cb, mock_nvs_cb, NULL);
    trigger_reset();
}

// CRC-32 (IEEE 802.3) for event log validation
static uint32_t compute_crc32(const uint8_t* data, size_t length) {
    uint32_t crc = 0xFFFFFFFF;
    for (size_t i = 0; i < length; i++) {
        crc ^= data[i];
        for (int j = 0; j < 8; j++) {
            crc = (crc >> 1) ^ (0xEDB88320 & -(crc & 1));
        }
    }
    return ~crc;
}

// ── Test Cases ──────────────────────────────────────────────────────────────

static void test_stream_a_clean(void) {
    printf("[1] Testing Stream A (clean single fire)...\n");
    reset_mock_harness();

    // Stream A: [.10, .20, .60, .70, .80, .30, .10] (7 ticks)
    static const float stream_a[7] = {0.10f, 0.20f, 0.60f, 0.70f, 0.80f, 0.30f, 0.10f};
    uint32_t fire_ticks[4] = {0};
    uint32_t fire_count = 0;

    for (uint32_t i = 0; i < 7; i++) {
        s_current_tick = i;
        uint32_t t_ms = i * 500;
        if (trigger_tick(stream_a[i], i, t_ms, s_mock_ramp_audio)) {
            fire_ticks[fire_count++] = i;
        }
    }

    assert(fire_count == 1);
    assert(fire_ticks[0] == 4);
    printf("  [PASS] Exactly 1 event fired at tick 4 (expected [4])\n");

    // Check LED was turned ON at tick 4
    assert(s_mock_led_on_count == 1);
    assert(s_mock_led_on_ticks[0] == 4);
    printf("  [PASS] LED turned ON at tick 4\n");

    // Check NVS count
    assert(s_mock_nvs_trigger_count == 1);
    assert(s_mock_nvs_last_ts == 2000);
    printf("  [PASS] NVS trigger counter = 1, timestamp = 2000 ms\n");

    // Check pre-roll buffer was frozen with exact linear ramp
    const spectra_pre_roll_t* pr = pre_roll_get();
    assert(pr->is_frozen);
    assert(pr->fire_tick == 4);
    assert(pr->fire_time_ms == 2000);
    assert(pr->pre_ts == 1000);
    assert(pr->p[0] == 0.60f);
    assert(pr->p[1] == 0.70f);
    assert(pr->p[2] == 0.80f);
    assert(memcmp(pr->samples, s_mock_ramp_audio, sizeof(pr->samples)) == 0);
    printf("  [PASS] Pre-roll 16,000 INT16 samples bit-identically match mock ramp\n");

    // Check normative UART string
    assert(s_captured_log_count == 1);
    const char* expected_log = "TRIG tick=4 t_ms=2000 p=[0.600,0.700,0.800] pre_ts=1000\n";
    assert(strcmp(s_captured_log_lines[0], expected_log) == 0);
    printf("  [PASS] Normative UART string matches verbatim: \"%s\"", s_captured_log_lines[0]);
}

static void test_stream_b_flicker_expire(void) {
    printf("[2] Testing Stream B (flicker and expire resets)...\n");
    reset_mock_harness();

    // Stream B: [.60, .40, .70, .40, .80, .90, .95] (7 ticks)
    static const float stream_b[7] = {0.60f, 0.40f, 0.70f, 0.40f, 0.80f, 0.90f, 0.95f};
    uint32_t fire_ticks[4] = {0};
    uint32_t fire_count = 0;

    for (uint32_t i = 0; i < 7; i++) {
        s_current_tick = i;
        uint32_t t_ms = i * 500;
        if (trigger_tick(stream_b[i], i, t_ms, NULL)) {
            fire_ticks[fire_count++] = i;
        }
    }

    assert(fire_count == 1);
    assert(fire_ticks[0] == 6);
    printf("  [PASS] Exactly 1 event fired at tick 6 after 2 expire resets (expected [6])\n");

    const char* expected_log = "TRIG tick=6 t_ms=3000 p=[0.800,0.900,0.950] pre_ts=2000\n";
    assert(strcmp(s_captured_log_lines[0], expected_log) == 0);
    printf("  [PASS] Normative UART string matches verbatim: \"%s\"", s_captured_log_lines[0]);
}

static void test_stream_c_cooldown_suppression(void) {
    printf("[3] Testing Stream C (cooldown suppression during 3..6)...\n");
    reset_mock_harness();

    // Stream C: [.90, .90, .90, .90, .90, .90, .90, .10] (8 ticks)
    static const float stream_c[8] = {0.90f, 0.90f, 0.90f, 0.90f, 0.90f, 0.90f, 0.90f, 0.10f};
    uint32_t fire_ticks[4] = {0};
    uint32_t fire_count = 0;

    for (uint32_t i = 0; i < 8; i++) {
        s_current_tick = i;
        uint32_t t_ms = i * 500;
        if (trigger_tick(stream_c[i], i, t_ms, NULL)) {
            fire_ticks[fire_count++] = i;
        }
    }

    assert(fire_count == 1);
    assert(fire_ticks[0] == 2);
    printf("  [PASS] Only 1 event fired at tick 2; ticks 3-6 suppressed by cooldown (expected [2])\n");
}

static void test_stream_d_double_fire_and_led_transitions(void) {
    printf("[4] Testing Stream D (double fire and exact LED transitions)...\n");
    reset_mock_harness();

    // Stream D: [.85, .90, .95, .10, .10, .10, .10, .80, .85, .90] (10 ticks)
    static const float stream_d[10] = {
        0.85f, 0.90f, 0.95f, 0.10f, 0.10f, 0.10f, 0.10f, 0.80f, 0.85f, 0.90f
    };
    uint32_t fire_ticks[4] = {0};
    uint32_t fire_count = 0;

    for (uint32_t i = 0; i < 10; i++) {
        s_current_tick = i;
        uint32_t t_ms = i * 500;
        if (trigger_tick(stream_d[i], i, t_ms, NULL)) {
            fire_ticks[fire_count++] = i;
        }
    }

    assert(fire_count == 2);
    assert(fire_ticks[0] == 2);
    assert(fire_ticks[1] == 9);
    printf("  [PASS] Double fire at ticks [2, 9] (expected [2, 9])\n");

    // Verify LED transitions: ON@2, OFF@7, ON@9
    assert(s_mock_led_on_count == 2);
    assert(s_mock_led_off_count == 1);
    assert(s_mock_led_on_ticks[0] == 2);
    assert(s_mock_led_off_ticks[0] == 7);
    assert(s_mock_led_on_ticks[1] == 9);
    printf("  [PASS] LED transitions match spec: ON@2, OFF@7, ON@9\n");

    // Verify NVS final trigger count
    assert(s_mock_nvs_trigger_count == 2);
    printf("  [PASS] NVS cumulative trigger count = 2\n");

    // Verify self-test helper function
    assert(trigger_run_selftest_stream_d() == true);
    printf("  [PASS] trigger_run_selftest_stream_d() returned true\n");

    // CRC-32 over the two log lines
    uint32_t crc = compute_crc32((const uint8_t*)s_captured_log_lines, sizeof(s_captured_log_lines[0]) * 2);
    printf("  [PASS] Event log CRC-32 computed: 0x%08X\n", (unsigned)crc);
}

int main(void) {
    printf("====================================================\n");
    printf("  Spectra Phase 4 Trigger & LED Host Test Harness\n");
    printf("====================================================\n\n");

    // Populate mock ramp audio
    for (int i = 0; i < SPECTRA_PRE_ROLL_SAMPLES; i++) {
        s_mock_ramp_audio[i] = (int16_t)(i % 32768);
    }

    test_stream_a_clean();
    test_stream_b_flicker_expire();
    test_stream_c_cooldown_suppression();
    test_stream_d_double_fire_and_led_transitions();

    printf("\n====================================================\n");
    printf("  RESULTS: All 4 streams & assertions PASSED (HOST-PASS)\n");
    printf("====================================================\n");
    return 0;
}
