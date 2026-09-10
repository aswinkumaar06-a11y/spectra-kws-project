/**
 * test_boot_guard_host.c — Spectra Boot Guard & Watchdog Host Unit Test
 *
 * Verifies:
 *   1. Normal boot transitions
 *   2. Consecutive crash accumulation (1, 2, 3)
 *   3. Safe Mode activation on >= 3 consecutive crashes
 *   4. Clean runtime recovery (mark_healthy resets counter)
 *   5. Task Watchdog Timer (TWDT) expiration and channel detection
 */

#include "boot_guard.h"
#include "watchdog.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
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
    printf("  Spectra Boot Guard & Watchdog Host Unit Test      \n");
    printf("====================================================\n");

    // ── Test 1: Boot Guard State Transitions ────────────────────────────
    printf("[1] Testing Boot-Loop Guard & Safe Mode Trigger...\n");
    boot_guard_reset_counter();

    // Boot 1
    boot_mode_t m1 = boot_guard_check();
    CHECK(m1 == BOOT_MODE_NORMAL, "Boot 1 enters BOOT_MODE_NORMAL");

    // Boot 2 (unclean crash)
    boot_mode_t m2 = boot_guard_check();
    CHECK(m2 == BOOT_MODE_NORMAL, "Boot 2 enters BOOT_MODE_NORMAL");

    // Boot 3 (unclean crash)
    boot_mode_t m3 = boot_guard_check();
    CHECK(m3 == BOOT_MODE_NORMAL, "Boot 3 enters BOOT_MODE_NORMAL (threshold reached)");

    // Boot 4 (3 previous crashes) -> must enter SAFE MODE
    boot_mode_t m4 = boot_guard_check();
    CHECK(m4 == BOOT_MODE_SAFE, "Boot 4 triggers SAFE MODE after 3 consecutive crashes");

    // Indicate visual alert
    boot_guard_indicate_safe_mode();

    // Boot 5 (Recovery: mark healthy resets crash counter)
    printf("\n[2] Testing Healthy Session Recovery...\n");
    boot_guard_mark_healthy();
    boot_guard_stats_t stats;
    boot_guard_get_stats(&stats);
    CHECK(stats.consecutive_crashes == 0, "mark_healthy reset consecutive crash counter to 0");
    CHECK(stats.is_marked_healthy, "Session marked healthy flag set");

    boot_mode_t m5 = boot_guard_check();
    CHECK(m5 == BOOT_MODE_NORMAL, "Subsequent boot returns to BOOT_MODE_NORMAL");

    // ── Test 3: Watchdog Timeout Verification ───────────────────────────
    printf("\n[3] Testing Task Watchdog Timer Expiration...\n");
    watchdog_init(1000); // 1.0s timeout
    watchdog_enable_channel(WATCHDOG_CHAN_AUDIO);
    watchdog_enable_channel(WATCHDOG_CHAN_INFER);

    // Initial feed at t = 0
    watchdog_feed(WATCHDOG_CHAN_AUDIO, 0);
    watchdog_feed(WATCHDOG_CHAN_INFER, 0);

    watchdog_channel_t expired_ch;
    bool ok = watchdog_check_all(500, &expired_ch);
    CHECK(ok, "Watchdog healthy at t = 500 ms (< 1000 ms timeout)");

    // Feed only audio at t = 600 ms, leave infer unfed
    watchdog_feed(WATCHDOG_CHAN_AUDIO, 600);

    // At t = 1100 ms: audio elapsed = 500 ms (healthy), infer elapsed = 1100 ms (expired!)
    ok = watchdog_check_all(1100, &expired_ch);
    CHECK(!ok, "Watchdog detects timeout at t = 1100 ms");
    CHECK(expired_ch == WATCHDOG_CHAN_INFER, "Expired channel correctly identified as WATCHDOG_CHAN_INFER");

    // Feed infer at t = 1150 ms -> recovered
    watchdog_feed(WATCHDOG_CHAN_INFER, 1150);
    ok = watchdog_check_all(1200, &expired_ch);
    CHECK(ok, "Watchdog recovers healthy state after feed");

    printf("\n====================================================\n");
    printf("  Results: %d passed, %d failed\n", g_passed, g_failed);
    printf("====================================================\n");

    return (g_failed == 0) ? 0 : 1;
}
