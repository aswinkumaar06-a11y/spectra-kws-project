/**
 * test_health_diag_host.c — Spectra Health & Diagnostic Telemetry Host Test
 */

#include "health_diag.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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
    printf("  Spectra Health Diagnostic Host Unit Test          \n");
    printf("====================================================\n");

    health_diag_init();

    health_stats_t st;
    health_diag_get_stats(&st);
    CHECK(st.total_inferences == 0, "Initial inference count is 0");
    CHECK(st.total_triggers == 0, "Initial trigger count is 0");

    // Simulate 10 inference hops with latency ~15,000 us
    for (int i = 1; i <= 10; i++) {
        health_diag_update(i * 500, 15000 + (i % 3) * 1000);
    }

    health_diag_record_trigger();
    health_diag_record_trigger();

    health_diag_get_stats(&st);
    CHECK(st.total_inferences == 10, "Recorded exactly 10 inferences");
    CHECK(st.total_triggers == 2, "Recorded exactly 2 triggers");
    CHECK(st.avg_inference_us >= 14000 && st.avg_inference_us <= 18000,
          "Exponential moving average latency within expected bounds");

    char summary[256];
    health_diag_format_summary(summary, sizeof(summary));
    printf("  Diagnostic Summary Line:\n    %s\n", summary);

    CHECK(strstr(summary, "HEALTH uptime=") != NULL, "Summary contains HEALTH uptime prefix");
    CHECK(strstr(summary, "inf_count=10") != NULL, "Summary reflects inf_count=10");
    CHECK(strstr(summary, "trigs=2") != NULL, "Summary reflects trigs=2");

    printf("\n====================================================\n");
    printf("  Results: %d passed, %d failed\n", g_passed, g_failed);
    printf("====================================================\n");

    return (g_failed == 0) ? 0 : 1;
}
