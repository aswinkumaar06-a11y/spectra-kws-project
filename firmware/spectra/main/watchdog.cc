/**
 * watchdog.cc — Spectra Task Watchdog Timer Implementation
 */

#include "watchdog.h"
#include <cstdio>
#include <cstring>

#ifdef ESP_PLATFORM
#include "esp_task_wdt.h"
#include "esp_log.h"
static const char* TAG = "WATCHDOG";
#else
#define ESP_LOGI(tag, fmt, ...) printf("[%s] " fmt "\n", tag, ##__VA_ARGS__)
#define ESP_LOGE(tag, fmt, ...) printf("[ERR][%s] " fmt "\n", tag, ##__VA_ARGS__)
#define ESP_LOGW(tag, fmt, ...) printf("[WARN][%s] " fmt "\n", tag, ##__VA_ARGS__)
static const char* TAG = "WATCHDOG";
#endif

static uint32_t s_timeout_ms = 3000;
static uint32_t s_feeds_count[WATCHDOG_NUM_CHANNELS] = {0};
static uint32_t s_last_feed_ms[WATCHDOG_NUM_CHANNELS] = {0};
static bool     s_channel_enabled[WATCHDOG_NUM_CHANNELS] = {false};
static bool     s_has_fed[WATCHDOG_NUM_CHANNELS] = {false};
static bool     s_initialized = false;

int watchdog_init(uint32_t timeout_ms) {
    s_timeout_ms = (timeout_ms > 0) ? timeout_ms : 3000;
    memset(s_feeds_count, 0, sizeof(s_feeds_count));
    memset(s_last_feed_ms, 0, sizeof(s_last_feed_ms));
    memset(s_channel_enabled, 0, sizeof(s_channel_enabled));
    memset(s_has_fed, 0, sizeof(s_has_fed));

#ifdef ESP_PLATFORM
    esp_task_wdt_config_t twdt_config = {
        .timeout_ms = s_timeout_ms,
        .idle_core_mask = (1 << 0),
        .trigger_panic = true
    };
    esp_err_t err = esp_task_wdt_reconfigure(&twdt_config);
    if (err != ESP_OK) {
        esp_task_wdt_init(&twdt_config);
    }
#endif

    s_initialized = true;
    ESP_LOGI(TAG, "Watchdog initialized with timeout = %u ms", (unsigned)s_timeout_ms);
    return 0;
}

void watchdog_enable_channel(watchdog_channel_t ch) {
    if (ch < WATCHDOG_NUM_CHANNELS) {
        s_channel_enabled[ch] = true;
    }
}

void watchdog_feed(watchdog_channel_t ch, uint32_t current_time_ms) {
    if (!s_initialized || ch >= WATCHDOG_NUM_CHANNELS) return;

    s_feeds_count[ch]++;
    s_last_feed_ms[ch] = current_time_ms;
    s_has_fed[ch] = true;

#ifdef ESP_PLATFORM
    esp_task_wdt_reset();
#endif
}

bool watchdog_check_all(uint32_t current_time_ms, watchdog_channel_t* out_expired_ch) {
    if (!s_initialized) return true;

    for (int ch = 0; ch < WATCHDOG_NUM_CHANNELS; ch++) {
        if (!s_channel_enabled[ch]) continue;

        // If channel was never fed, initialize baseline
        if (!s_has_fed[ch]) {
            s_has_fed[ch] = true;
            s_last_feed_ms[ch] = current_time_ms;
            continue;
        }

        uint32_t elapsed = current_time_ms - s_last_feed_ms[ch];
        if (elapsed > s_timeout_ms) {
            ESP_LOGE(TAG, "Channel %d watchdog TIMEOUT: %u ms without feed (timeout: %u ms)",
                     ch, (unsigned)elapsed, (unsigned)s_timeout_ms);
            if (out_expired_ch != NULL) {
                *out_expired_ch = (watchdog_channel_t)ch;
            }
            return false;
        }
    }

    return true;
}

void watchdog_get_stats(watchdog_stats_t* stats) {
    if (stats == NULL) return;
    stats->timeout_ms = s_timeout_ms;
    for (int i = 0; i < WATCHDOG_NUM_CHANNELS; i++) {
        stats->feeds_count[i] = s_feeds_count[i];
        stats->last_feed_ms[i] = s_last_feed_ms[i];
        stats->channel_enabled[i] = s_channel_enabled[i];
    }
}
