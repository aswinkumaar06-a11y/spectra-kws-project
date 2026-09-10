/**
 * power_mgr.cc — Spectra Power Optimization Implementation
 */

#include "power_mgr.h"
#include <cstdio>
#include <cstring>

#ifdef ESP_PLATFORM
#include "esp_pm.h"
#include "esp_sleep.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
static const char* TAG = "POWER_MGR";

static esp_pm_lock_handle_t s_perf_lock = NULL;
static uint32_t s_active_locks = 0;
static bool s_light_sleep_enabled = false;

int power_mgr_init(bool light_sleep_enable) {
    s_light_sleep_enabled = light_sleep_enable;

    esp_pm_config_t pm_config = {
        .max_freq_mhz = 240,
        .min_freq_mhz = 80,
        .light_sleep_enable = light_sleep_enable
    };

    esp_err_t err = esp_pm_configure(&pm_config);
    if (err != ESP_OK) {
        ESP_LOGW(TAG, "esp_pm_configure failed (0x%x). DFS may be unavailable in this config.", err);
    }

    if (s_perf_lock == NULL) {
        esp_pm_lock_create(ESP_PM_CPU_FREQ_MAX, 0, "spectra_perf", &s_perf_lock);
    }

    ESP_LOGI(TAG, "Power Manager initialized: Max=%u MHz, Min=%u MHz, LightSleep=%s",
             pm_config.max_freq_mhz, pm_config.min_freq_mhz,
             light_sleep_enable ? "ENABLED" : "DISABLED");

    return (err == ESP_OK) ? 0 : -1;
}

void power_lock_acquire(power_client_t client) {
    if (s_perf_lock != NULL && s_active_locks == 0) {
        esp_pm_lock_acquire(s_perf_lock);
    }
    s_active_locks |= (uint32_t)client;
}

void power_lock_release(power_client_t client) {
    s_active_locks &= ~(uint32_t)client;
    if (s_perf_lock != NULL && s_active_locks == 0) {
        esp_pm_lock_release(s_perf_lock);
    }
}

uint32_t power_mgr_get_freq_mhz(void) {
    return (s_active_locks > 0) ? 240 : 80;
}

bool power_mgr_is_high_perf(void) {
    return (s_active_locks > 0);
}

void power_mgr_idle_sleep(uint32_t sleep_ms) {
    if (sleep_ms == 0) return;
    vTaskDelay(pdMS_TO_TICKS(sleep_ms));
}

void power_mgr_get_stats(power_mgr_stats_t* stats) {
    if (stats == NULL) return;
    stats->max_freq_mhz = 240;
    stats->min_freq_mhz = 80;
    stats->light_sleep_enable = s_light_sleep_enabled;
    stats->active_locks = s_active_locks;
}

#else
// ── Host Mock Implementation ────────────────────────────────────────────────
static uint32_t s_active_locks = 0;
static bool s_light_sleep_enabled = false;

int power_mgr_init(bool light_sleep_enable) {
    s_light_sleep_enabled = light_sleep_enable;
    s_active_locks = 0;
    return 0;
}

void power_lock_acquire(power_client_t client) {
    s_active_locks |= (uint32_t)client;
}

void power_lock_release(power_client_t client) {
    s_active_locks &= ~(uint32_t)client;
}

uint32_t power_mgr_get_freq_mhz(void) {
    return (s_active_locks > 0) ? 240 : 80;
}

bool power_mgr_is_high_perf(void) {
    return (s_active_locks > 0);
}

void power_mgr_idle_sleep(uint32_t sleep_ms) {
    (void)sleep_ms;
}

void power_mgr_get_stats(power_mgr_stats_t* stats) {
    if (stats == NULL) return;
    stats->max_freq_mhz = 240;
    stats->min_freq_mhz = 80;
    stats->light_sleep_enable = s_light_sleep_enabled;
    stats->active_locks = s_active_locks;
}
#endif
