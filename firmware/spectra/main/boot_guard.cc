/**
 * boot_guard.cc — Spectra NVS Boot-Loop Guard & Safe Mode Implementation
 */

#include "boot_guard.h"
#include "spectra_config.h"
#include <cstdio>
#include <cstring>

#if defined(ESP_PLATFORM) && !defined(SPECTRA_HOST_TEST)
#include "nvs_flash.h"
#include "nvs.h"
#include "driver/gpio.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
static const char* TAG = "BOOT_GUARD";

static uint32_t s_total_boots = 0;
static uint32_t s_consecutive_crashes = 0;
static boot_mode_t s_current_mode = BOOT_MODE_NORMAL;
static bool s_marked_healthy = false;

boot_mode_t boot_guard_check(void) {
    nvs_handle_t handle;
    esp_err_t err = nvs_open("boot_guard", NVS_READWRITE, &handle);
    if (err == ESP_OK) {
        nvs_get_u32(handle, "tot_boots", &s_total_boots);
        nvs_get_u32(handle, "crashes", &s_consecutive_crashes);

        s_total_boots++;
        nvs_set_u32(handle, "tot_boots", s_total_boots);

        if (s_consecutive_crashes >= BOOT_GUARD_MAX_CRASHES) {
            s_current_mode = BOOT_MODE_SAFE;
            ESP_LOGE(TAG, "TRIPWIRE ACTIVATED: %u consecutive crashes detected! Entering SAFE MODE.",
                     (unsigned)s_consecutive_crashes);
        } else {
            s_current_mode = BOOT_MODE_NORMAL;
            s_consecutive_crashes++;
            nvs_set_u32(handle, "crashes", s_consecutive_crashes);
            ESP_LOGI(TAG, "Boot #%u, Crash count = %u (threshold = %u)",
                     (unsigned)s_total_boots, (unsigned)s_consecutive_crashes, BOOT_GUARD_MAX_CRASHES);
        }
        nvs_commit(handle);
        nvs_close(handle);
    } else {
        ESP_LOGW(TAG, "NVS unavailable for boot guard (err=0x%x). Defaulting to NORMAL.", err);
        s_current_mode = BOOT_MODE_NORMAL;
    }

    return s_current_mode;
}

void boot_guard_mark_healthy(void) {
    s_marked_healthy = true;
    s_consecutive_crashes = 0;

    nvs_handle_t handle;
    if (nvs_open("boot_guard", NVS_READWRITE, &handle) == ESP_OK) {
        nvs_set_u32(handle, "crashes", 0);
        nvs_commit(handle);
        nvs_close(handle);
        ESP_LOGI(TAG, "Boot session confirmed healthy — crash counter reset to 0.");
    }
}

void boot_guard_reset_counter(void) {
    s_consecutive_crashes = 0;
    nvs_handle_t handle;
    if (nvs_open("boot_guard", NVS_READWRITE, &handle) == ESP_OK) {
        nvs_set_u32(handle, "crashes", 0);
        nvs_commit(handle);
        nvs_close(handle);
    }
}

void boot_guard_indicate_safe_mode(void) {
    ESP_LOGW(TAG, "Signaling SAFE MODE on GPIO %d (5 rapid flashes)...", SPECTRA_LED_GPIO);
    for (int i = 0; i < 5; i++) {
        gpio_set_level((gpio_num_t)SPECTRA_LED_GPIO, 1);
        vTaskDelay(pdMS_TO_TICKS(100));
        gpio_set_level((gpio_num_t)SPECTRA_LED_GPIO, 0);
        vTaskDelay(pdMS_TO_TICKS(100));
    }
}

void boot_guard_get_stats(boot_guard_stats_t* stats) {
    if (stats == NULL) return;
    stats->total_boots = s_total_boots;
    stats->consecutive_crashes = s_consecutive_crashes;
    stats->current_mode = s_current_mode;
    stats->is_marked_healthy = s_marked_healthy;
}

#else
// ── Host Mock Implementation ────────────────────────────────────────────────
static uint32_t s_mock_total_boots = 0;
static uint32_t s_mock_crashes = 0;
static boot_mode_t s_mock_mode = BOOT_MODE_NORMAL;
static bool s_mock_healthy = false;

boot_mode_t boot_guard_check(void) {
    s_mock_total_boots++;
    s_mock_healthy = false;

    if (s_mock_crashes >= BOOT_GUARD_MAX_CRASHES) {
        s_mock_mode = BOOT_MODE_SAFE;
    } else {
        s_mock_mode = BOOT_MODE_NORMAL;
        s_mock_crashes++;
    }
    return s_mock_mode;
}

void boot_guard_mark_healthy(void) {
    s_mock_healthy = true;
    s_mock_crashes = 0;
}

void boot_guard_reset_counter(void) {
    s_mock_crashes = 0;
    s_mock_mode = BOOT_MODE_NORMAL;
}

void boot_guard_indicate_safe_mode(void) {
    printf("[BOOT_GUARD] Visual signal: SAFE MODE (5 rapid flashes)\n");
}

void boot_guard_get_stats(boot_guard_stats_t* stats) {
    if (stats == NULL) return;
    stats->total_boots = s_mock_total_boots;
    stats->consecutive_crashes = s_mock_crashes;
    stats->current_mode = s_mock_mode;
    stats->is_marked_healthy = s_mock_healthy;
}
#endif
