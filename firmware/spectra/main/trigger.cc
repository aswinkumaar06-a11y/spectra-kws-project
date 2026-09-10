/**
 * trigger.cc - Spectra KWS Confidence Trigger Implementation
 *
 * Deterministic integer-tick trigger state machine with NVS counters,
 * pre-roll freeze, LED gating, and normative UART logging.
 */

#include "trigger.h"
#include <stdio.h>
#include <string.h>

#ifndef SPECTRA_HOST_TEST
#include "esp_log.h"
#include "esp_timer.h"
#include "driver/gpio.h"
#include "nvs_flash.h"
#include "nvs.h"
static const char* TAG = "TRIGGER";
#endif

// ── Static Pre-Roll Storage (Internal SRAM BSS, 32 KB) ──────────────────────
static spectra_pre_roll_t s_pre_roll;

void pre_roll_init(void) {
    memset(&s_pre_roll, 0, sizeof(s_pre_roll));
}

void pre_roll_freeze(uint32_t fire_tick,
                     uint32_t fire_time_ms,
                     const float p_history[3],
                     const int16_t* src_16k,
                     uint32_t pre_ts) {
    s_pre_roll.fire_tick = fire_tick;
    s_pre_roll.fire_time_ms = fire_time_ms;
    s_pre_roll.pre_ts = pre_ts;
    if (p_history != NULL) {
        s_pre_roll.p[0] = p_history[0];
        s_pre_roll.p[1] = p_history[1];
        s_pre_roll.p[2] = p_history[2];
    }
    if (src_16k != NULL) {
        memcpy(s_pre_roll.samples, src_16k, sizeof(s_pre_roll.samples));
    }
    s_pre_roll.is_frozen = true;
}

const spectra_pre_roll_t* pre_roll_get(void) {
    return &s_pre_roll;
}

// ── Trigger Context & State Machine ─────────────────────────────────────────
static trigger_context_t s_ctx;

// Default callback handlers
static void default_led_cb(bool on) {
#ifndef SPECTRA_HOST_TEST
    gpio_set_level((gpio_num_t)SPECTRA_LED_GPIO, on ? 1 : 0);
#else
    (void)on;
#endif
}

static void default_log_cb(const char* line) {
#ifndef SPECTRA_HOST_TEST
    printf("%s", line);
#else
    printf("%s", line);
#endif
}

static void default_nvs_cb(uint32_t count, uint32_t ts_ms) {
#ifndef SPECTRA_HOST_TEST
    nvs_handle_t nvs_handle;
    esp_err_t err = nvs_open("spectra", NVS_READWRITE, &nvs_handle);
    if (err == ESP_OK) {
        nvs_set_u32(nvs_handle, "trigger_count", count);
        nvs_set_u32(nvs_handle, "last_trig_ms", ts_ms);
        nvs_commit(nvs_handle);
        nvs_close(nvs_handle);
    }
#else
    (void)count;
    (void)ts_ms;
#endif
}

static void default_freeze_cb(uint32_t fire_tick,
                              uint32_t t_ms,
                              const float p_history[3],
                              uint32_t pre_ts) {
    pre_roll_freeze(fire_tick, t_ms, p_history, NULL, pre_ts);
}

static trigger_led_cb_t s_led_cb = default_led_cb;
static trigger_log_cb_t s_log_cb = default_log_cb;
static trigger_nvs_cb_t s_nvs_cb = default_nvs_cb;
static trigger_freeze_cb_t s_freeze_cb = default_freeze_cb;

void trigger_set_callbacks(trigger_led_cb_t led_cb,
                           trigger_log_cb_t log_cb,
                           trigger_nvs_cb_t nvs_cb,
                           trigger_freeze_cb_t freeze_cb) {
    s_led_cb = (led_cb != NULL) ? led_cb : default_led_cb;
    s_log_cb = (log_cb != NULL) ? log_cb : default_log_cb;
    s_nvs_cb = (nvs_cb != NULL) ? nvs_cb : default_nvs_cb;
    s_freeze_cb = (freeze_cb != NULL) ? freeze_cb : default_freeze_cb;
}

const trigger_context_t* trigger_get_context(void) {
    return &s_ctx;
}

void trigger_reset(void) {
    s_ctx.state = TRIGGER_STATE_LISTEN;
    s_ctx.candidate_count = 0;
    s_ctx.cooldown_ticks_left = 0;
    s_ctx.total_triggers = 0;
    s_ctx.led_state = false;
    s_ctx.p_history[0] = 0.0f;
    s_ctx.p_history[1] = 0.0f;
    s_ctx.p_history[2] = 0.0f;
    s_ctx.last_log_line[0] = '\0';
    s_led_cb(false);
}

void trigger_init(void) {
    memset(&s_ctx, 0, sizeof(s_ctx));
    pre_roll_init();
    trigger_reset();

#ifndef SPECTRA_HOST_TEST
    // Initialize NVS
    esp_err_t err = nvs_flash_init();
    if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        nvs_flash_erase();
        nvs_flash_init();
    }

    nvs_handle_t nvs_handle;
    err = nvs_open("spectra", NVS_READWRITE, &nvs_handle);
    if (err == ESP_OK) {
        uint32_t boots = 0;
        nvs_get_u32(nvs_handle, "boot_count", &boots);
        boots++;
        nvs_set_u32(nvs_handle, "boot_count", boots);
        s_ctx.boot_count = boots;

        uint32_t trig = 0;
        nvs_get_u32(nvs_handle, "trigger_count", &trig);
        s_ctx.total_triggers = trig;

        uint32_t last_ms = 0;
        nvs_get_u32(nvs_handle, "last_trig_ms", &last_ms);
        s_ctx.last_trigger_ts_ms = last_ms;

        nvs_commit(nvs_handle);
        nvs_close(nvs_handle);

        ESP_LOGI(TAG, "NVS loaded: boot_count=%u, trigger_count=%u, last_trig_ms=%u",
                 (unsigned)s_ctx.boot_count,
                 (unsigned)s_ctx.total_triggers,
                 (unsigned)s_ctx.last_trigger_ts_ms);
    } else {
        ESP_LOGW(TAG, "NVS open failed: %d (counters volatile in RAM)", err);
    }
#endif
}

bool trigger_tick(float p_pos, uint32_t tick, uint32_t t_ms, const int16_t* src_window_16k) {
    // Shift rolling probability history [p_oldest, p_mid, p_newest]
    s_ctx.p_history[0] = s_ctx.p_history[1];
    s_ctx.p_history[1] = s_ctx.p_history[2];
    s_ctx.p_history[2] = p_pos;

    // ── State: COOLDOWN ─────────────────────────────────────────────────────
    if (s_ctx.state == TRIGGER_STATE_COOLDOWN) {
        if (s_ctx.cooldown_ticks_left > 0) {
            s_ctx.cooldown_ticks_left--;
            // Still in cooldown during this tick; probabilities are strictly ignored
            return false;
        }

        // Cooldown has completed: turn LED OFF and return fresh to LISTEN
        s_ctx.state = TRIGGER_STATE_LISTEN;
        s_ctx.candidate_count = 0;
        if (s_ctx.led_state) {
            s_ctx.led_state = false;
            s_led_cb(false);
        }
        // Fall through to process this tick in LISTEN state
    }

    // ── Check Threshold ─────────────────────────────────────────────────────
    bool above_thresh = (p_pos > SPECTRA_TRIGGER_THRESHOLD);

    if (s_ctx.state == TRIGGER_STATE_LISTEN) {
        if (above_thresh) {
            s_ctx.state = TRIGGER_STATE_CANDIDATE;
            s_ctx.candidate_count = 1;
        }
        return false;
    }

    if (s_ctx.state == TRIGGER_STATE_CANDIDATE) {
        if (above_thresh) {
            s_ctx.candidate_count++;
            if (s_ctx.candidate_count >= SPECTRA_TRIGGER_N_FIRE) {
                // ── FIRE ACTIONS ────────────────────────────────────────────
                s_ctx.state = TRIGGER_STATE_TRIGGERED;
                s_ctx.total_triggers++;
                s_ctx.last_trigger_ts_ms = t_ms;

                // 1. LED ON (remains ON through end of cooldown)
                s_ctx.led_state = true;
                s_led_cb(true);

                // 2. Format normative UART log line
                // Format: TRIG tick=%u t_ms=%lu p=[%.3f,%.3f,%.3f] pre_ts=%lu
                uint32_t pre_ts = (t_ms >= 1000) ? (t_ms - 1000) : 0;
                snprintf(s_ctx.last_log_line, sizeof(s_ctx.last_log_line),
                         "TRIG tick=%u t_ms=%lu p=[%.3f,%.3f,%.3f] pre_ts=%lu\n",
                         (unsigned)tick,
                         (unsigned long)t_ms,
                         s_ctx.p_history[0], s_ctx.p_history[1], s_ctx.p_history[2],
                         (unsigned long)pre_ts);
                s_log_cb(s_ctx.last_log_line);

                // 3. Persistent NVS increment
                s_nvs_cb(s_ctx.total_triggers, t_ms);

                // 4. Pre-roll audio window freeze
                if (src_window_16k != NULL) {
                    pre_roll_freeze(tick, t_ms, s_ctx.p_history, src_window_16k, pre_ts);
                } else {
                    s_freeze_cb(tick, t_ms, s_ctx.p_history, pre_ts);
                }

                // 5. Enter COOLDOWN for 4 ticks (2.0s)
                s_ctx.state = TRIGGER_STATE_COOLDOWN;
                s_ctx.cooldown_ticks_left = SPECTRA_TRIGGER_COOLDOWN_TICKS;
                return true;
            }
        } else {
            // Expire path: drop below threshold resets count and returns to LISTEN
            s_ctx.state = TRIGGER_STATE_LISTEN;
            s_ctx.candidate_count = 0;
        }
        return false;
    }

    return false;
}

bool trigger_run_selftest_stream_d(void) {
    // Stream D vectors: [.85,.90,.95,.10,.10,.10,.10,.80,.85,.90]
    // Expected fire ticks: [2, 9]
    static const float stream_d[10] = {
        0.85f, 0.90f, 0.95f, 0.10f, 0.10f, 0.10f, 0.10f, 0.80f, 0.85f, 0.90f
    };

    trigger_context_t saved_ctx = s_ctx;
    trigger_reset();

    uint32_t fired_ticks[4] = {0};
    uint32_t fire_count = 0;

    for (uint32_t tick = 0; tick < 10; tick++) {
        uint32_t t_ms = tick * 500;
        if (trigger_tick(stream_d[tick], tick, t_ms, NULL)) {
            if (fire_count < 4) {
                fired_ticks[fire_count++] = tick;
            }
        }
    }

    // Restore context
    s_ctx = saved_ctx;

    bool pass = (fire_count == 2) && (fired_ticks[0] == 2) && (fired_ticks[1] == 9);
    return pass;
}
