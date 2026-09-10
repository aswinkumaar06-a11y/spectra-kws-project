/**
 * audio_capture.cc — I2S Microphone Driver Implementation
 *
 * Target: XIAO ESP32-C5 (RISC-V, ESP-IDF v5.5.2+)
 * Uses ESP-IDF standard I2S RX driver with DMA.
 */

#include "audio_capture.h"
#include "spectra_config.h"

#ifdef ESP_PLATFORM
#include "esp_log.h"
#include "driver/i2s_std.h"
#include "driver/gpio.h"
#include "freertos/FreeRTOS.h"

static const char* TAG = "AUDIO_CAPTURE";
static i2s_chan_handle_t s_rx_chan = NULL;
static bool s_running = false;

esp_err_t audio_capture_init(void) {
    if (s_rx_chan != NULL) {
        return ESP_OK;
    }

    ESP_LOGI(TAG, "Initializing I2S standard RX driver for INMP441...");
    ESP_LOGI(TAG, "Pinout: BCLK=GPIO%d, WS=GPIO%d, DIN=GPIO%d",
             SPECTRA_I2S_BCLK_GPIO, SPECTRA_I2S_WS_GPIO, SPECTRA_I2S_DIN_GPIO);

    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
    chan_cfg.dma_desc_num = 4;
    chan_cfg.dma_frame_num = 1000; // 1000 samples @ 16kHz = 62.5ms per DMA block
    chan_cfg.auto_clear = true;

    esp_err_t ret = i2s_new_channel(&chan_cfg, NULL, &s_rx_chan);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to create I2S RX channel (err=0x%x)", ret);
        return ret;
    }

    i2s_std_config_t std_cfg = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(SPECTRA_SAMPLE_RATE),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO),
        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = (gpio_num_t)SPECTRA_I2S_BCLK_GPIO,
            .ws   = (gpio_num_t)SPECTRA_I2S_WS_GPIO,
            .dout = I2S_GPIO_UNUSED,
            .din  = (gpio_num_t)SPECTRA_I2S_DIN_GPIO,
            .invert_flags = {
                .mclk_inv = false,
                .bclk_inv = false,
                .ws_inv   = false,
            },
        },
    };
    std_cfg.slot_cfg.slot_mask = I2S_STD_SLOT_LEFT; // INMP441 L/R pin tied to GND

    ret = i2s_channel_init_std_mode(s_rx_chan, &std_cfg);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to initialize I2S std mode (err=0x%x)", ret);
        i2s_del_channel(s_rx_chan);
        s_rx_chan = NULL;
        return ret;
    }

    ESP_LOGI(TAG, "I2S initialized successfully: %d Hz, 16-bit, Mono Left", SPECTRA_SAMPLE_RATE);
    return ESP_OK;
}

esp_err_t audio_capture_start(void) {
    if (s_rx_chan == NULL) {
        return ESP_ERR_INVALID_STATE;
    }
    if (s_running) {
        return ESP_OK;
    }
    esp_err_t ret = i2s_channel_enable(s_rx_chan);
    if (ret == ESP_OK) {
        s_running = true;
        ESP_LOGI(TAG, "I2S RX channel enabled");
    }
    return ret;
}

esp_err_t audio_capture_read(int16_t* dest, size_t samples_to_read, size_t* samples_read, uint32_t timeout_ms) {
    if (s_rx_chan == NULL || !s_running || dest == NULL) {
        return ESP_ERR_INVALID_STATE;
    }
    size_t bytes_read = 0;
    size_t bytes_to_read = samples_to_read * sizeof(int16_t);
    esp_err_t ret = i2s_channel_read(s_rx_chan, dest, bytes_to_read, &bytes_read, pdMS_TO_TICKS(timeout_ms));
    if (samples_read) {
        *samples_read = bytes_read / sizeof(int16_t);
    }
    return ret;
}

esp_err_t audio_capture_stop(void) {
    if (s_rx_chan == NULL || !s_running) {
        return ESP_OK;
    }
    esp_err_t ret = i2s_channel_disable(s_rx_chan);
    if (ret == ESP_OK) {
        s_running = false;
        ESP_LOGI(TAG, "I2S RX channel disabled");
    }
    return ret;
}

esp_err_t audio_capture_deinit(void) {
    if (s_rx_chan == NULL) {
        return ESP_OK;
    }
    if (s_running) {
        audio_capture_stop();
    }
    esp_err_t ret = i2s_del_channel(s_rx_chan);
    s_rx_chan = NULL;
    ESP_LOGI(TAG, "I2S RX driver deinitialized");
    return ret;
}

#else
// Host mock implementation for host testing
esp_err_t audio_capture_init(void) { return ESP_OK; }
esp_err_t audio_capture_start(void) { return ESP_OK; }
esp_err_t audio_capture_read(int16_t* dest, size_t samples_to_read, size_t* samples_read, uint32_t timeout_ms) {
    if (samples_read) *samples_read = 0;
    return ESP_OK;
}
esp_err_t audio_capture_stop(void) { return ESP_OK; }
esp_err_t audio_capture_deinit(void) { return ESP_OK; }
#endif
