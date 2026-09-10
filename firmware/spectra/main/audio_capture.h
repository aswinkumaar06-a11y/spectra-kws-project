/**
 * audio_capture.h — I2S Microphone Driver for Spectra KWS
 *
 * Target: XIAO ESP32-C5 (RISC-V, ESP-IDF v5.5.2+)
 * Sensor: INMP441 Omnidirectional MEMS Microphone (I2S standard)
 *
 * Pinout:
 *   - SCK / BCLK: GPIO 23 (D4)
 *   - WS  / LRCLK: GPIO 24 (D5)
 *   - SD  / DIN:   GPIO 11 (D6)
 *   - L/R:         GND (Left Channel)
 *   - VDD:         3.3V
 *   - GND:         GND
 */

#ifndef SPECTRA_AUDIO_CAPTURE_H_
#define SPECTRA_AUDIO_CAPTURE_H_

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef ESP_PLATFORM
#include "esp_err.h"
#else
typedef int esp_err_t;
#define ESP_OK 0
#define ESP_FAIL -1
#define ESP_ERR_INVALID_STATE -2
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Configure and initialize the I2S peripheral in standard RX mode.
 * Allocates DMA descriptors and buffers in internal SRAM.
 *
 * @return ESP_OK on success, error code otherwise.
 */
esp_err_t audio_capture_init(void);

/**
 * Enable/start the I2S RX channel DMA.
 */
esp_err_t audio_capture_start(void);

/**
 * Read 16-bit mono PCM samples from I2S DMA.
 *
 * @param dest Output buffer for int16_t samples
 * @param samples_to_read Number of samples requested (e.g. SPECTRA_CAPTURE_BLOCK_SAMPLES = 4000)
 * @param samples_read Output: actual number of samples read
 * @param timeout_ms Timeout in milliseconds (e.g. 500 ms)
 * @return ESP_OK on success, ESP_ERR_TIMEOUT or error code otherwise.
 */
esp_err_t audio_capture_read(int16_t* dest, size_t samples_to_read, size_t* samples_read, uint32_t timeout_ms);

/**
 * Stop/disable I2S RX channel DMA.
 */
esp_err_t audio_capture_stop(void);

/**
 * Deinitialize I2S peripheral and release channel & DMA resources.
 */
esp_err_t audio_capture_deinit(void);

#ifdef __cplusplus
}
#endif

#endif  // SPECTRA_AUDIO_CAPTURE_H_
