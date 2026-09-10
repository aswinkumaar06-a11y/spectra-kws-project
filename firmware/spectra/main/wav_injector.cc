/**
 * wav_injector.cc — Deterministic Test Audio Injector Implementation
 */

#include "wav_injector.h"
#include "spectra_config.h"

#include <cmath>
#include <cstring>

static injector_mode_t s_mode = INJECTOR_MODE_RAMP;
static uint32_t s_sample_counter = 0;
static float s_freq_hz = 1000.0f;
static const int16_t* s_ext_buffer = NULL;
static size_t s_ext_buffer_len = 0;
static size_t s_ext_buffer_pos = 0;

void wav_injector_init(injector_mode_t mode, float frequency_hz) {
    s_mode = mode;
    s_freq_hz = frequency_hz > 0.0f ? frequency_hz : 1000.0f;
    s_sample_counter = 0;
    s_ext_buffer_pos = 0;
}

void wav_injector_set_buffer(const int16_t* buffer, size_t total_samples) {
    s_ext_buffer = buffer;
    s_ext_buffer_len = total_samples;
    s_ext_buffer_pos = 0;
}

size_t wav_injector_generate_block(int16_t* dest, size_t n_samples) {
    if (dest == NULL || n_samples == 0) {
        return 0;
    }

    if (s_mode == INJECTOR_MODE_RAMP) {
        for (size_t i = 0; i < n_samples; i++) {
            // Linear 16-bit ramp: wraps neatly within [-32768, 32767]
            dest[i] = (int16_t)(s_sample_counter & 0xFFFF);
            s_sample_counter++;
        }
    } else if (s_mode == INJECTOR_MODE_SINE) {
        const float sample_rate = (float)SPECTRA_SAMPLE_RATE;
        const float amplitude = 16384.0f; // -6 dBFS half-scale tone
        const float two_pi = 6.28318530717958647692f;

        for (size_t i = 0; i < n_samples; i++) {
            float t = (float)s_sample_counter / sample_rate;
            float val = amplitude * sinf(two_pi * s_freq_hz * t);
            dest[i] = (int16_t)val;
            s_sample_counter++;
        }
    } else if (s_mode == INJECTOR_MODE_BUFFER && s_ext_buffer != NULL && s_ext_buffer_len > 0) {
        for (size_t i = 0; i < n_samples; i++) {
            dest[i] = s_ext_buffer[s_ext_buffer_pos];
            s_ext_buffer_pos = (s_ext_buffer_pos + 1) % s_ext_buffer_len;
        }
    } else {
        memset(dest, 0, n_samples * sizeof(int16_t));
    }

    return n_samples;
}

uint32_t wav_injector_compute_crc32(const int16_t* data, size_t n_samples) {
    if (data == NULL || n_samples == 0) {
        return 0;
    }

    const uint8_t* byte_ptr = (const uint8_t*)data;
    size_t byte_len = n_samples * sizeof(int16_t);

    // Standard IEEE 802.3 CRC-32 (polynomial 0xEDB88320)
    uint32_t crc = 0xFFFFFFFF;
    for (size_t i = 0; i < byte_len; i++) {
        crc ^= byte_ptr[i];
        for (int b = 0; b < 8; b++) {
            crc = (crc >> 1) ^ (0xEDB88320 & (-(int32_t)(crc & 1)));
        }
    }
    return ~crc;
}

void wav_injector_reset(void) {
    s_sample_counter = 0;
    s_ext_buffer_pos = 0;
}
