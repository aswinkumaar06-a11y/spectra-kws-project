/**
 * audio_ring.cc — Circular Audio Buffer Implementation
 *
 * Target: XIAO ESP32-C5 (PSRAM allocation: 8 MB external SPI RAM)
 * Also builds on host (x86/x64) for regression testing.
 */

#include "audio_ring.h"

#include <cstring>
#include <cstdlib>

#ifdef ESP_PLATFORM
#include "esp_log.h"
#include "esp_heap_caps.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
static const char* TAG = "AUDIO_RING";
#else
#include <cstdio>
#define ESP_LOGI(tag, fmt, ...) printf("[%s] " fmt "\n", tag, ##__VA_ARGS__)
#define ESP_LOGW(tag, fmt, ...) printf("[WARN][%s] " fmt "\n", tag, ##__VA_ARGS__)
#define ESP_LOGE(tag, fmt, ...) printf("[ERR][%s] " fmt "\n", tag, ##__VA_ARGS__)
static const char* TAG = "AUDIO_RING";
#endif

static int16_t* s_buffer = NULL;
static size_t s_capacity = 0;
static size_t s_head = 0;
static size_t s_tail = 0;
static size_t s_available = 0;
static uint32_t s_overrun_count = 0;
static uint32_t s_dropped_samples = 0;
static uint64_t s_total_samples_written = 0;
static bool s_is_psram = false;

#ifdef ESP_PLATFORM
static SemaphoreHandle_t s_mutex = NULL;
#define RING_LOCK()   do { if (s_mutex) xSemaphoreTake(s_mutex, portMAX_DELAY); } while (0)
#define RING_UNLOCK() do { if (s_mutex) xSemaphoreGive(s_mutex); } while (0)
#else
#define RING_LOCK()   do {} while (0)
#define RING_UNLOCK() do {} while (0)
#endif

int audio_ring_init(size_t capacity_samples) {
    if (capacity_samples == 0) {
        return -1;
    }

#ifdef ESP_PLATFORM
    if (s_mutex == NULL) {
        s_mutex = xSemaphoreCreateMutex();
    }
#endif

    RING_LOCK();

    // Free any existing allocation
    if (s_buffer != NULL) {
        free(s_buffer);
        s_buffer = NULL;
    }

    size_t alloc_bytes = capacity_samples * sizeof(int16_t);

#ifdef ESP_PLATFORM
    // Attempt PSRAM allocation first
    s_buffer = (int16_t*)heap_caps_malloc(alloc_bytes, MALLOC_CAP_SPIRAM);
    if (s_buffer != NULL) {
        s_is_psram = true;
        ESP_LOGI(TAG, "Allocated %u bytes (%u samples) in PSRAM",
                 (unsigned)alloc_bytes, (unsigned)capacity_samples);
    } else {
        ESP_LOGW(TAG, "PSRAM allocation failed, falling back to internal SRAM");
        s_buffer = (int16_t*)heap_caps_malloc(alloc_bytes, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
        s_is_psram = false;
        if (s_buffer != NULL) {
            ESP_LOGI(TAG, "Allocated %u bytes in internal SRAM", (unsigned)alloc_bytes);
        }
    }
#else
    s_buffer = (int16_t*)malloc(alloc_bytes);
    s_is_psram = false;
#endif

    if (s_buffer == NULL) {
        ESP_LOGE(TAG, "Failed to allocate %u bytes for audio ring", (unsigned)alloc_bytes);
        RING_UNLOCK();
        return -2;
    }

    memset(s_buffer, 0, alloc_bytes);
    s_capacity = capacity_samples;
    s_head = 0;
    s_tail = 0;
    s_available = 0;
    s_overrun_count = 0;
    s_dropped_samples = 0;
    s_total_samples_written = 0;

    RING_UNLOCK();
    return 0;
}

void audio_ring_deinit(void) {
    RING_LOCK();
    if (s_buffer != NULL) {
        free(s_buffer);
        s_buffer = NULL;
    }
    s_capacity = 0;
    s_head = 0;
    s_tail = 0;
    s_available = 0;
    s_is_psram = false;
    RING_UNLOCK();

#ifdef ESP_PLATFORM
    if (s_mutex != NULL) {
        vSemaphoreDelete(s_mutex);
        s_mutex = NULL;
    }
#endif
}

size_t audio_ring_write(const int16_t* src, size_t n_samples) {
    if (src == NULL || n_samples == 0 || s_buffer == NULL || s_capacity == 0) {
        return 0;
    }

    RING_LOCK();

    // If write size exceeds total buffer capacity, keep only the newest s_capacity samples
    if (n_samples > s_capacity) {
        size_t ignored = n_samples - s_capacity;
        src += ignored;
        n_samples = s_capacity;
        s_overrun_count++;
        s_dropped_samples += ignored;
        ESP_LOGW(TAG, "Overrun: write exceeds capacity! Dropped %u excess samples", (unsigned)ignored);
    }

    // Drop-oldest overrun policy: if write overflows remaining space, advance tail
    if (s_available + n_samples > s_capacity) {
        size_t overflow = (s_available + n_samples) - s_capacity;
        s_tail = (s_tail + overflow) % s_capacity;
        s_available -= overflow;
        s_overrun_count++;
        s_dropped_samples += overflow;
        ESP_LOGW(TAG, "Overrun: dropped %u oldest samples (event #%u, total dropped: %u)",
                 (unsigned)overflow, (unsigned)s_overrun_count, (unsigned)s_dropped_samples);
    }

    // Write samples in one or two contiguous chunks
    size_t first_chunk = (s_capacity - s_head < n_samples) ? (s_capacity - s_head) : n_samples;
    memcpy(&s_buffer[s_head], src, first_chunk * sizeof(int16_t));

    size_t second_chunk = n_samples - first_chunk;
    if (second_chunk > 0) {
        memcpy(&s_buffer[0], src + first_chunk, second_chunk * sizeof(int16_t));
    }

    s_head = (s_head + n_samples) % s_capacity;
    s_available += n_samples;
    s_total_samples_written += n_samples;

    RING_UNLOCK();
    return n_samples;
}

bool audio_ring_peek_window(int16_t* dst, size_t window_samples) {
    if (dst == NULL || window_samples == 0 || s_buffer == NULL) {
        return false;
    }

    RING_LOCK();
    if (s_available < window_samples) {
        RING_UNLOCK();
        return false;
    }

    // Copy window samples without advancing read pointer
    size_t first_chunk = (s_capacity - s_tail < window_samples) ? (s_capacity - s_tail) : window_samples;
    memcpy(dst, &s_buffer[s_tail], first_chunk * sizeof(int16_t));

    size_t second_chunk = window_samples - first_chunk;
    if (second_chunk > 0) {
        memcpy(dst + first_chunk, &s_buffer[0], second_chunk * sizeof(int16_t));
    }

    RING_UNLOCK();
    return true;
}

bool audio_ring_advance(size_t hop_samples) {
    if (hop_samples == 0 || s_buffer == NULL) {
        return false;
    }

    RING_LOCK();
    if (s_available < hop_samples) {
        // Not enough samples to advance full hop
        RING_UNLOCK();
        return false;
    }

    s_tail = (s_tail + hop_samples) % s_capacity;
    s_available -= hop_samples;

    RING_UNLOCK();
    return true;
}

size_t audio_ring_available(void) {
    RING_LOCK();
    size_t avail = s_available;
    RING_UNLOCK();
    return avail;
}

uint32_t audio_ring_overruns(void) {
    RING_LOCK();
    uint32_t count = s_overrun_count;
    RING_UNLOCK();
    return count;
}

void audio_ring_get_stats(audio_ring_stats_t* stats) {
    if (stats == NULL) return;
    RING_LOCK();
    stats->capacity_samples = s_capacity;
    stats->available_samples = s_available;
    stats->overrun_count = s_overrun_count;
    stats->dropped_samples = s_dropped_samples;
    stats->is_psram = s_is_psram;
    RING_UNLOCK();
}

void audio_ring_reset(void) {
    RING_LOCK();
    s_head = 0;
    s_tail = 0;
    s_available = 0;
    s_overrun_count = 0;
    s_dropped_samples = 0;
    s_total_samples_written = 0;
    if (s_buffer != NULL && s_capacity > 0) {
        memset(s_buffer, 0, s_capacity * sizeof(int16_t));
    }
    RING_UNLOCK();
}

size_t audio_ring_read_abs(uint64_t start_abs_idx, int16_t* dst, size_t n_samples) {
    if (dst == NULL || n_samples == 0 || s_buffer == NULL || s_capacity == 0) {
        return 0;
    }

    RING_LOCK();

    uint64_t total_written = s_total_samples_written;
    uint64_t oldest_avail = (total_written > s_capacity) ? (total_written - s_capacity) : 0;

    // If requested range is entirely ahead of write pointer (underflow)
    if (start_abs_idx >= total_written) {
        RING_UNLOCK();
        return 0;
    }

    // If requested start is older than resident circular history, skip stale samples
    uint64_t read_start = start_abs_idx;
    if (read_start < oldest_avail) {
        read_start = oldest_avail;
    }

    // Number of resident samples available from read_start
    uint64_t avail_from_start = total_written - read_start;
    size_t samples_to_read = (n_samples < avail_from_start) ? n_samples : (size_t)avail_from_start;

    if (samples_to_read == 0) {
        RING_UNLOCK();
        return 0;
    }

    // Map read_start to ring buffer index
    size_t buf_idx = (size_t)(read_start % s_capacity);
    size_t first_chunk = (s_capacity - buf_idx < samples_to_read) ? (s_capacity - buf_idx) : samples_to_read;
    memcpy(dst, &s_buffer[buf_idx], first_chunk * sizeof(int16_t));

    size_t second_chunk = samples_to_read - first_chunk;
    if (second_chunk > 0) {
        memcpy(dst + first_chunk, &s_buffer[0], second_chunk * sizeof(int16_t));
    }

    RING_UNLOCK();
    return samples_to_read;
}

uint64_t audio_ring_get_total_written(void) {
    RING_LOCK();
    uint64_t total = s_total_samples_written;
    RING_UNLOCK();
    return total;
}
