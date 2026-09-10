/**
 * streamer.cc — Spectra Audio Streamer & ASR Client Implementation
 */

#include "streamer.h"
#include "netwrap.h"
#include <cstring>
#include <cstdio>
#include <cstdlib>

#ifdef ESP_PLATFORM
#include "esp_log.h"
static const char* TAG = "STREAMER";
#else
#define ESP_LOGI(tag, fmt, ...) printf("[%s] " fmt "\n", tag, ##__VA_ARGS__)
#define ESP_LOGW(tag, fmt, ...) printf("[WARN][%s] " fmt "\n", tag, ##__VA_ARGS__)
#define ESP_LOGE(tag, fmt, ...) printf("[ERR][%s] " fmt "\n", tag, ##__VA_ARGS__)
static const char* TAG = "STREAMER";
#endif

void streamer_init(streamer_context_t* ctx, const streamer_config_t* cfg) {
    if (ctx == NULL) return;

    memset(ctx, 0, sizeof(streamer_context_t));
    ctx->state = STREAMER_IDLE;
    ctx->sock_fd = -1;

    float vad_th = (cfg && cfg->vad_threshold > 0.0f) ? cfg->vad_threshold : SPECTRA_VAD_THRESHOLD;
    vad_init(&ctx->vad, vad_th);
}

bool streamer_start(streamer_context_t* ctx, const char* host, uint16_t port,
                    uint32_t utt_id, const int16_t* pre_roll, size_t pre_roll_samples) {
    if (ctx == NULL || host == NULL || port == 0) return false;

    streamer_close(ctx);

    ctx->sock_fd = netwrap_connect(host, port);
    if (ctx->sock_fd < 0) {
        ESP_LOGE(TAG, "Failed to connect to ASR server %s:%u", host, port);
        ctx->state = STREAMER_ERROR;
        return false;
    }

    ctx->current_utt_id = utt_id;
    ctx->seq_num = 0;
    ctx->chunk_seq = 0;
    ctx->total_samples_sent = 0;
    ctx->has_final = false;
    ctx->last_transcript[0] = '\0';
    vad_reset(&ctx->vad);

    // 1. Send STREAM_MSG_UTT_START
    stream_payload_start_t start_pl;
    start_pl.utt_id = utt_id;
    start_pl.sample_rate = SPECTRA_SAMPLE_RATE;
    start_pl.channels = 1;
    start_pl.bits_per_sample = 16;

    stream_header_t hdr;
    stream_build_header(&hdr, STREAM_MSG_UTT_START, ctx->seq_num++,
                        (const uint8_t*)&start_pl, sizeof(start_pl));

    if (netwrap_send_exact(ctx->sock_fd, &hdr, sizeof(hdr)) < 0 ||
        netwrap_send_exact(ctx->sock_fd, &start_pl, sizeof(start_pl)) < 0) {
        ESP_LOGE(TAG, "Failed to send UTT_START header/payload");
        streamer_close(ctx);
        ctx->state = STREAMER_ERROR;
        return false;
    }

    ESP_LOGI(TAG, "Started utterance #%u to %s:%u", (unsigned)utt_id, host, port);

    // 2. Send Pre-roll PCM buffer (e.g. 1.0s = 16,000 samples)
    if (pre_roll != NULL && pre_roll_samples > 0) {
        stream_payload_pcm_header_t pcm_hdr;
        pcm_hdr.utt_id = utt_id;
        pcm_hdr.chunk_seq = ctx->chunk_seq++;
        pcm_hdr.sample_count = (uint16_t)pre_roll_samples;

        uint16_t payload_len = (uint16_t)(sizeof(pcm_hdr) + pre_roll_samples * sizeof(int16_t));
        
        // Allocate temporary payload buffer for CRC and send
        uint8_t* pcm_payload = (uint8_t*)malloc(payload_len);
        if (pcm_payload != NULL) {
            memcpy(pcm_payload, &pcm_hdr, sizeof(pcm_hdr));
            memcpy(pcm_payload + sizeof(pcm_hdr), pre_roll, pre_roll_samples * sizeof(int16_t));

            stream_build_header(&hdr, STREAM_MSG_PCM_BIN, ctx->seq_num++, pcm_payload, payload_len);

            netwrap_send_exact(ctx->sock_fd, &hdr, sizeof(hdr));
            netwrap_send_exact(ctx->sock_fd, pcm_payload, payload_len);
            free(pcm_payload);

            ctx->total_samples_sent += pre_roll_samples;
            ESP_LOGI(TAG, "Sent %u pre-roll samples", (unsigned)pre_roll_samples);
        }
    }

    ctx->state = STREAMER_STREAMING;
    return true;
}

bool streamer_feed_chunk(streamer_context_t* ctx, const int16_t* pcm, size_t n_samples) {
    if (ctx == NULL || pcm == NULL || n_samples == 0) return false;
    if (ctx->state != STREAMER_STREAMING || ctx->sock_fd < 0) return false;

    // Send PCM chunk frame
    stream_payload_pcm_header_t pcm_hdr;
    pcm_hdr.utt_id = ctx->current_utt_id;
    pcm_hdr.chunk_seq = ctx->chunk_seq++;
    pcm_hdr.sample_count = (uint16_t)n_samples;

    uint16_t payload_len = (uint16_t)(sizeof(pcm_hdr) + n_samples * sizeof(int16_t));
    uint8_t* pcm_payload = (uint8_t*)malloc(payload_len);
    if (pcm_payload == NULL) return false;

    memcpy(pcm_payload, &pcm_hdr, sizeof(pcm_hdr));
    memcpy(pcm_payload + sizeof(pcm_hdr), pcm, n_samples * sizeof(int16_t));

    stream_header_t hdr;
    stream_build_header(&hdr, STREAM_MSG_PCM_BIN, ctx->seq_num++, pcm_payload, payload_len);

    int err1 = netwrap_send_exact(ctx->sock_fd, &hdr, sizeof(hdr));
    int err2 = netwrap_send_exact(ctx->sock_fd, pcm_payload, payload_len);
    free(pcm_payload);

    if (err1 < 0 || err2 < 0) {
        ESP_LOGE(TAG, "Socket write error while sending PCM chunk");
        ctx->state = STREAMER_ERROR;
        return false;
    }

    ctx->total_samples_sent += n_samples;

    // Evaluate VAD state
    vad_state_t vad_st = vad_process_frame(&ctx->vad, pcm);
    if (vad_st == VAD_STATE_END) {
        ESP_LOGI(TAG, "VAD endpoint detected after %u samples — ending utterance",
                 (unsigned)ctx->total_samples_sent);

        // Send STREAM_MSG_UTT_END
        stream_payload_end_t end_pl;
        end_pl.utt_id = ctx->current_utt_id;
        end_pl.total_samples = ctx->total_samples_sent;
        end_pl.reason = (uint8_t)STREAM_END_VAD_ENDPOINT;

        stream_build_header(&hdr, STREAM_MSG_UTT_END, ctx->seq_num++,
                            (const uint8_t*)&end_pl, sizeof(end_pl));

        netwrap_send_exact(ctx->sock_fd, &hdr, sizeof(hdr));
        netwrap_send_exact(ctx->sock_fd, &end_pl, sizeof(end_pl));

        ctx->end_reason = STREAM_END_VAD_ENDPOINT;
        ctx->state = STREAMER_WAITING_FINAL;
    }

    return true;
}

void streamer_abort(streamer_context_t* ctx) {
    if (ctx == NULL || ctx->sock_fd < 0) return;

    if (ctx->state == STREAMER_STREAMING) {
        ESP_LOGW(TAG, "Aborting streaming on user request");
        stream_payload_end_t end_pl;
        end_pl.utt_id = ctx->current_utt_id;
        end_pl.total_samples = ctx->total_samples_sent;
        end_pl.reason = (uint8_t)STREAM_END_ABORT;

        stream_header_t hdr;
        stream_build_header(&hdr, STREAM_MSG_UTT_END, ctx->seq_num++,
                            (const uint8_t*)&end_pl, sizeof(end_pl));

        netwrap_send_exact(ctx->sock_fd, &hdr, sizeof(hdr));
        netwrap_send_exact(ctx->sock_fd, &end_pl, sizeof(end_pl));

        ctx->end_reason = STREAM_END_ABORT;
        ctx->state = STREAMER_WAITING_FINAL;
    }
}

bool streamer_poll(streamer_context_t* ctx, int timeout_ms) {
    if (ctx == NULL || ctx->sock_fd < 0) return false;

    stream_header_t hdr;
    int r = netwrap_recv_exact(ctx->sock_fd, &hdr, sizeof(hdr), timeout_ms);
    if (r <= 0) {
        return false; // Nothing received or timeout
    }

    if (hdr.magic[0] != STREAM_MAGIC_0 || hdr.magic[1] != STREAM_MAGIC_1) {
        ESP_LOGE(TAG, "Protocol error: invalid magic 0x%02X 0x%02X", hdr.magic[0], hdr.magic[1]);
        ctx->state = STREAMER_ERROR;
        return false;
    }

    uint8_t* payload = NULL;
    if (hdr.payload_len > 0) {
        payload = (uint8_t*)malloc(hdr.payload_len);
        if (payload == NULL) return false;
        if (netwrap_recv_exact(ctx->sock_fd, payload, hdr.payload_len, timeout_ms) <= 0) {
            free(payload);
            ctx->state = STREAMER_ERROR;
            return false;
        }
    }

    if (!stream_verify_header(&hdr, payload)) {
        ESP_LOGE(TAG, "CRC-16 mismatch on received message type 0x%02X", hdr.msg_type);
        if (payload) free(payload);
        ctx->state = STREAMER_ERROR;
        return false;
    }

    if (hdr.msg_type == STREAM_MSG_PARTIAL && payload != NULL) {
        stream_payload_partial_t* part = (stream_payload_partial_t*)payload;
        size_t text_bytes = (hdr.payload_len > sizeof(stream_payload_partial_t))
                            ? (hdr.payload_len - sizeof(stream_payload_partial_t)) : 0;
        size_t copy_len = (text_bytes < sizeof(ctx->last_transcript) - 1)
                          ? text_bytes : (sizeof(ctx->last_transcript) - 1);
        memcpy(ctx->last_transcript, payload + sizeof(stream_payload_partial_t), copy_len);
        ctx->last_transcript[copy_len] = '\0';
        ESP_LOGI(TAG, "Partial: \"%s\"", ctx->last_transcript);
    } else if (hdr.msg_type == STREAM_MSG_FINAL && payload != NULL) {
        stream_payload_final_t* fin = (stream_payload_final_t*)payload;
        ctx->latency_ms = fin->latency_ms;
        size_t text_bytes = (hdr.payload_len > sizeof(stream_payload_final_t))
                            ? (hdr.payload_len - sizeof(stream_payload_final_t)) : 0;
        size_t copy_len = (text_bytes < sizeof(ctx->last_transcript) - 1)
                          ? text_bytes : (sizeof(ctx->last_transcript) - 1);
        memcpy(ctx->last_transcript, payload + sizeof(stream_payload_final_t), copy_len);
        ctx->last_transcript[copy_len] = '\0';
        ctx->has_final = true;
        ctx->state = STREAMER_COMPLETED;
        ESP_LOGI(TAG, "FINAL (latency=%u ms): \"%s\"", (unsigned)ctx->latency_ms, ctx->last_transcript);
    } else if (hdr.msg_type == STREAM_MSG_ERROR) {
        ESP_LOGE(TAG, "Server error received");
        ctx->state = STREAMER_ERROR;
    }

    if (payload != NULL) {
        free(payload);
    }

    return true;
}

void streamer_close(streamer_context_t* ctx) {
    if (ctx == NULL) return;

    if (ctx->sock_fd >= 0) {
        netwrap_close(ctx->sock_fd);
        ctx->sock_fd = -1;
    }
    ctx->state = STREAMER_IDLE;
}
