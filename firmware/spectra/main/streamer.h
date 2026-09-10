/**
 * streamer.h — Spectra Audio Streamer & ASR Client
 *
 * Coordinates live audio streaming from ring buffer to ASR server:
 *   1. Connects to ASR server (default: port 8765)
 *   2. Sends STREAM_MSG_UTT_START
 *   3. Sends pre-roll freeze (16,000 samples = 1.0s)
 *   4. Streams 100 ms live chunks while monitoring VAD endpointing
 *   5. Sends STREAM_MSG_UTT_END on VAD endpoint, 10s cap, or abort button
 *   6. Awaits STREAM_MSG_FINAL from server
 */

#ifndef SPECTRA_STREAMER_H_
#define SPECTRA_STREAMER_H_

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include "stream_proto.h"
#include "vad.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    STREAMER_IDLE = 0,
    STREAMER_CONNECTING,
    STREAMER_STREAMING,
    STREAMER_WAITING_FINAL,
    STREAMER_COMPLETED,
    STREAMER_ERROR
} streamer_state_t;

typedef struct {
    char     server_host[64];
    uint16_t server_port;
    float    vad_threshold;
    uint32_t max_duration_ms;
} streamer_config_t;

typedef struct {
    streamer_state_t state;
    int              sock_fd;
    uint32_t         current_utt_id;
    uint32_t         seq_num;
    uint32_t         chunk_seq;
    uint32_t         total_samples_sent;
    vad_config_t     vad;
    char             last_transcript[256];
    bool             has_final;
    uint32_t         latency_ms;
    stream_end_reason_t end_reason;
} streamer_context_t;

/**
 * Initialize streamer context with default or user configuration.
 */
void streamer_init(streamer_context_t* ctx, const streamer_config_t* cfg);

/**
 * Start streaming an utterance.
 * Connects socket, sends UTT_START, and transmits pre-roll buffer.
 */
bool streamer_start(streamer_context_t* ctx, const char* host, uint16_t port,
                    uint32_t utt_id, const int16_t* pre_roll, size_t pre_roll_samples);

/**
 * Feed a live 100 ms audio chunk (1600 samples).
 * Evaluates VAD endpointing and transmits PCM_BIN frame.
 * If endpoint or max duration reached, automatically sends UTT_END.
 */
bool streamer_feed_chunk(streamer_context_t* ctx, const int16_t* pcm, size_t n_samples);

/**
 * Abort active streaming (e.g. BOOT button pressed).
 * Sends UTT_END with STREAM_END_ABORT.
 */
void streamer_abort(streamer_context_t* ctx);

/**
 * Poll for incoming server responses (PARTIAL, FINAL, ERROR).
 * Non-blocking / low-timeout.
 */
bool streamer_poll(streamer_context_t* ctx, int timeout_ms);

/**
 * Finish streaming session, close socket and reset state.
 */
void streamer_close(streamer_context_t* ctx);

#ifdef __cplusplus
}
#endif

#endif  // SPECTRA_STREAMER_H_
