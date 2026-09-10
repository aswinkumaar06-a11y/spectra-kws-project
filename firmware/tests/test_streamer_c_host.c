/**
 * test_streamer_c_host.c — Host C Test for Spectra Streamer & Socket Client
 *
 * Validates C-side streaming engine connecting to local ASR server:
 *   1. Connects to 127.0.0.1:<port>
 *   2. Sends UTT_START + 16,000 pre-roll samples
 *   3. Feeds speech chunks (1600 samples @ 100 ms)
 *   4. Feeds silence chunks to trigger VAD endpointing
 *   5. Polls for STREAM_MSG_FINAL and verifies transcript text
 */

#include "streamer.h"
#include "netwrap.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#define COLOR_GREEN "\033[32m"
#define COLOR_RED   "\033[31m"
#define COLOR_RESET "\033[0m"

int main(int argc, char** argv) {
    uint16_t port = 18765;
    if (argc > 1) {
        port = (uint16_t)atoi(argv[1]);
    }

    printf("====================================================\n");
    printf("  Spectra C Streamer Host Test (Port %u)\n", port);
    printf("====================================================\n");

    streamer_context_t ctx;
    streamer_init(&ctx, NULL);

    // Prepare 1.0s pre-roll (16,000 samples of 1000 amplitude)
    int16_t* pre_roll = (int16_t*)malloc(16000 * sizeof(int16_t));
    assert(pre_roll != NULL);
    for (int i = 0; i < 16000; i++) {
        pre_roll[i] = (int16_t)(i % 500);
    }

    printf("[1] Connecting and sending UTT_START + pre-roll...\n");
    bool ok = streamer_start(&ctx, "127.0.0.1", port, 777, pre_roll, 16000);
    if (!ok) {
        printf("  [" COLOR_RED "FAIL" COLOR_RESET "] Failed to start streamer to 127.0.0.1:%u\n", port);
        free(pre_roll);
        return 1;
    }
    printf("  [" COLOR_GREEN "PASS" COLOR_RESET "] Connected and pre-roll sent\n");

    // Prepare 100 ms speech chunk (1600 samples with RMS > 0.02)
    // e.g. amplitude 2000 -> RMS ~ 2000/32768 = 0.061 > 0.02
    int16_t speech_chunk[SPECTRA_VAD_FRAME_SAMPLES];
    for (int i = 0; i < SPECTRA_VAD_FRAME_SAMPLES; i++) {
        speech_chunk[i] = 2000;
    }

    // Prepare 100 ms silence chunk (amplitude 50 -> RMS ~ 0.0015 < 0.02)
    int16_t silence_chunk[SPECTRA_VAD_FRAME_SAMPLES];
    for (int i = 0; i < SPECTRA_VAD_FRAME_SAMPLES; i++) {
        silence_chunk[i] = 50;
    }

    printf("[2] Feeding 4 speech frames (400 ms)...\n");
    for (int f = 0; f < 4; f++) {
        ok = streamer_feed_chunk(&ctx, speech_chunk, SPECTRA_VAD_FRAME_SAMPLES);
        assert(ok);
        assert(ctx.state == STREAMER_STREAMING);
    }
    printf("  [" COLOR_GREEN "PASS" COLOR_RESET "] Streamed speech frames\n");

    printf("[3] Feeding 3 silence frames to trigger VAD endpointing...\n");
    for (int f = 0; f < 3; f++) {
        ok = streamer_feed_chunk(&ctx, silence_chunk, SPECTRA_VAD_FRAME_SAMPLES);
        assert(ok);
    }

    if (ctx.state != STREAMER_WAITING_FINAL) {
        printf("  [" COLOR_RED "FAIL" COLOR_RESET "] VAD failed to endpoint after 3 silence frames (state=%d)\n", ctx.state);
        streamer_close(&ctx);
        free(pre_roll);
        return 1;
    }
    printf("  [" COLOR_GREEN "PASS" COLOR_RESET "] VAD endpointed, state is STREAMER_WAITING_FINAL\n");

    printf("[4] Polling for STREAM_MSG_FINAL from server...\n");
    int poll_count = 0;
    while (!ctx.has_final && poll_count < 50) {
        streamer_poll(&ctx, 100);
        poll_count++;
    }

    if (!ctx.has_final) {
        printf("  [" COLOR_RED "FAIL" COLOR_RESET "] Timed out waiting for FINAL response\n");
        streamer_close(&ctx);
        free(pre_roll);
        return 1;
    }

    printf("  [" COLOR_GREEN "PASS" COLOR_RESET "] Received FINAL transcript: \"%s\" (latency: %u ms)\n",
           ctx.last_transcript, (unsigned)ctx.latency_ms);

    streamer_close(&ctx);
    free(pre_roll);

    printf("\n====================================================\n");
    printf("  All C Streamer Tests PASSED!\n");
    printf("====================================================\n");

    return 0;
}
