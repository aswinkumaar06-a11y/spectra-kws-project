/**
 * stream_proto.h — Spectra Streaming Protocol Frame Codec
 *
 * Binary framing protocol between XIAO ESP32-C5 and ASR Server:
 *   [Magic: 2B ('SP')] [Type: 1B] [Flags: 1B] [PayloadLen: 2B] [Seq: 4B] [CRC16: 2B] [Payload: N B]
 * Total header = 12 bytes.
 */

#ifndef SPECTRA_STREAM_PROTO_H_
#define SPECTRA_STREAM_PROTO_H_

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define STREAM_MAGIC_0        0x53  // 'S'
#define STREAM_MAGIC_1        0x50  // 'P'
#define STREAM_HEADER_SIZE    12
#define STREAM_MAX_PAYLOAD    8192

// Message Types
typedef enum {
    STREAM_MSG_UTT_START = 0x01,  // Device -> Server: Begin utterance
    STREAM_MSG_PCM_BIN   = 0x02,  // Device -> Server: Audio chunk (PCM 16-bit)
    STREAM_MSG_UTT_END   = 0x03,  // Device -> Server: End utterance
    STREAM_MSG_PARTIAL   = 0x04,  // Server -> Device: Partial transcript
    STREAM_MSG_FINAL     = 0x05,  // Server -> Device: Final transcript
    STREAM_MSG_PING      = 0x06,  // Device <-> Server: Ping
    STREAM_MSG_PONG      = 0x07,  // Device <-> Server: Pong
    STREAM_MSG_ERROR     = 0xFF   // Server -> Device: Error response
} stream_msg_type_t;

// End Utterance Reasons
typedef enum {
    STREAM_END_VAD_ENDPOINT = 0,
    STREAM_END_MAX_DURATION = 1,
    STREAM_END_ABORT        = 2,
    STREAM_END_ERROR        = 3
} stream_end_reason_t;

#pragma pack(push, 1)

typedef struct {
    uint8_t  magic[2];       // 'S', 'P'
    uint8_t  msg_type;       // stream_msg_type_t
    uint8_t  flags;          // reserved (0)
    uint16_t payload_len;    // bytes in payload
    uint32_t seq_num;        // frame sequence number
    uint16_t crc16;          // CRC-16 over type, flags, payload_len, seq_num, and payload
} stream_header_t;

// Payload for STREAM_MSG_UTT_START
typedef struct {
    uint32_t utt_id;
    uint16_t sample_rate;       // 16000
    uint8_t  channels;          // 1
    uint8_t  bits_per_sample;   // 16
} stream_payload_start_t;

// Header for STREAM_MSG_PCM_BIN (followed immediately by int16_t samples)
typedef struct {
    uint32_t utt_id;
    uint32_t chunk_seq;
    uint16_t sample_count;      // e.g. 1600 for 100 ms or 16000 for pre-roll
} stream_payload_pcm_header_t;

// Payload for STREAM_MSG_UTT_END
typedef struct {
    uint32_t utt_id;
    uint32_t total_samples;
    uint8_t  reason;            // stream_end_reason_t
} stream_payload_end_t;

// Payload for STREAM_MSG_FINAL
typedef struct {
    uint32_t utt_id;
    uint32_t latency_ms;
    uint16_t text_len;          // followed immediately by UTF-8 characters (no null term required)
} stream_payload_final_t;

// Payload for STREAM_MSG_PARTIAL
typedef struct {
    uint32_t utt_id;
    uint32_t seq_num;
    uint16_t text_len;          // followed immediately by UTF-8 characters
} stream_payload_partial_t;

#pragma pack(pop)

/**
 * Standard CRC-16-CCITT (poly=0x1021, init=0xFFFF).
 */
static inline uint16_t stream_crc16_calc(const uint8_t* data, size_t len) {
    uint16_t crc = 0xFFFF;
    for (size_t i = 0; i < len; i++) {
        crc ^= ((uint16_t)data[i] << 8);
        for (int b = 0; b < 8; b++) {
            if (crc & 0x8000) {
                crc = (crc << 1) ^ 0x1021;
            } else {
                crc = (crc << 1);
            }
        }
    }
    return crc;
}

/**
 * Calculate CRC-16 for an entire frame (header fields starting from msg_type + payload).
 */
static inline uint16_t stream_frame_crc16(uint8_t msg_type, uint8_t flags,
                                          uint16_t payload_len, uint32_t seq_num,
                                          const uint8_t* payload) {
    uint8_t hdr_bytes[8];
    hdr_bytes[0] = msg_type;
    hdr_bytes[1] = flags;
    hdr_bytes[2] = (uint8_t)(payload_len & 0xFF);
    hdr_bytes[3] = (uint8_t)((payload_len >> 8) & 0xFF);
    hdr_bytes[4] = (uint8_t)(seq_num & 0xFF);
    hdr_bytes[5] = (uint8_t)((seq_num >> 8) & 0xFF);
    hdr_bytes[6] = (uint8_t)((seq_num >> 16) & 0xFF);
    hdr_bytes[7] = (uint8_t)((seq_num >> 24) & 0xFF);

    uint16_t crc = 0xFFFF;
    for (size_t i = 0; i < 8; i++) {
        crc ^= ((uint16_t)hdr_bytes[i] << 8);
        for (int b = 0; b < 8; b++) {
            if (crc & 0x8000) {
                crc = (crc << 1) ^ 0x1021;
            } else {
                crc = (crc << 1);
            }
        }
    }

    if (payload != NULL && payload_len > 0) {
        for (size_t i = 0; i < payload_len; i++) {
            crc ^= ((uint16_t)payload[i] << 8);
            for (int b = 0; b < 8; b++) {
                if (crc & 0x8000) {
                    crc = (crc << 1) ^ 0x1021;
                } else {
                    crc = (crc << 1);
                }
            }
        }
    }

    return crc;
}

/**
 * Build and encode a stream frame header.
 */
static inline void stream_build_header(stream_header_t* out_hdr, uint8_t msg_type,
                                       uint32_t seq_num, const uint8_t* payload,
                                       uint16_t payload_len) {
    out_hdr->magic[0] = STREAM_MAGIC_0;
    out_hdr->magic[1] = STREAM_MAGIC_1;
    out_hdr->msg_type = msg_type;
    out_hdr->flags = 0;
    out_hdr->payload_len = payload_len;
    out_hdr->seq_num = seq_num;
    out_hdr->crc16 = stream_frame_crc16(msg_type, 0, payload_len, seq_num, payload);
}

/**
 * Verify received frame header and payload CRC.
 */
static inline bool stream_verify_header(const stream_header_t* hdr, const uint8_t* payload) {
    if (hdr->magic[0] != STREAM_MAGIC_0 || hdr->magic[1] != STREAM_MAGIC_1) {
        return false;
    }
    uint16_t expected_crc = stream_frame_crc16(hdr->msg_type, hdr->flags,
                                               hdr->payload_len, hdr->seq_num, payload);
    return (hdr->crc16 == expected_crc);
}

#ifdef __cplusplus
}
#endif

#endif  // SPECTRA_STREAM_PROTO_H_
