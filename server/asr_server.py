#!/usr/bin/env python3
"""
asr_server.py — Spectra High-Performance TCP ASR Streaming Server

Implements the Spectra binary protocol (SP) for real-time transcription
from XIAO ESP32-C5 or host streamer clients using faster-whisper.

Default Whisper configuration (Phase 5 frozen spec):
  - Model: tiny.en or tiny
  - Compute: int8 / CPU
  - Decoding: greedy (beam_size=1)
  - VAD: False (device-side VAD endpoints audio)
"""

import os
import sys
import time
import struct
import asyncio
import logging
import argparse
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ASR_SERVER")

# Protocol Constants
STREAM_MAGIC_0 = 0x53  # 'S'
STREAM_MAGIC_1 = 0x50  # 'P'
STREAM_HEADER_SIZE = 12

STREAM_MSG_UTT_START = 0x01
STREAM_MSG_PCM_BIN   = 0x02
STREAM_MSG_UTT_END   = 0x03
STREAM_MSG_PARTIAL   = 0x04
STREAM_MSG_FINAL     = 0x05
STREAM_MSG_PING      = 0x06
STREAM_MSG_PONG      = 0x07
STREAM_MSG_ERROR     = 0xFF

REASON_NAMES = {
    0: "VAD_ENDPOINT",
    1: "MAX_DURATION",
    2: "ABORT",
    3: "ERROR"
}


def crc16_ccitt(data: bytes, init: int = 0xFFFF) -> int:
    """Standard CRC-16-CCITT (poly=0x1021, init=0xFFFF)."""
    crc = init
    for b in data:
        crc ^= (b << 8) & 0xFFFF
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


def build_frame(msg_type: int, seq_num: int, payload: bytes = b"") -> bytes:
    """Construct a full binary frame with header and CRC-16."""
    payload_len = len(payload)
    flags = 0
    hdr_fields = struct.pack("<BBHI", msg_type, flags, payload_len, seq_num)
    crc = crc16_ccitt(hdr_fields + payload)
    header = struct.pack("<2sBBHIH", b"SP", msg_type, flags, payload_len, seq_num, crc)
    return header + payload


def parse_header(header_bytes: bytes):
    """Parse 12-byte header, returns (magic, msg_type, flags, payload_len, seq_num, crc16)."""
    if len(header_bytes) != STREAM_HEADER_SIZE:
        return None
    magic, msg_type, flags, payload_len, seq_num, crc = struct.unpack("<2sBBHIH", header_bytes)
    if magic != b"SP":
        return None
    return msg_type, flags, payload_len, seq_num, crc


class AsrEngine:
    """Wraps faster-whisper model with fallback for testing."""

    def __init__(self, model_size="tiny", device="cpu", compute_type="int8", mock=False):
        self.mock = mock
        self.model = None
        if not mock:
            try:
                from faster_whisper import WhisperModel
                logger.info("Loading faster-whisper model '%s' on %s (%s)...", model_size, device, compute_type)
                t0 = time.time()
                self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
                logger.info("Model loaded in %.2f s", time.time() - t0)
            except Exception as e:
                logger.warning("Could not load faster-whisper (%s). Falling back to mock engine.", e)
                self.mock = True

    def transcribe(self, audio_f32: np.ndarray, sample_rate: int = 16000) -> str:
        if self.mock or self.model is None:
            # Deterministic mock response based on audio energy
            rms = float(np.sqrt(np.mean(audio_f32**2))) if len(audio_f32) > 0 else 0.0
            if rms < 0.01:
                return ""
            return f"[MOCK TRANSCRIPT: audio duration {len(audio_f32)/sample_rate:.2f}s, rms={rms:.3f}]"

        segments, _ = self.model.transcribe(
            audio_f32,
            beam_size=1,
            vad_filter=False,
            language="en",
            temperature=0.0
        )
        texts = [s.text.strip() for s in segments if s.text]
        return " ".join(texts).strip()


class UtteranceSession:
    """Tracks state and audio samples for an active utterance."""

    def __init__(self, utt_id: int, sample_rate: int = 16000):
        self.utt_id = utt_id
        self.sample_rate = sample_rate
        self.pcm_chunks = []
        self.total_samples = 0
        self.start_time = time.time()


async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, engine: AsrEngine, save_dir: str = None):
    peer = writer.get_extra_info("peername")
    logger.info("Client connected from %s", peer)

    session = None
    server_seq = 0

    try:
        while True:
            # Read 12-byte header
            header_bytes = await reader.readexactly(STREAM_HEADER_SIZE)
            parsed = parse_header(header_bytes)
            if not parsed:
                logger.error("Invalid frame magic or header from %s", peer)
                break

            msg_type, flags, payload_len, client_seq, client_crc = parsed

            # Read payload
            payload = b""
            if payload_len > 0:
                payload = await reader.readexactly(payload_len)

            # Verify CRC-16
            hdr_fields = struct.pack("<BBHI", msg_type, flags, payload_len, client_seq)
            expected_crc = crc16_ccitt(hdr_fields + payload)
            if client_crc != expected_crc:
                logger.error("CRC-16 mismatch from %s (got 0x%04X, expected 0x%04X)", peer, client_crc, expected_crc)
                break

            # Process Message
            if msg_type == STREAM_MSG_UTT_START:
                utt_id, sr, channels, bits = struct.unpack("<IHBB", payload[:8])
                session = UtteranceSession(utt_id, sr)
                logger.info("[UTT #%u] START received: sr=%u, ch=%u, bits=%u", utt_id, sr, channels, bits)

            elif msg_type == STREAM_MSG_PCM_BIN:
                if not session:
                    logger.warning("Received PCM_BIN without UTT_START; creating default session")
                    session = UtteranceSession(1, 16000)

                utt_id, chunk_seq, sample_count = struct.unpack("<IIH", payload[:10])
                pcm_data = payload[10:]
                expected_bytes = sample_count * 2
                if len(pcm_data) >= expected_bytes:
                    chunk_samples = np.frombuffer(pcm_data[:expected_bytes], dtype=np.int16)
                    session.pcm_chunks.append(chunk_samples)
                    session.total_samples += sample_count
                    logger.debug("[UTT #%u] PCM chunk #%u: %u samples (total: %u)",
                                 utt_id, chunk_seq, sample_count, session.total_samples)

            elif msg_type == STREAM_MSG_UTT_END:
                utt_id, total_samples_reported, reason = struct.unpack("<IIB", payload[:9])
                reason_str = REASON_NAMES.get(reason, f"UNKNOWN({reason})")
                logger.info("[UTT #%u] END received: reason=%s, samples_reported=%u",
                            utt_id, reason_str, total_samples_reported)

                if session and reason != 2:  # Not aborted
                    t_inf_start = time.time()
                    if session.pcm_chunks:
                        audio_int16 = np.concatenate(session.pcm_chunks)
                    else:
                        audio_int16 = np.zeros(0, dtype=np.int16)

                    audio_f32 = audio_int16.astype(np.float32) / 32768.0

                    # Save wav if configured
                    if save_dir:
                        import soundfile as sf
                        os.makedirs(save_dir, exist_ok=True)
                        wav_path = os.path.join(save_dir, f"utt_{utt_id}_{int(time.time())}.wav")
                        sf.write(wav_path, audio_f32, session.sample_rate)
                        logger.info("[UTT #%u] Saved audio to %s", utt_id, wav_path)

                    # Run transcription
                    transcript = engine.transcribe(audio_f32, session.sample_rate)
                    latency_ms = int((time.time() - t_inf_start) * 1000)
                    logger.info("[UTT #%u] FINAL (%d ms): \"%s\"", utt_id, latency_ms, transcript)

                    # Send STREAM_MSG_FINAL
                    text_bytes = transcript.encode("utf-8")
                    final_payload = struct.pack("<IIH", utt_id, latency_ms, len(text_bytes)) + text_bytes
                    frame = build_frame(STREAM_MSG_FINAL, server_seq, final_payload)
                    server_seq += 1
                    writer.write(frame)
                    await writer.drain()

                session = None

            elif msg_type == STREAM_MSG_PING:
                # Reply with PONG
                frame = build_frame(STREAM_MSG_PONG, server_seq, payload)
                server_seq += 1
                writer.write(frame)
                await writer.drain()

    except asyncio.IncompleteReadError:
        logger.info("Client %s disconnected normally", peer)
    except Exception as e:
        logger.error("Error handling client %s: %s", peer, e)
    finally:
        writer.close()
        await writer.wait_closed()
        logger.info("Connection closed for %s", peer)


async def main():
    parser = argparse.ArgumentParser(description="Spectra ASR Streaming Server")
    parser.add_argument("--host", default="0.0.0.0", help="Listen host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8765, help="Listen port (default: 8765)")
    parser.add_argument("--model", default="tiny", help="Whisper model name (default: tiny)")
    parser.add_argument("--device", default="cpu", help="Compute device: cpu or cuda (default: cpu)")
    parser.add_argument("--compute_type", default="int8", help="Compute type: int8 or float32 (default: int8)")
    parser.add_argument("--save-dir", default=None, help="Directory to save incoming audio WAV files")
    parser.add_argument("--mock", action="store_true", help="Run in mock mode without loading faster-whisper")
    args = parser.parse_args()

    engine = AsrEngine(
        model_size=args.model,
        device=args.device,
        compute_type=args.compute_type,
        mock=args.mock
    )

    server = await asyncio.start_server(
        lambda r, w: handle_client(r, w, engine, args.save_dir),
        args.host,
        args.port
    )

    addrs = ", ".join(str(sock.getsockname()) for sock in server.sockets)
    logger.info("Spectra ASR Server listening on %s (Port %d)", addrs, args.port)

    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
