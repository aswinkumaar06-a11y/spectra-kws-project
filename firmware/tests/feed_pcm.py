#!/usr/bin/env python3
"""
feed_pcm.py — Test Feeder for Spectra ASR Streaming Server

Streams a WAV file (or synthetic tone) over TCP to the ASR server using
the binary protocol (SP) and displays the returned transcription and latency.

Usage:
  python feed_pcm.py [--host 127.0.0.1] [--port 8765] [--wav path/to/audio.wav] [--realtime]
"""

import os
import sys
import time
import socket
import struct
import argparse
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from server.asr_server import (
    build_frame, parse_header,
    STREAM_HEADER_SIZE,
    STREAM_MSG_UTT_START, STREAM_MSG_PCM_BIN, STREAM_MSG_UTT_END,
    STREAM_MSG_PARTIAL, STREAM_MSG_FINAL
)


def load_audio(wav_path: str = None, duration_s: float = 2.0, sr: int = 16000) -> np.ndarray:
    if wav_path and os.path.exists(wav_path):
        import soundfile as sf
        audio, in_sr = sf.read(wav_path, dtype="int16")
        if audio.ndim > 1:
            audio = audio[:, 0]
        if in_sr != sr:
            import librosa
            audio_f = audio.astype(np.float32) / 32768.0
            resampled = librosa.resample(audio_f, orig_sr=in_sr, target_sr=sr)
            audio = (resampled * 32767.0).astype(np.int16)
        print(f"Loaded {wav_path}: {len(audio)} samples ({len(audio)/sr:.2f}s) @ {sr} Hz")
        return audio
    else:
        print(f"Generating synthetic 440 Hz test audio ({duration_s}s @ {sr} Hz)...")
        t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
        audio = (np.sin(2 * np.pi * 440 * t) * 12000).astype(np.int16)
        return audio


def feed_audio(host: str, port: int, audio: np.ndarray, utt_id: int = 1, realtime: bool = False):
    print(f"Connecting to ASR server at {host}:{port}...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))

    try:
        seq = 0
        # 1. Send UTT_START
        start_pl = struct.pack("<IHBB", utt_id, 16000, 1, 16)
        sock.sendall(build_frame(STREAM_MSG_UTT_START, seq, start_pl))
        seq += 1
        print(f"Sent UTT_START (UTT #{utt_id})")

        # 2. Stream in 100ms chunks (1600 samples)
        chunk_size = 1600
        chunk_seq = 0
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i + chunk_size]
            c_hdr = struct.pack("<IIH", utt_id, chunk_seq, len(chunk))
            sock.sendall(build_frame(STREAM_MSG_PCM_BIN, seq, c_hdr + chunk.tobytes()))
            seq += 1
            chunk_seq += 1

            if realtime:
                time.sleep(0.1)

        print(f"Streamed {len(audio)} samples in {chunk_seq} chunks")

        # 3. Send UTT_END
        end_pl = struct.pack("<IIB", utt_id, len(audio), 0)  # Reason = VAD_ENDPOINT
        sock.sendall(build_frame(STREAM_MSG_UTT_END, seq, end_pl))
        print("Sent UTT_END, waiting for final transcript...")

        # 4. Receive responses
        while True:
            hdr_bytes = sock.recv(STREAM_HEADER_SIZE)
            if not hdr_bytes:
                break
            parsed = parse_header(hdr_bytes)
            if not parsed:
                print("Error: Invalid header received")
                break

            msg_type, flags, payload_len, s_seq, crc = parsed
            payload = sock.recv(payload_len) if payload_len > 0 else b""

            if msg_type == STREAM_MSG_PARTIAL:
                p_utt, p_seq, t_len = struct.unpack("<IIH", payload[:10])
                partial_text = payload[10:10 + t_len].decode("utf-8")
                print(f"  [PARTIAL] {partial_text}")

            elif msg_type == STREAM_MSG_FINAL:
                f_utt, latency_ms, t_len = struct.unpack("<IIH", payload[:10])
                final_text = payload[10:10 + t_len].decode("utf-8")
                print("\n" + "=" * 50)
                print(f"  FINAL TRANSCRIPT (Latency: {latency_ms} ms)")
                print(f"  \"{final_text}\"")
                print("=" * 50 + "\n")
                break

    finally:
        sock.close()


def main():
    parser = argparse.ArgumentParser(description="Feed PCM audio to Spectra ASR Server")
    parser.add_argument("--host", default="127.0.0.1", help="ASR Server host")
    parser.add_argument("--port", type=int, default=8765, help="ASR Server port")
    parser.add_argument("--wav", default=None, help="Path to WAV audio file")
    parser.add_argument("--realtime", action="store_true", help="Simulate real-time streaming pace (100ms pauses)")
    args = parser.parse_args()

    audio = load_audio(args.wav)
    feed_audio(args.host, args.port, audio, realtime=args.realtime)


if __name__ == "__main__":
    main()
