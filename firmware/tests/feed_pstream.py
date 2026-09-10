#!/usr/bin/env python3
"""
feed_pstream.py - Serial P(pos) Stream Feeder for Spectra P4 Trigger Demo

Allows demonstrating and testing the trigger state machine on the XIAO ESP32-C5
without requiring a live microphone. Streams probability values over UART
and monitors device trigger events in real time.

Usage:
    python feed_pstream.py --port COM3 --stream A
    python feed_pstream.py --port COM3 --stream D
    python feed_pstream.py --port COM3 --custom 0.85,0.90,0.95,0.1,0.1
"""

import sys
import time
import struct
import argparse
import serial
import serial.tools.list_ports

# Predefined test streams matching normative spec §3
STREAMS = {
    "A": [0.10, 0.20, 0.60, 0.70, 0.80, 0.30, 0.10],
    "B": [0.60, 0.40, 0.70, 0.40, 0.80, 0.90, 0.95],
    "C": [0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 0.10],
    "D": [0.85, 0.90, 0.95, 0.10, 0.10, 0.10, 0.10, 0.80, 0.85, 0.90],
}


def send_stream(port: str, baud: int, probs: list[float], interval_s: float = 0.5):
    print(f"[FEED] Connecting to {port} at {baud} baud...")
    ser = serial.Serial(port, baud, timeout=0.1)
    time.sleep(1.0)  # Allow board boot/settle if DTR triggered reset

    print(f"[FEED] Streaming {len(probs)} probability ticks (interval={interval_s*1000:.0f} ms)...")
    print(f"[FEED] Vector: {[f'{p:.2f}' for p in probs]}")

    for i, p in enumerate(probs):
        # Format binary packet: 'SPTP' (4B magic) + uint32 tick + float32 probability
        packet = b"SPTP" + struct.pack("<If", i, float(p))
        ser.write(packet)
        # Also write plain text fallback for human terminal inspection
        ser.write(f"P {p:.3f}\n".encode("utf-8"))

        print(f"  Tick {i:02d} | p = {p:.3f} sent", end="")

        # Poll incoming UART for ~interval_s
        t_end = time.time() + interval_s
        responses = []
        while time.time() < t_end:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if line:
                responses.append(line)

        if responses:
            print(f" -> Device: {' | '.join(responses)}")
        else:
            print()

    # Listen for another 2 seconds for cooldown completion logs
    print("[FEED] Stream finished. Listening for trailing logs (2.0s)...")
    t_end = time.time() + 2.0
    while time.time() < t_end:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if line:
            print(f"  Device: {line}")

    ser.close()
    print("[FEED] Completed.")


def main():
    parser = argparse.ArgumentParser(description="Feed probability streams to Spectra ESP32-C5")
    parser.add_argument("--port", type=str, default="", help="Serial COM port (e.g. COM3)")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate (default: 115200)")
    parser.add_argument("--stream", type=str, choices=["A", "B", "C", "D"], default="D",
                        help="Predefined test stream (A, B, C, or D)")
    parser.add_argument("--custom", type=str, default="",
                        help="Comma-separated floats (e.g. 0.85,0.90,0.95)")
    parser.add_argument("--interval", type=float, default=0.5,
                        help="Tick interval in seconds (default: 0.5s = 500ms)")
    args = parser.parse_args()

    if not args.port:
        ports = list(serial.tools.list_ports.comports())
        print("Available serial ports:")
        for p in ports:
            print(f"  {p.device}: {p.description}")
        print("\nPlease specify a port with --port <PORT>")
        sys.exit(1)

    if args.custom:
        probs = [float(x.strip()) for x in args.custom.split(",") if x.strip()]
    else:
        probs = STREAMS[args.stream]

    send_stream(args.port, args.baud, probs, args.interval)


if __name__ == "__main__":
    main()
