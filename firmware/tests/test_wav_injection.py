"""
test_wav_injection.py — Python Validation of Audio Ring Buffer and WAV Injection

Validates:
  1. Bit-exact circular ring buffer mathematical model in Python
  2. Multi-hop windowing (16,000 sample window, 8,000 sample hop)
  3. Continuous streaming without phase distortion or boundary glitches
  4. Drop-oldest overrun policy under saturation
  5. Golden CRC-32 checksum generation for on-device hex/CRC verification
"""

import sys
import zlib
import struct
import numpy as np

# Contract constants from spectra_config.h
SAMPLE_RATE = 16000
RING_CAPACITY = 48000           # 3 seconds @ 16kHz
WINDOW_SAMPLES = 16000          # 1.0 second inference window
HOP_SAMPLES = 8000              # 0.5 second hop (50% overlap)
BLOCK_SAMPLES = 4000            # 0.25 second DMA block


class AudioRingReference:
    """Python reference implementation of the Spectra audio ring buffer."""

    def __init__(self, capacity=RING_CAPACITY):
        self.capacity = capacity
        self.buffer = np.zeros(capacity, dtype=np.int16)
        self.head = 0
        self.tail = 0
        self.available = 0
        self.overrun_count = 0
        self.dropped_samples = 0

    def write(self, samples: np.ndarray) -> int:
        n = len(samples)
        if n > self.capacity:
            ignored = n - self.capacity
            samples = samples[ignored:]
            n = self.capacity
            self.overrun_count += 1
            self.dropped_samples += ignored

        # Drop-oldest overrun policy
        if self.available + n > self.capacity:
            overflow = (self.available + n) - self.capacity
            self.tail = (self.tail + overflow) % self.capacity
            self.available -= overflow
            self.overrun_count += 1
            self.dropped_samples += overflow

        first_chunk = min(self.capacity - self.head, n)
        self.buffer[self.head:self.head + first_chunk] = samples[:first_chunk]
        second_chunk = n - first_chunk
        if second_chunk > 0:
            self.buffer[:second_chunk] = samples[first_chunk:]

        self.head = (self.head + n) % self.capacity
        self.available += n
        return n

    def peek_window(self, window_size=WINDOW_SAMPLES):
        if self.available < window_size:
            return None
        first_chunk = min(self.capacity - self.tail, window_size)
        w = np.empty(window_size, dtype=np.int16)
        w[:first_chunk] = self.buffer[self.tail:self.tail + first_chunk]
        second_chunk = window_size - first_chunk
        if second_chunk > 0:
            w[first_chunk:] = self.buffer[:second_chunk]
        return w

    def advance(self, hop_size=HOP_SAMPLES) -> bool:
        if self.available < hop_size:
            return False
        self.tail = (self.tail + hop_size) % self.capacity
        self.available -= hop_size
        return True


def compute_crc32_pcm(samples: np.ndarray) -> int:
    """Compute IEEE 802.3 CRC-32 across int16 little-endian bytes."""
    raw_bytes = samples.astype('<i2').tobytes()
    return zlib.crc32(raw_bytes) & 0xFFFFFFFF


def generate_ramp_block(start_counter: int, count: int) -> np.ndarray:
    indices = np.arange(start_counter, start_counter + count, dtype=np.int64)
    return (indices & 0xFFFF).astype(np.int16)


def test_ring_and_injection():
    print("====================================================")
    print("  Spectra WAV Injection & Ring Buffer Verification")
    print("====================================================")

    ring = AudioRingReference()
    passed = 0

    # 1. Incomplete window test
    b0 = generate_ramp_block(0, BLOCK_SAMPLES)
    ring.write(b0)
    assert ring.available == BLOCK_SAMPLES
    assert ring.peek_window() is None, "Should fail when available < 16000"
    print("[PASS] Incomplete window peek correctly returned None")
    passed += 1

    # 2. Fill to full 16,000 window
    for b in range(1, 4):
        ring.write(generate_ramp_block(b * BLOCK_SAMPLES, BLOCK_SAMPLES))
    assert ring.available == WINDOW_SAMPLES

    w0 = ring.peek_window()
    assert w0 is not None
    expected_w0 = np.arange(0, WINDOW_SAMPLES, dtype=np.int16)
    assert np.array_equal(w0, expected_w0), "Window 0 does not match expected ramp"
    crc0 = compute_crc32_pcm(w0)
    print(f"[PASS] Window 0 verified: shape={w0.shape}, CRC32=0x{crc0:08X}")
    passed += 1

    # 3. Hop advance & streaming over multiple windows
    expected_crcs = []
    expected_crcs.append((0, crc0))

    cur_sample = 16000
    for hop_idx in range(1, 10):
        assert ring.advance(HOP_SAMPLES), f"Failed to advance hop {hop_idx}"

        # Ingest 2 new blocks (8000 samples)
        for _ in range(2):
            ring.write(generate_ramp_block(cur_sample, BLOCK_SAMPLES))
            cur_sample += BLOCK_SAMPLES

        w = ring.peek_window()
        assert w is not None
        start_val = hop_idx * HOP_SAMPLES
        expected_w = ((np.arange(start_val, start_val + WINDOW_SAMPLES, dtype=np.int64)) & 0xFFFF).astype(np.int16)
        assert np.array_equal(w, expected_w), f"Hop {hop_idx} window mismatch!"

        crc = compute_crc32_pcm(w)
        expected_crcs.append((hop_idx, crc))
        print(f"[PASS] Hop {hop_idx:2d} verified: start={start_val:6d}, CRC32=0x{crc:08X}")
        passed += 1

    assert ring.overrun_count == 0, "Nominal streaming produced spurious overruns"
    print("[PASS] Nominal streaming completed with 0 overruns")
    passed += 1

    # 4. Overrun test
    print("\n--- Testing Drop-Oldest Overrun Under Saturation ---")
    ring.advance(HOP_SAMPLES)
    # Write 50,000 samples in blocks without advancing
    for _ in range(10):
        ring.write(generate_ramp_block(cur_sample, 5000))
        cur_sample += 5000

    assert ring.available == RING_CAPACITY, f"Available {ring.available} != capacity {RING_CAPACITY}"
    assert ring.overrun_count > 0, "Overrun count was not incremented"
    print(f"[PASS] Overrun correctly detected: events={ring.overrun_count}, dropped={ring.dropped_samples}")
    passed += 1

    w_after = ring.peek_window()
    assert w_after is not None, "Peek failed after overrun recovery"
    print(f"[PASS] Peek succeeds after overrun: window shape={w_after.shape}")
    passed += 1

    print("\n====================================================")
    print(f"  All {passed} checks PASSED successfully!")
    print("====================================================")

    print("\nGolden Checksums Table for On-Device Parity:")
    print("| Hop Index | Audio Range (samples) | CRC-32 Checksum |")
    print("|---|---|---|")
    for h, c in expected_crcs[:6]:
        start = h * HOP_SAMPLES
        end = start + WINDOW_SAMPLES
        print(f"| Hop {h} | [{start:5d}, {end:5d}) | 0x{c:08X} |")


if __name__ == '__main__':
    test_ring_and_injection()
