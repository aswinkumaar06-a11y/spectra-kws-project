# Spectra KWS — XIAO ESP32-C5 Firmware

Keyword spotting firmware for **Seeed Studio XIAO ESP32-C5** (RISC-V single-core @ 240 MHz, 384 KB internal SRAM, 8 MB PSRAM, 8 MB Flash) running a DS-CNN model via TensorFlow Lite for Microcontrollers (TFLM).

## Frozen Contract

| Parameter | Value |
|---|---|
| Audio | 16 kHz, mono, 16-bit PCM |
| Clip duration | 1.0 s (16,000 samples) |
| Hop duration | 0.5 s (8,000 samples, 50% overlap) |
| DMA read block | 0.25 s (4,000 samples / 8 KB) |
| Ring buffer | 3.0 s (48,000 samples / 96 KB in PSRAM) |
| Overrun policy | Drop oldest unread samples + timestamped warning log |
| Features | 40 MFCC × 32 frames (FFT=1024, hop=512) |
| Model input | `int8[1, 40, 32, 1]`, scale=3.9962144, zp=60 |
| Model output | `int8[1, 2]`, scale=0.00390625, zp=−128 |
| Model size | 39,552 bytes (~38.6 KB) |
| Target Board | Seeed Studio XIAO ESP32-C5 |
| User LED | GPIO 27 (active HIGH) |
| I2S Mic (INMP441) | BCLK=GPIO 1 (D0), WS=GPIO 0 (D1), DIN=GPIO 23 (D4) |

## Prerequisites

### ESP-IDF Installation (Windows)

1. Download the **ESP-IDF v5.5.2+** installer (ESP32-C5 target support requires v5.5.x+) from:
   https://dl.espressif.com/dl/esp-idf/

2. Run the installer — it installs:
   - ESP-IDF framework (v5.5.2 or newer)
   - RISC-V (`riscv32-esp-elf`) toolchain
   - CMake, Ninja, Python environment

3. After installation, use the **ESP-IDF PowerShell** or **ESP-IDF Command Prompt**
   (these set up `IDF_PATH` and the toolchain in PATH automatically).

4. Verify installation:
   ```powershell
   idf.py --version            # Must report v5.5.x or newer
   riscv32-esp-elf-gcc --version
   ```

## Build & Flash

From the ESP-IDF command prompt:

```bash
cd firmware/spectra

# Set target to ESP32-C5 (first time only)
idf.py set-target esp32c5

# Configure options if needed (GPIOs, buffer duration, test injector)
idf.py menuconfig

# Build
idf.py build

# Flash (adjust COM port as needed)
idf.py -p COM3 flash

# Monitor UART output
idf.py -p COM3 monitor

# Build + Flash + Monitor in one command
idf.py -p COM3 flash monitor
```

## Expected Boot & Streaming Output

After flashing, the UART monitor displays Phase 0b contract verification, Phase 1 PSRAM ring allocation, deterministic self-test with golden CRC matching, and enters the streaming audio soak loop:

```
[SPECTRA] ========================================
[SPECTRA]   Spectra KWS — Phase 0b: Target Lock
[SPECTRA]   Board: XIAO ESP32-C5
[SPECTRA] ========================================
[SPECTRA] --- Heap (pre-model) ---
[SPECTRA] Heap internal: 312456 / 393216 bytes free (79.5%)
[SPECTRA] Heap PSRAM:    8388608 / 8388608 bytes free (100.0%)
[SPECTRA] Model loaded: 39552 bytes (schema v3)
[SPECTRA] Tensor arena: 54128 / 81920 bytes used (66.1%) [BSS placement]
[SPECTRA] Input:  shape=[1,40,32,1] dtype=INT8 scale=3.996214 zp=60
[SPECTRA] Output: shape=[1,2] dtype=INT8 scale=0.003906 zp=-128
[SPECTRA] Zero-input inference: neg=0.9961 pos=0.0000 (raw: 127, -128)
[SPECTRA] ========================================
[SPECTRA] Phase 0b PASS — model contract verified on XIAO ESP32-C5
[SPECTRA] ========================================
[SPECTRA] ========================================
[SPECTRA]   Spectra KWS — Phase 1: Audio Capture
[SPECTRA]   Target: XIAO ESP32-C5 | INMP441 I2S
[SPECTRA] ========================================
[SPECTRA] Allocating Audio Ring: 48000 samples (3.0s = 93 KB)...
[AUDIO_RING] Allocated 96000 bytes (48000 samples) in PSRAM
[SPECTRA] Ring Buffer Placement: PSRAM (External SPI RAM)
[SPECTRA] --- Starting WAV Injection Self-Test ---
[SPECTRA] Window 0 CRC-32: 0x6B2E496E (Golden: 0x6B2E496E) -> MATCH
[SPECTRA] Hop 1    CRC-32: 0x86F0B27A (Golden: 0x86F0B27A) -> MATCH
[SPECTRA] Testing forced-overrun drop-oldest behavior...
[WARN][AUDIO_RING] Overrun: dropped 8000 oldest samples (event #1, total dropped: 8000)
[SPECTRA] Forced-overrun result: overruns=1, dropped=8000 samples
[SPECTRA] WAV Injection Self-Test PASS: bit-exact match & overrun verified
[SPECTRA] Configuring INMP441 I2S on BCLK=23, WS=24, DIN=11
[AUDIO_CAPTURE] Initializing I2S standard RX driver for INMP441...
[AUDIO_CAPTURE] I2S initialized successfully: 16000 Hz, 16-bit, Mono Left
[SPECTRA] Phase 1 PASS — Capture & Ring Operational
[SPECTRA] Entering continuous audio ingestion loop...
[SPECTRA] [SOAK] Hops: 10 | Samples: 80000 | Peak: 0.412 | RMS: 0.0824 | Overruns: 0
[SPECTRA] [SOAK] Hops: 20 | Samples: 160000 | Peak: 0.395 | RMS: 0.0791 | Overruns: 0
```

## Host-Side Tests (No ESP-IDF Needed)

### 1. Python TFLite Model Contract Test
```bash
venv\Scripts\python.exe firmware/tests/test_model_contract.py
```

### 2. Python WAV Injection & Ring Buffer Verification
```bash
venv\Scripts\python.exe firmware/tests/test_wav_injection.py
```

### 3. C Model Array Validation
```bash
gcc -o test_model_array firmware/tests/test_model_array.c firmware/spectra/main/model_data.cc -I firmware/spectra/main
./test_model_array
```

### 3. C Audio Ring Buffer & WAV Injector Unit Test
```bash
gcc -o test_audio_ring firmware/tests/test_audio_ring.c firmware/spectra/main/audio_ring.cc firmware/spectra/main/wav_injector.cc -I firmware/spectra/main
./test_audio_ring
```

### 4. C MFCC Parity Host Regression Test (50 Clips & Edge Cases)
```bash
gcc -O3 -o test_mfcc_host firmware/tests/test_mfcc_host.c firmware/spectra/main/mfcc.cc -I firmware/spectra/main -lm
./test_mfcc_host
```

### 5. Automated Python Table & MFCC Parity Test Suite
```bash
venv\Scripts\python.exe -m unittest firmware/tests/test_mfcc_parity.py
```

### 6. Phase 3 TFLM Invariants, Zeros Smoke & Golden Logits Test Suite
```bash
venv\Scripts\python.exe -m unittest firmware/tests/test_logits_parity.py
```

### 10. Phase 5 Endpoint VAD & U2 Boundary Verification
```bash
gcc -O2 -o test_vad_host.exe firmware/tests/test_vad_host.c firmware/spectra/main/vad.cc firmware/spectra/main/mfcc.cc -I firmware/spectra/main -lm
./test_vad_host.exe
```

### 11. Phase 5 Streaming Protocol Loopback Test Suite
```bash
venv\Scripts\python.exe -m unittest firmware/tests/test_stream_proto.py
```

### 12. Phase 5 ASR TCP Server (Laptop)
```bash
venv\Scripts\python.exe server/asr_server.py --port 8765 --model tiny --compute_type int8
```

### 13. Phase 5 PCM Test Feeder
```bash
venv\Scripts\python.exe firmware/tests/feed_pcm.py --port 8765
```

## Reconciled Memory Budget Audit (U1 Remediated)

| Component | Placement | Size (Bytes) | Size (KB) | Status |
|---|---|---|---|---|
| **TFLM Tensor Arena** | Internal SRAM (.bss) | 81,920 | 80.0 KB | Compliant |
| **MFCC FFT & Mel Scratch** | Internal SRAM (.bss) | 28,676 | 28.0 KB | Cut from 96.7 KB (U1 fix) |
| **Pre-Roll Audio Buffer** | Internal SRAM (.bss) | 32,000 | 31.25 KB | 1.0s @ 16 kHz |
| **DMA Block Buffer** | Internal SRAM | 8,000 | 7.8 KB | 0.25s DMA chunk |
| **FreeRTOS Task Stacks** | Internal SRAM | 16,384 | 16.0 KB | 4 KB x 4 tasks |
| **App Static Internal SRAM** | Internal SRAM | **166,980** | **163.1 KB** | < 170 KB target |
| **IDF Base Runtime (est)** | Internal SRAM | ~45,000 | ~44.0 KB | Heap/stack/ROM |
| **Total Internal SRAM** | Internal SRAM | **~211,980** | **~207.0 KB** | **< 256.0 KB Target** |
| **Wi-Fi Headroom Available** | Internal SRAM | **~50,164** | **~49.0 KB** | **COMPLIANT** |
| Audio Ring Buffer (3.0s) | External PSRAM | 96,000 | 93.75 KB | 8 MB PSRAM |
| TFLite Model Weights | Flash (RO) | 39,552 | 38.6 KB | 8 MB Flash |
| DSP Tables (CSR Mel) | Flash (RO) | 35,468 | 34.6 KB | 8 MB Flash |

## Project Structure

```
firmware/
├── README.md                       ← Firmware documentation, phase tracker & vocabulary
├── spectra/
│   ├── CMakeLists.txt              ← ESP-IDF project root
│   ├── sdkconfig.defaults          ← XIAO ESP32-C5 config (IDF v5.5.2+, PSRAM enabled)
│   ├── partitions.csv              ← NVS + app partition (no SPIFFS)
│   ├── main/
│   │   ├── CMakeLists.txt          ← Component registration (I2S, TFLM, GPIO, NVS, WiFi, LwIP, PM)
│   │   ├── Kconfig.projbuild       ← Menuconfig options (mic, trigger, Wi-Fi, DFS, Watchdog)
│   │   ├── main.cc                 ← Entry: TFLM verify + PSRAM ring + MFCC + Trigger + Power + Guard
│   │   ├── spectra_config.h        ← Frozen contract: GPIOs, buffer, MFCC, TFLM, VAD, Power, Watchdog
│   │   ├── audio_capture.h/.cc     ← INMP441 I2S standard RX driver with DMA
│   │   ├── audio_ring.h/.cc        ← PSRAM circular buffer with drop-oldest overrun policy
│   │   ├── wav_injector.h/.cc      ← Deterministic synthetic test injector & CRC-32 engine
│   │   ├── mfcc.h/.cc              ← C MFCC engine (on-the-fly windowing, 28 KB scratch)
│   │   ├── tflm.h/.cc              ← TFLM engine: minimal resolver <5>, internal SRAM arena, invoke
│   │   ├── logits_test.h/.cc       ← 5-clip boot self-test, SRAM budget audit, UART streaming
│   │   ├── test_clips_5.h          ← 5 embedded golden clips (2 pos, 1 neg, 2 hard neg)
│   │   ├── trigger.h/.cc           ← Trigger state machine (LISTEN, CANDIDATE, TRIGGERED, COOLDOWN)
│   │   ├── pre_roll.h              ← Pre-roll audio window freeze buffer (16,000 int16 samples)
│   │   ├── vad.h/.cc               ← Endpoint VAD (100 ms RMS frames, TH=0.02, deterministic state machine)
│   │   ├── vad_calib.h/.cc         ← Acoustic VAD noise floor auto-calibration (clamped [0.012, 0.060])
│   │   ├── power_mgr.h/.cc         ← Dynamic Frequency Scaling (80/240 MHz), power locks, light sleep
│   │   ├── watchdog.h/.cc          ← Multi-channel Task Watchdog Timer supervisor (audio, infer, stream)
│   │   ├── boot_guard.h/.cc        ← NVS Boot-loop guard (3-crash tripwire to Safe Mode with 5-pulse alert)
│   │   ├── health_diag.h/.cc       ← Production health diagnostics & memory telemetry formatter
│   │   ├── netwrap.h/.cc           ← Cross-platform socket abstraction (ESP-IDF LwIP / Host Winsock)
│   │   ├── streamer.h/.cc          ← Streamer coordinator (pre-roll + live chunks + VAD endpoint)
│   │   ├── stream_proto.h          ← Binary streaming framing protocol (SP magic, CRC-16)
│   │   ├── hann_table.h            ← 1024-point periodic Hann window table
│   │   ├── twiddle_table.h         ← 512-point complex twiddle factors table
│   │   ├── mel_table.h             ← 128x513 sparse Slaney Mel filterbank CSR table
│   │   ├── dct_table.h             ← 40x128 Type-II orthonormal DCT table
│   │   ├── model_data.h            ← Model extern declarations
│   │   └── model_data.cc           ← Model byte array (aligned)
│   └── components/
│       └── tflm/
│           ├── CMakeLists.txt      ← TFLM component wrapper
│           └── idf_component.yml   ← Managed component manifest (esp-tflite-micro)
└── tests/
    ├── CMakeLists.txt              ← Host-side C test build
    ├── test_model_contract.py      ← Python contract validation
    ├── test_model_array.c          ← C model array validation
    ├── test_wav_injection.py       ← Python ring buffer & golden CRC generator
    ├── test_audio_ring.c           ← C circular ring unit test (31 test assertions)
    ├── gen_mfcc_goldens.py         ← Golden table & 50-clip binary bundle generator
    ├── mfcc_goldens.bin            ← 50-clip test bundle with Python ground truth
    ├── test_mfcc_host.c            ← C host regression test (50 clips, gates, edge cases)
    ├── test_mfcc_parity.py         ← Python test suite asserting table math & C parity
    ├── gen_logits_ref.py           ← Desktop golden logits generator (50 clips + 5 embedded)
    ├── logits_goldens.json         ← Full 50-clip desktop golden reference dataset
    ├── test_logits_parity.py       ← Host parity and model invariant test suite
    ├── test_trigger_host.c         ← C host test for Trigger state machine (Streams A, B, C, D)
    ├── feed_pstream.py             ← Python UART probability feeder for trigger demo
    ├── test_vad_host.c             ← C host test for Endpoint VAD (V1, V2, V3) & U2 boundary
    ├── test_calib_host.c           ← C host test for Acoustic VAD auto-calibration (8 assertions)
    ├── test_power_sim.py           ← Python duty cycle and battery simulation (DFS & light sleep)
    ├── test_boot_guard_host.c      ← C host test for Boot-loop Guard & Watchdog (11 assertions)
    ├── test_health_diag_host.c     ← C host test for Diagnostic Telemetry (8 assertions)
    ├── test_streamer_c_host.c      ← C host test for streamer connecting to ASR server
    ├── test_stream_proto.py        ← Python streaming protocol & TCP loopback test suite
    └── feed_pcm.py                 ← Python test feeder streaming audio to ASR server
```

## Binding Gate Status Vocabulary

- **IMPLEMENTED**: Code exists, syntax valid, unexecuted on device.
- **HOST-PASS**: Verified green on desktop/host environment (GCC/Python).
- **MEASURED**: Quantitative numbers measured on physical hardware.
- **PASS**: Measured on physical hardware AND within specified gate tolerances.
*(A gate requiring hardware can never be marked PASS on host evidence alone).*

## Firmware Phases

| Phase | Description | Key Dependency | Status |
|---|---|---|---|
| **0b** | **Target lock + Scaffold + Contract verification** | XIAO ESP32-C5, TFLM, LED proof | **HOST-PASS** (Flash pending) |
| **1** | **I2S audio capture + PSRAM ring buffer** | INMP441 microphone, 96 KB PSRAM ring | **HOST-PASS** (DMA on HW pending) |
| **2** | **MFCC feature extraction (C/DSP)** | 1024-pt FFT, 128 Mel, 40 DCT, INT8 | **HOST-PASS** (C5 cycles pending) |
| **3** | **TFLM inference + Memory discipline** | Minimal resolver <5>, arena in BSS, logits parity | **HOST-PASS** (Equivalence on HW pending) |
| **4** | **Confidence trigger + LED demo** | State machine (N=3, CD=4), pre-roll freeze, NVS | **HOST-PASS** (Live demo on HW pending) |
| **5** | **Streaming + ASR (faster-whisper)** | Endpoint VAD, Binary framing, 10s cap, laptop TCP server | **HOST-PASS** (Device Wi-Fi pending) |
| **6** | **Power optimization & VAD calibration** | DFS (80/240 MHz), Power locks, noise auto-calibration | **HOST-PASS** (Power analyzer pending) |
| **7** | **Production hardening & resilience** | Multi-channel TWDT, NVS Boot-loop guard, Safe Mode | **HOST-PASS** (Hardware fault pending) |
| **8** | **Hardware Execution Runbook** | Physical wiring, 6 validation gates, on-device runbook | **COMPLETED** (`docs/HARDWARE_RUNBOOK.md`) |


