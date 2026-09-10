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
| I2S Mic (INMP441) | BCLK=GPIO23 (D4), WS=GPIO24 (D5), DIN=GPIO11 (D6) |

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

### 4. C Audio Ring Buffer & WAV Injector Unit Test
```bash
gcc -o test_audio_ring firmware/tests/test_audio_ring.c firmware/spectra/main/audio_ring.cc firmware/spectra/main/wav_injector.cc -I firmware/spectra/main
./test_audio_ring
```

## Project Structure

```
firmware/
├── README.md                       ← Firmware documentation and phase tracker
├── spectra/
│   ├── CMakeLists.txt              ← ESP-IDF project root
│   ├── sdkconfig.defaults          ← XIAO ESP32-C5 config (IDF v5.5.2+, PSRAM enabled)
│   ├── partitions.csv              ← NVS + app partition (no SPIFFS)
│   ├── main/
│   │   ├── CMakeLists.txt          ← Component registration with I2S driver
│   │   ├── Kconfig.projbuild       ← Menuconfig options (mic GPIOs, ring size, test switch)
│   │   ├── main.cc                 ← Entry: TFLM verify + PSRAM ring init + capture loop
│   │   ├── spectra_config.h        ← Frozen contract, GPIOs, ring constants
│   │   ├── audio_capture.h/.cc     ← INMP441 I2S standard RX driver with DMA
│   │   ├── audio_ring.h/.cc        ← PSRAM circular buffer with drop-oldest overrun policy
│   │   ├── wav_injector.h/.cc      ← Deterministic synthetic test injector & CRC-32 engine
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
    └── test_audio_ring.c           ← C circular ring unit test (26 test assertions)
```

## Firmware Phases

| Phase | Description | Key Dependency | Status |
|---|---|---|---|
| **0b** | **Target lock + Scaffold + Contract verification** | XIAO ESP32-C5, TFLM, LED proof | ✅ **Complete** |
| **1** | **I2S audio capture + PSRAM ring buffer** | INMP441 microphone, 96 KB PSRAM ring | ✅ **Complete** |
| 2 | MFCC feature extraction (C/DSP) | Fixed-point DSP engine | ⬜ Planned |
| 3 | Inference pipeline integration | Minimal op resolver + arena optimization | ⬜ Planned |
| 4 | Energy gating + Wake logic | Threshold detection | ⬜ Planned |
| 5 | Power optimization | Dynamic frequency scaling & sleep modes | ⬜ Planned |
| 6 | OTA update support | Dual OTA partitions | ⬜ Planned |
| 7 | Production hardening | Watchdog, brownout, fail-safe recovery | ⬜ Planned |
