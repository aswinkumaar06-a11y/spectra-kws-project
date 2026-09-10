# Spectra KWS — XIAO ESP32-C5 Firmware

Keyword spotting firmware for **Seeed Studio XIAO ESP32-C5** (RISC-V single-core @ 240 MHz, 384 KB internal SRAM, 8 MB PSRAM, 8 MB Flash) running a DS-CNN model via TensorFlow Lite for Microcontrollers (TFLM).

## Frozen Contract

| Parameter | Value |
|---|---|
| Audio | 16 kHz, mono, 16-bit PCM |
| Clip duration | 1.0 s (16,000 samples) |
| Features | 40 MFCC × 32 frames (FFT=1024, hop=512) |
| Model input | `int8[1, 40, 32, 1]`, scale=3.9962144, zp=60 |
| Model output | `int8[1, 2]`, scale=0.00390625, zp=−128 |
| Model size | 39,552 bytes (~38.6 KB) |
| Target Board | Seeed Studio XIAO ESP32-C5 |
| User LED | GPIO 27 (active HIGH) |
| I2S Mic (INMP441) | BCLK=GPIO23, WS=GPIO24, DIN=GPIO11 |

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

# Build
idf.py build

# Flash (adjust COM port as needed)
idf.py -p COM3 flash

# Monitor UART output
idf.py -p COM3 monitor

# Build + Flash + Monitor in one command
idf.py -p COM3 flash monitor
```

## Phase 0b Expected Output

After flashing, the UART monitor displays contract verification + heap report, and the onboard LED blinks 3× on PASS (or steady on FAIL):

```
[SPECTRA] ========================================
[SPECTRA]   Spectra KWS — Phase 0b: Target Lock
[SPECTRA]   Board: XIAO ESP32-C5
[SPECTRA] ========================================
[SPECTRA] --- Heap (pre-model) ---
[SPECTRA] Heap internal: XXXXX / 393216 bytes free (XX.X%)
[SPECTRA] Heap PSRAM:    XXXXX / 8388608 bytes free (XX.X%)
[SPECTRA] Model loaded: 39552 bytes (schema v3)
[SPECTRA] Tensor arena: XXXXX / 81920 bytes used (XX.X%) [BSS placement]
[SPECTRA] --- Heap (post-model) ---
[SPECTRA] Heap internal: XXXXX / 393216 bytes free (XX.X%)
[SPECTRA] Heap PSRAM:    XXXXX / 8388608 bytes free (XX.X%)
[SPECTRA] Input:  shape=[1,40,32,1] dtype=INT8 scale=3.996214 zp=60
[SPECTRA] Output: shape=[1,2] dtype=INT8 scale=0.003906 zp=-128
[SPECTRA] Zero-input inference: neg=0.9961 pos=0.0000 (raw: 127, -128)
[SPECTRA] ========================================
[SPECTRA] Phase 0b PASS — model contract verified on XIAO ESP32-C5
[SPECTRA] ========================================
```

## Host-Side Tests (No ESP-IDF Needed)

### Python Contract Test

```bash
cd <project-root>
venv\Scripts\python.exe firmware/tests/test_model_contract.py
```

### C Array Validation (requires gcc)

```bash
gcc -o test_model_array firmware/tests/test_model_array.c firmware/spectra/main/model_data.cc -I firmware/spectra/main
./test_model_array
```

## Project Structure

```
firmware/
├── README.md                       ← Firmware documentation and phase tracker
├── spectra/
│   ├── CMakeLists.txt              ← ESP-IDF project root
│   ├── sdkconfig.defaults          ← XIAO ESP32-C5 config (IDF v5.5.2+)
│   ├── partitions.csv              ← NVS + app partition (no SPIFFS)
│   ├── main/
│   │   ├── CMakeLists.txt          ← Component registration
│   │   ├── main.cc                 ← Entry: TFLM AllOpsResolver + contract verify + LED blink
│   │   ├── model_data.h            ← Model extern declarations
│   │   ├── model_data.cc           ← Model byte array (aligned)
│   │   └── spectra_config.h        ← DSP/model frozen contract & GPIO definitions
│   └── components/
│       └── tflm/
│           ├── CMakeLists.txt      ← TFLM component wrapper
│           └── idf_component.yml   ← Managed component manifest (esp-tflite-micro)
└── tests/
    ├── CMakeLists.txt              ← Host-side C test build
    ├── test_model_contract.py      ← Python contract validation
    └── test_model_array.c          ← C array/magic validation
```

## Firmware Phases

| Phase | Description | Key Dependency | Status |
|---|---|---|---|
| **0b** | **Target lock + Scaffold + Contract verification** | XIAO ESP32-C5, TFLM, LED proof | ✅ **Complete** |
| 1 | I2S audio capture + Ring buffer | INMP441 microphone | ⬜ Planned |
| 2 | MFCC feature extraction (C/DSP) | Fixed-point DSP engine | ⬜ Planned |
| 3 | Inference pipeline integration | Minimal op resolver + arena optimization | ⬜ Planned |
| 4 | Energy gating + Wake logic | Threshold detection | ⬜ Planned |
| 5 | Power optimization | Dynamic frequency scaling & sleep modes | ⬜ Planned |
| 6 | OTA update support | Dual OTA partitions | ⬜ Planned |
| 7 | Production hardening | Watchdog, brownout, fail-safe recovery | ⬜ Planned |
