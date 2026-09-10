# Spectra KWS — ESP32-S3 Firmware

Keyword spotting firmware for ESP32-S3 running a DS-CNN model via TensorFlow Lite for Microcontrollers (TFLM).

## Frozen Contract

| Parameter | Value |
|---|---|
| Audio | 16 kHz, mono, 16-bit PCM |
| Clip duration | 1.0 s (16,000 samples) |
| Features | 40 MFCC × 32 frames (FFT=1024, hop=512) |
| Model input | `int8[1, 40, 32, 1]`, scale=3.9962144, zp=60 |
| Model output | `int8[1, 2]`, scale=0.00390625, zp=−128 |
| Model size | 39,552 bytes (~38.6 KB) |

## Prerequisites

### ESP-IDF Installation (Windows)

1. Download the **ESP-IDF v5.3** installer from:
   https://dl.espressif.com/dl/esp-idf/

2. Run the installer — it installs:
   - ESP-IDF framework
   - Xtensa and RISC-V toolchains
   - CMake, Ninja, Python environment

3. After installation, use the **ESP-IDF PowerShell** or **ESP-IDF Command Prompt**
   (these set up `IDF_PATH` and the toolchain in PATH automatically).

4. Verify installation:
   ```powershell
   idf.py --version
   xtensa-esp32s3-elf-gcc --version
   ```

## Build & Flash

From the ESP-IDF command prompt:

```bash
cd firmware/spectra

# Set target (first time only)
idf.py set-target esp32s3

# Build
idf.py build

# Flash (adjust COM port as needed)
idf.py -p COM3 flash

# Monitor UART output
idf.py -p COM3 monitor

# Build + Flash + Monitor in one command
idf.py -p COM3 flash monitor
```

## Phase 0 Expected Output

After flashing, the UART monitor should display:

```
[SPECTRA] ========================================
[SPECTRA]   Spectra KWS — Phase 0: Target Lock
[SPECTRA] ========================================
[SPECTRA] Model loaded: 39552 bytes (schema vN)
[SPECTRA] Tensor arena: XXXXX / 81920 bytes used (XX.X%)
[SPECTRA] Input:  shape=[1,40,32,1] dtype=INT8 scale=3.996214 zp=60
[SPECTRA] Output: shape=[1,2] dtype=INT8 scale=0.003906 zp=-128
[SPECTRA] Zero-input inference: neg=X.XXXX pos=X.XXXX
[SPECTRA] ========================================
[SPECTRA] Phase 0 PASS — model contract verified
[SPECTRA] ========================================
```

## Host-Side Tests (No ESP-IDF Needed)

### Python Contract Test

```bash
cd <project-root>
python firmware/tests/test_model_contract.py
```

### C Array Validation (requires gcc)

```bash
gcc -o test_model_array firmware/tests/test_model_array.c firmware/spectra/main/model_data.cc -I firmware/spectra/main
./test_model_array
```

## Project Structure

```
firmware/
├── README.md                       ← This file
├── spectra/
│   ├── CMakeLists.txt              ← ESP-IDF project root
│   ├── sdkconfig.defaults          ← ESP32-S3 config
│   ├── partitions.csv              ← Flash partition table
│   ├── main/
│   │   ├── CMakeLists.txt          ← Component registration
│   │   ├── main.cc                 ← Entry: TFLM load + verify
│   │   ├── model_data.h            ← Model extern declarations
│   │   ├── model_data.cc           ← Model byte array (aligned)
│   │   └── spectra_config.h        ← DSP/model frozen contract
│   └── components/
│       └── tflm/
│           ├── CMakeLists.txt      ← TFLM component wrapper
│           └── idf_component.yml   ← Managed component manifest
└── tests/
    ├── CMakeLists.txt              ← Host-side C test build
    ├── test_model_contract.py      ← Python contract validation
    └── test_model_array.c          ← C array/magic validation
```

## Firmware Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Target lock + scaffold | ✅ Complete |
| 1 | MFCC feature extraction (C/DSP) | ⬜ Planned |
| 2 | I2S audio capture + ring buffer | ⬜ Planned |
| 3 | Inference pipeline integration | ⬜ Planned |
| 4 | Energy gating + wake logic | ⬜ Planned |
| 5 | Power optimization | ⬜ Planned |
| 6 | OTA update support | ⬜ Planned |
| 7 | Production hardening | ⬜ Planned |
