# Spectra KWS — Hardware Execution Runbook (XIAO ESP32-C5)

**Target Device:** Seeed Studio XIAO ESP32-C5 (RISC-V single-core @ 240 MHz, 8 MB Flash, 8 MB PSRAM)  
**Audio Frontend:** INMP441 Omnidirectional Digital I2S Microphone  
**Status:** Canonical deployment & validation runbook for physical hardware bring-up.

---

## 1. Physical Wiring & Pinout

| INMP441 Pin | XIAO ESP32-C5 Pin | ESP32-C5 GPIO | Notes |
|---|---|---|---|
| **VDD** | 3V3 | 3.3V Power | Decouple with 100 nF capacitor if line noise occurs |
| **GND** | GND | Ground | Common system ground |
| **SD** (Serial Data) | D6 | **GPIO11** | I2S Data In (DIN) |
| **WS** (Word Select) | D5 | **GPIO24** | I2S Left/Right Clock (WS) |
| **SCK** (Serial Clock) | D4 | **GPIO23** | I2S Bit Clock (BCLK) |
| **L/R** | GND | Ground | Ties microphone channel to Left |

*Onboard LED:* **GPIO27** (Active HIGH, driven by internal GPIO matrix).

---

## 2. Toolchain Setup & Flashing

### 2.1 Prerequisites
- ESP-IDF v5.1+ or v5.3+ installed and exported in terminal environment.
- Python virtual environment with dependencies:
  ```powershell
  .\venv\Scripts\activate
  pip install pyserial faster-whisper numpy scipy soundfile librosa
  ```

### 2.2 Target Configuration & Build
```powershell
cd firmware/spectra

# Set target to ESP32-C5
idf.py set-target esp32c5

# Build the complete production firmware bundle
idf.py build
```

### 2.3 Flashing & Serial Monitor
Identify your device COM port (e.g. `COM3` on Windows or `/dev/ttyACM0` on Linux):
```powershell
idf.py -p COM3 flash monitor
```

---

## 3. The 6 Physical Hardware Validation Gates

When executing on real hardware, verify each gate sequentially against the serial monitor output:

```
+-------------------------------------------------------------------------------+
|                      SPECTRA C5 HARDWARE VALIDATION GATES                     |
+-------------------------------------------------------------------------------+
|  [GATE 1]  TFLM Contract & 5-Clip Minimal vs Reference Equivalence Proof     |
|  [GATE 2]  Acoustic VAD Auto-Calibration (Noise Floor Floor Clamping)         |
|  [GATE 3]  Internal SRAM & Power Budget Audit (<256 KB SRAM, DFS Active)      |
|  [GATE 4]  Trigger Action State Machine (Stream D [2, 9] & LED 3-Blink)       |
|  [GATE 5]  Network Streaming & Faster-Whisper ASR Loop (<1500 ms Latency)     |
|  [GATE 6]  Production Resilience (Multi-Channel TWDT & Boot Guard Safe Mode)  |
+-------------------------------------------------------------------------------+
```

### Gate 1: TFLM Contract & 5-Clip Equivalence Proof
- **Verification:** On boot, the firmware initializes the TensorFlow Lite Micro engine with the exact 5 minimal operators (`DEPTHWISE_CONV_2D`, `CONV_2D`, `RESHAPE`, `AVERAGE_POOL_2D`, `FULLY_CONNECTED`).
- **Expected UART Log:**
  ```text
  [SPECTRA] TFLM Model Version: 3 | Arena BSS: 81920 bytes
  [SPECTRA] Zeros-Smoke: p_pos = 0.9961 [PASS]
  [LOGITS] Clip 0: Minimal vs Reference | Max|Delta| = 0.0000 <= 0.02 [PASS]
  [LOGITS] Clip 1: Minimal vs Reference | Max|Delta| = 0.0000 <= 0.02 [PASS]
  [LOGITS] Clip 2: Minimal vs Reference | Max|Delta| = 0.0000 <= 0.02 [PASS]
  [LOGITS] Clip 3: Minimal vs Reference | Max|Delta| = 0.0000 <= 0.02 [PASS]
  [LOGITS] Clip 4: Minimal vs Reference | Max|Delta| = 0.0000 <= 0.02 [PASS]
  [LOGITS] 5-Clip Minimal Resolver Equivalence: ALL 5 PASSED
  ```

### Gate 2: Acoustic VAD Auto-Calibration
- **Verification:** During the first 2.0–2.5 seconds of boot, the microphone samples ambient room acoustic noise.
- **Expected UART Log:**
  ```text
  [VAD_CALIB] Calibration complete (20 frames): Noise mu=0.0042, sigma=0.0011 -> TH=0.0120 (raw=0.0075)
  ```
  *(Note: Clamped cleanly to lower safety bound $0.012$ in quiet rooms, or upper bound $0.060$ in noisy rooms).*

### Gate 3: Memory & Power Audit
- **Verification:** Internal SRAM footprint must remain strictly $< 256\text{ KB}$ discipline, leaving $\ge 45\text{ KB}$ free for Wi-Fi stack and DMA buffers.
- **Expected UART Log:**
  ```text
  [SPECTRA] Heap internal: 92160 / 262144 bytes free (35.2%)
  [SPECTRA] Heap PSRAM:    8257536 / 8388608 bytes free (98.4%)
  [SPECTRA] Ring Buffer Placement: PSRAM (External SPI RAM)
  [POWER_MGR] DFS configured: min=80MHz, max=240MHz, light_sleep=1
  ```

### Gate 4: Keyword Triggering & LED Confirmation
- **Verification:** The device evaluates keyword confidence every 500 ms hop.
- **Action:** Speak keyword *"Spectra"* clearly into the INMP441 microphone.
- **Expected UART Log:**
  ```text
  [SOAK] Hop: 42 | Peak: 0.742 | Voiced: YES | MFCC: 7200 us | TFLM: 15300 us | P("Spectra"): 0.9845 | Trigs: 1
  [TRIGGER] TRIG tick=42 t_ms=21000 p=[0.921,0.965,0.985] pre_ts=20000
  >>> KEYWORD TRIGGERED! Event fired on hop 42 <<<
  ```
- **Visual:** Onboard LED illuminates solidly upon trigger and stays ON for 2.0 s (4-hop cooldown window).

### Gate 5: Live Streaming & ASR Transcription
1. Start laptop ASR server:
   ```powershell
   python server/asr_server.py --port 8765 --model tiny.en
   ```
2. Once the keyword triggers, the device opens a TCP socket connection, transmits pre-roll PCM (1.0 s), followed by 100 ms live chunks until VAD endpointing or 10 s maximum duration.
3. Server prints live transcript with verified end-to-end latency:
   ```text
   [ASR_SERVER] Connected: 192.168.1.45:51234 (utt_id=1)
   [ASR_SERVER] VAD Endpoint detected (duration: 3.2s)
   [ASR_SERVER] Final transcript: "turn on the living room lights" (latency=412ms)
   ```

### Gate 6: Production Resilience & Watchdogs
- **Watchdog Supervisor:** Audio capture and Inference loops feed their respective channels. If DMA or TFLM hangs $>3.0\text{ s}$, `esp_task_wdt` initiates a clean hardware reboot.
- **NVS Boot Guard:**
  - After 30 seconds of stable operation, boot is marked healthy:
    ```text
    [BOOT_GUARD] Uptime >= 30s. System marked healthy. Crash counter reset to 0.
    ```
  - If 3 consecutive crashes occur before 30 s:
    ```text
    [BOOT_GUARD] CRITICAL: consecutive crashes = 3 >= max 3! Activating SAFE MODE.
    ```
    LED flashes 5 rapid pulses repeatedly; Wi-Fi and audio streaming are held quiescent to prevent hardware bricking.

---

## 4. On-Device UART Logits Parity Test

To verify bit-exact floating-point logits directly between laptop TFLite and on-target TFLM over serial:

```powershell
python firmware/tests/test_logits_parity.py --port COM3 --baud 115200
```

**Expected Output:**
```text
============================================================
  SPECTRA KWS -- Hardware Logits Parity Runner
  Port: COM3 | Baud: 115200
============================================================
  Sending Clip 0 ... PASS (|Delta| = 0.0000)
  Sending Clip 1 ... PASS (|Delta| = 0.0000)
  Sending Clip 2 ... PASS (|Delta| = 0.0000)
  Sending Clip 3 ... PASS (|Delta| = 0.0000)
  Sending Clip 4 ... PASS (|Delta| = 0.0000)
------------------------------------------------------------
Logits Parity Gate: ALL 5 CLIPS MATCH (Max |Delta| <= 0.0200)
STATUS: PASS
============================================================
```

---

## 5. Troubleshooting & Diagnostics

| Symptom | Probable Cause | Corrective Action |
|---|---|---|
| **Audio flatlined / peak=0.000** | Wiring loose or INMP441 missing 3.3V | Check GPIO23 (SCK), GPIO24 (WS), GPIO11 (SD). Ensure L/R is tied to GND. |
| **High acoustic noise floor ($>0.08$)** | Electrical noise on 3.3V rail | Add 100 nF ceramic bypass capacitor directly between VDD and GND on microphone breakout. |
| **Brownout reset on Wi-Fi connect** | Peak USB current spike ($>400\text{ mA}$) | Power through a high-current USB-C port or powered USB hub. |
| **Rapid 5x LED blink at boot** | Safe Mode active (3 consecutive crashes) | Read UART crash backtrace. Run `boot_guard_reset_counter()` via firmware or reflash partition table. |
| **Heap allocation failure** | PSRAM not enabled in SDK | Verify `CONFIG_SPIRAM=y` and `CONFIG_SPIRAM_MODE_OCT=y` in `sdkconfig`. |
