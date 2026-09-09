"""
Quick microphone diagnostic - tests all input devices to find a working one.
Speak or clap loudly while this runs!
"""
import sounddevice as sd
import numpy as np

print("=" * 60)
print("  Microphone Diagnostic - Speak or clap during the test!")
print("=" * 60)

devices = sd.query_devices()
input_devices = [(i, d) for i, d in enumerate(devices) if d['max_input_channels'] > 0]

print(f"\nFound {len(input_devices)} input devices. Testing each for 1 second...\n")

working = []

for idx, dev in input_devices:
    name = dev['name']
    try:
        audio = sd.rec(16000, samplerate=16000, channels=1, dtype='float32',
                       device=idx, blocking=True)
        max_amp = float(np.max(np.abs(audio)))
        status = "WORKING!" if max_amp > 0.001 else "silent"
        print(f"  Device {idx:2d}: {name[:55]:55s} -> {max_amp:.6f}  {status}")
        if max_amp > 0.001:
            working.append((idx, name, max_amp))
    except Exception as e:
        print(f"  Device {idx:2d}: {name[:55]:55s} -> ERROR: {e}")

print("\n" + "=" * 60)
if working:
    best = max(working, key=lambda x: x[2])
    print(f"  Best device: #{best[0]} - {best[1]}")
    print(f"  Max amplitude: {best[2]:.6f}")
    print(f"\n  To use this device in the live test, run:")
    print(f"    python scripts\\test_spectra_model.py --mode live --device {best[0]}")
else:
    print("  No working microphone detected!")
    print("  Please check:")
    print("    1. Right-click speaker icon -> Sound settings -> Input")
    print("    2. Make sure a microphone is selected and volume is not 0")
    print("    3. Speak into the mic - the level meter should move")
    print("    4. Ensure 'Let desktop apps access your microphone' is ON")
    print("       (Settings -> Privacy & Security -> Microphone)")
print("=" * 60)
