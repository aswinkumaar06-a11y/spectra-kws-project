/**
 * mfcc.h — Spectra KWS MFCC Feature Extraction Engine
 *
 * Target: XIAO ESP32-C5 (RISC-V @ 240 MHz) & Host x86/x64
 *
 * Fixed-point / float32 hybrid architecture:
 *   - 16,000 samples int16 PCM (1.0s window)
 *   - Energy gate: threshold >= 0.03 peak amplitude
 *   - Peak normalization: [-1.0, 1.0]
 *   - Zero padding: 512 zeros each side (17,024 total samples)
 *   - 32 frames x 1024-point periodic Hann window (hop=512)
 *   - Radix-2 DIT FFT -> 513 power spectrum bins
 *   - Sparse Slaney Mel filterbank (128 mel bins)
 *   - Log power & top_db=80 clip relative to clip-global max
 *   - DCT Type-II orthonormal projection -> 40 MFCCs
 *   - Banker's round-half-even INT8 quantization (scale=3.9962144, zp=60)
 */

#ifndef SPECTRA_MFCC_H_
#define SPECTRA_MFCC_H_

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#include "spectra_config.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Initialize the MFCC feature extraction engine.
 * Precomputes bit-reversal lookup tables for 1024-point FFT.
 */
void mfcc_init(void);

/**
 * Check if audio window contains speech energy (peak amplitude >= 0.03).
 *
 * @param window_16k Pointer to 16,000 int16 PCM samples (1.0s window)
 * @param out_peak_amp Optional output for unnormalized peak amplitude [0.0, 1.0]
 * @return true if peak >= SPECTRA_ENERGY_GATE_THRESHOLD (0.03), false otherwise.
 */
bool mfcc_check_energy_gate(const int16_t* window_16k, float* out_peak_amp);

/**
 * Execute the complete 8-stage MFCC feature extraction pipeline.
 *
 * @param window_16k Pointer to 16,000 int16 PCM samples (1.0s audio window)
 * @param out_mfcc_float Output buffer for 40x32 float32 features (can be NULL)
 * @param out_mfcc_int8 Output buffer for 40x32 INT8 quantized features (can be NULL)
 * @return true if window had sufficient energy and was processed, false if gated.
 */
bool mfcc_process_window(const int16_t* window_16k,
                         float* out_mfcc_float,
                         int8_t* out_mfcc_int8);

/**
 * Banker's round-half-even affine INT8 quantization.
 * q = clip(round_half_even(x / scale) + zero_point, -128, 127)
 */
int8_t mfcc_quantize_sample(float val, float scale, int zero_point);

#ifdef __cplusplus
}
#endif

#endif  // SPECTRA_MFCC_H_
