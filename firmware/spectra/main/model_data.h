/**
 * model_data.h — Spectra KWS TFLite Model Data Declarations
 *
 * Declares the INT8-quantized TFLite FlatBuffer model embedded as a
 * C byte array. The array is defined in model_data.cc with 16-byte
 * alignment for efficient DMA and SIMD access on ESP32-S3.
 */

#ifndef MODEL_DATA_H_
#define MODEL_DATA_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The TFLite FlatBuffer model data.
 * Aligned to 16 bytes for optimal memory access on ESP32-S3.
 */
extern const unsigned char spectra_model[];

/**
 * Size of the TFLite FlatBuffer model in bytes.
 * Expected value: 39552 (see spectra_config.h SPECTRA_MODEL_SIZE_BYTES).
 */
extern const unsigned int spectra_model_len;

#ifdef __cplusplus
}
#endif

#endif  // MODEL_DATA_H_
