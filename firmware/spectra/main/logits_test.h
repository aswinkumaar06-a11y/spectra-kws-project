/**
 * logits_test.h — Phase 3 Equivalence Proof & UART Parity Protocol
 *
 * Implements:
 *   1. 5-clip on-device equivalence proof (minimal-vs-reference bit-identical validation)
 *   2. Measured memory budget accounting (< 256 KB internal SRAM discipline)
 *   3. UART streaming receiver for 50-clip golden parity verification
 *
 * Target: Seeed Studio XIAO ESP32-C5 (RISC-V @ 240 MHz)
 */

#ifndef SPECTRA_LOGITS_TEST_H_
#define SPECTRA_LOGITS_TEST_H_

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Run autonomous 5-clip equivalence proof on boot:
 * Evaluates 2 positive, 1 normal negative, and 2 synthetic hard negative clips.
 * Asserts bit-identical raw INT8 logits and zero classification error.
 */
bool logits_test_run_equivalence_5(void);

/**
 * Print measured memory budget table on-device:
 * Audits all memory segments against the 256 KB internal SRAM discipline.
 */
void logits_test_print_memory_budget(void);

/**
 * Process a single UART packet from host test runner if data is available.
 * Returns true if a packet was received and answered.
 */
bool logits_test_poll_uart(void);

#ifdef __cplusplus
}
#endif

#endif  // SPECTRA_LOGITS_TEST_H_
