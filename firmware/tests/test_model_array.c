/**
 * test_model_array.c — Host-side C validation of model data array.
 *
 * Compiles on x86 (gcc/MSVC) without ESP-IDF.
 * Validates:
 *   1. Model array length matches expected size (39552 bytes)
 *   2. FlatBuffer magic bytes ("TFL3") at offset 4-7
 *   3. Array is non-null and first byte is readable
 *
 * Build:  gcc -o test_model_array test_model_array.c -I ../spectra/main
 * Run:    ./test_model_array
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* Bring in the model declarations */
#include "model_data.h"

/* Expected values from spectra_config.h */
#define EXPECTED_MODEL_SIZE  39552
#define TFLITE_MAGIC_OFFSET  4
#define TFLITE_MAGIC_BYTE_0  0x54  /* 'T' */
#define TFLITE_MAGIC_BYTE_1  0x46  /* 'F' */
#define TFLITE_MAGIC_BYTE_2  0x4C  /* 'L' */
#define TFLITE_MAGIC_BYTE_3  0x33  /* '3' */

static int tests_passed = 0;
static int tests_failed = 0;

#define CHECK(cond, msg)                                          \
    do {                                                          \
        if (cond) {                                               \
            printf("[PASS] %s\n", msg);                           \
            tests_passed++;                                       \
        } else {                                                  \
            printf("[FAIL] %s\n", msg);                           \
            tests_failed++;                                       \
        }                                                         \
    } while (0)

int main(void) {
    printf("=== Spectra Model Array Validation ===\n\n");

    /* Test 1: Array is non-null */
    CHECK(spectra_model != NULL, "Model array is non-null");

    /* Test 2: Length matches expected size */
    printf("  Model length: %u bytes (expected %d)\n",
           spectra_model_len, EXPECTED_MODEL_SIZE);
    CHECK(spectra_model_len == EXPECTED_MODEL_SIZE,
          "Model length matches expected size");

    /* Test 3: FlatBuffer magic bytes at offset 4 */
    if (spectra_model_len > TFLITE_MAGIC_OFFSET + 4) {
        uint8_t m0 = spectra_model[TFLITE_MAGIC_OFFSET + 0];
        uint8_t m1 = spectra_model[TFLITE_MAGIC_OFFSET + 1];
        uint8_t m2 = spectra_model[TFLITE_MAGIC_OFFSET + 2];
        uint8_t m3 = spectra_model[TFLITE_MAGIC_OFFSET + 3];
        printf("  Magic bytes: 0x%02x 0x%02x 0x%02x 0x%02x ('%c%c%c%c')\n",
               m0, m1, m2, m3, m0, m1, m2, m3);
        CHECK(m0 == TFLITE_MAGIC_BYTE_0 &&
              m1 == TFLITE_MAGIC_BYTE_1 &&
              m2 == TFLITE_MAGIC_BYTE_2 &&
              m3 == TFLITE_MAGIC_BYTE_3,
              "FlatBuffer magic 'TFL3' at offset 4");
    } else {
        printf("[FAIL] Model too small to contain magic bytes\n");
        tests_failed++;
    }

    /* Test 4: First and last bytes are readable (no segfault) */
    uint8_t first = spectra_model[0];
    uint8_t last = spectra_model[spectra_model_len - 1];
    printf("  First byte: 0x%02x, Last byte: 0x%02x\n", first, last);
    CHECK(1, "Array boundary access (first + last byte readable)");

    /* Summary */
    printf("\n=== Results: %d passed, %d failed ===\n",
           tests_passed, tests_failed);

    return tests_failed > 0 ? 1 : 0;
}
