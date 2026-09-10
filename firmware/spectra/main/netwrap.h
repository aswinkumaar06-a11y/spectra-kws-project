/**
 * netwrap.h — Spectra Cross-Platform Network Socket Abstraction
 *
 * Provides a unified socket interface for ESP-IDF (lwIP) and Host (POSIX/Winsock).
 */

#ifndef SPECTRA_NETWRAP_H_
#define SPECTRA_NETWRAP_H_

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Initialize platform networking subsystem (e.g. WSAStartup on Windows).
 */
int netwrap_init(void);

/**
 * Cleanup networking subsystem.
 */
void netwrap_cleanup(void);

/**
 * Connect to TCP server. Returns socket file descriptor >= 0 on success, negative on error.
 */
int netwrap_connect(const char* host, uint16_t port);

/**
 * Send exact len bytes over socket. Returns bytes sent, or negative on error.
 */
int netwrap_send_exact(int sock, const void* data, size_t len);

/**
 * Receive exact len bytes over socket with optional timeout.
 * Returns bytes received, 0 on disconnect, or negative on error/timeout.
 */
int netwrap_recv_exact(int sock, void* data, size_t len, int timeout_ms);

/**
 * Close socket descriptor.
 */
void netwrap_close(int sock);

#ifdef __cplusplus
}
#endif

#endif  // SPECTRA_NETWRAP_H_
