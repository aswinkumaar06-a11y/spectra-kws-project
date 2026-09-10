/**
 * netwrap.cc — Spectra Cross-Platform Network Socket Implementation
 */

#include "netwrap.h"
#include <cstdio>
#include <cstring>

#if defined(ESP_PLATFORM)
#include "esp_log.h"
#include "lwip/sockets.h"
#include "lwip/netdb.h"
#include <unistd.h>
static const char* TAG = "NETWRAP";

int netwrap_init(void) {
    return 0;
}

void netwrap_cleanup(void) {}

void netwrap_close(int sock) {
    if (sock >= 0) {
        close(sock);
    }
}

#elif defined(_WIN32) || defined(__CYGWIN__)
#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#define close(s) closesocket(s)
static const char* TAG = "NETWRAP";

static bool s_wsa_initialized = false;

int netwrap_init(void) {
    if (!s_wsa_initialized) {
        WSADATA wsa;
        if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) {
            return -1;
        }
        s_wsa_initialized = true;
    }
    return 0;
}

void netwrap_cleanup(void) {
    if (s_wsa_initialized) {
        WSACleanup();
        s_wsa_initialized = false;
    }
}

void netwrap_close(int sock) {
    if (sock >= 0) {
        closesocket((SOCKET)sock);
    }
}

#else
#include <sys/socket.h>
#include <netdb.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/time.h>
static const char* TAG = "NETWRAP";

int netwrap_init(void) { return 0; }
void netwrap_cleanup(void) {}
void netwrap_close(int sock) {
    if (sock >= 0) close(sock);
}
#endif

int netwrap_connect(const char* host, uint16_t port) {
    netwrap_init();

    char port_str[16];
    snprintf(port_str, sizeof(port_str), "%u", (unsigned)port);

    struct addrinfo hints;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;

    struct addrinfo* res = NULL;
    int err = getaddrinfo(host, port_str, &hints, &res);
    if (err != 0 || res == NULL) {
        return -1;
    }

    int sock = (int)socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (sock < 0) {
        freeaddrinfo(res);
        return -2;
    }

    if (connect(sock, res->ai_addr, (int)res->ai_addrlen) != 0) {
        netwrap_close(sock);
        freeaddrinfo(res);
        return -3;
    }

    freeaddrinfo(res);
    return sock;
}

int netwrap_send_exact(int sock, const void* data, size_t len) {
    if (sock < 0 || data == NULL) return -1;

    size_t total_sent = 0;
    const char* ptr = (const char*)data;

    while (total_sent < len) {
        int sent = send(sock, ptr + total_sent, (int)(len - total_sent), 0);
        if (sent <= 0) {
            return -2;
        }
        total_sent += sent;
    }

    return (int)total_sent;
}

int netwrap_recv_exact(int sock, void* data, size_t len, int timeout_ms) {
    if (sock < 0 || data == NULL) return -1;

    if (timeout_ms > 0) {
#if defined(_WIN32)
        DWORD tv = timeout_ms;
        setsockopt((SOCKET)sock, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tv, sizeof(tv));
#else
        struct timeval tv;
        tv.tv_sec = timeout_ms / 1000;
        tv.tv_usec = (timeout_ms % 1000) * 1000;
        setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, (const void*)&tv, sizeof(tv));
#endif
    }

    size_t total_recvd = 0;
    char* ptr = (char*)data;

    while (total_recvd < len) {
        int recvd = recv(sock, ptr + total_recvd, (int)(len - total_recvd), 0);
        if (recvd == 0) {
            return 0; // Peer disconnected
        }
        if (recvd < 0) {
            return -2; // Error or timeout
        }
        total_recvd += recvd;
    }

    return (int)total_recvd;
}
