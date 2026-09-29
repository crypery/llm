/*
 * Git: https://github.com/crypery
 * Author: https://crypery.com
 * License: GNU AGPL v3 (Affero GPL)
 */

/*
 * compat.h - portability layer for the mini-c-server modules.
 *
 * Lets log, sett, server, and ssl_test compile and run on both
 * Windows (Winsock + WinAPI) and POSIX (Berkeley sockets + pthreads).
 *
 * On Windows the native APIs are used: SOCKET, WSA* errors, CRITICAL_SECTION,
 * HANDLE threads/events, GetTickCount, GetLocalTime/GetSystemTime.
 * On POSIX these are mapped to close()/errno, pthread mutexes/threads/conds,
 * clock_gettime(CLOCK_MONOTONIC) and gmtime_r/localtime_r.
 *
 * The http module (http.h) provides its own self-contained copy of the socket,
 * string and thread helpers and defines HTTP_PLAT_DEFINED, so the overlapping
 * definitions below are skipped when it has been included already.
 */

#ifndef COMPAT_H
#define COMPAT_H

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <direct.h>
#include <stdlib.h>
#include <string.h>
#else
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/select.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <unistd.h>
#include <pthread.h>
#include <errno.h>
#include <strings.h>
#include <string.h>
#include <time.h>
#include <stdlib.h>
#endif

#ifndef HTTP_PLAT_DEFINED
#define HTTP_PLAT_DEFINED

#ifdef _WIN32
#else
/* --- socket aliases --- */
typedef int SOCKET;
#define INVALID_SOCKET (-1)
#define SOCKET_ERROR   (-1)
#define closesocket    close
#define WSAGetLastError() (errno)
#define WSAEWOULDBLOCK EWOULDBLOCK
#define WSAETIMEDOUT   EAGAIN
#define SD_BOTH        SHUT_RDWR
#define SD_SEND        SHUT_WR
#define SD_RECV        SHUT_RD

/* --- string aliases --- */
#define _stricmp   strcasecmp
#define _strnicmp  strncasecmp
#define strtok_s   strtok_r
#endif /* _WIN32 */

/* Monotonic tick counter in milliseconds.
 * @return Milliseconds since an arbitrary epoch. */
static inline unsigned long compat_tick(void) {
#ifdef _WIN32
    return (unsigned long)GetTickCount();
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (unsigned long)(ts.tv_sec * 1000UL + ts.tv_nsec / 1000000UL);
#endif
}

/* Initialize the socket layer (WSAStartup on Windows, no-op on POSIX).
 * @return 0 on success. */
static inline int sock_init(void) {
#ifdef _WIN32
    WSADATA wsa;
    return WSAStartup(MAKEWORD(2, 2), &wsa);
#else
    return 0;
#endif
}

/* Tear down the socket layer. */
static inline void sock_cleanup(void) {
#ifdef _WIN32
    WSACleanup();
#endif
}

/* --- mutual exclusion --- */
#ifdef _WIN32
typedef CRITICAL_SECTION mtx_t;
static inline void mtx_init(mtx_t *m)   { InitializeCriticalSection(m); }
static inline void mtx_lock(mtx_t *m)   { EnterCriticalSection(m); }
static inline void mtx_unlock(mtx_t *m) { LeaveCriticalSection(m); }
static inline void mtx_destroy(mtx_t *m){ DeleteCriticalSection(m); }
#else
typedef pthread_mutex_t mtx_t;
static inline void mtx_init(mtx_t *m)   { pthread_mutex_init(m, NULL); }
static inline void mtx_lock(mtx_t *m)   { pthread_mutex_lock(m); }
static inline void mtx_unlock(mtx_t *m) { pthread_mutex_unlock(m); }
static inline void mtx_destroy(mtx_t *m){ pthread_mutex_destroy(m); }
#endif

/* --- threads --- */
typedef void *(*thr_fn_t)(void *);

#ifdef _WIN32
typedef HANDLE thr_t;
typedef struct { thr_fn_t fn; void *arg; } thr_arg_t;
static DWORD WINAPI thr_trampoline(LPVOID p) {
    thr_arg_t *ta = (thr_arg_t *)p;
    ta->fn(ta->arg);
    free(ta);
    return 0;
}
static inline int thr_create(thr_t *t, thr_fn_t fn, void *arg) {
    thr_arg_t *ta = (thr_arg_t *)malloc(sizeof(thr_arg_t));
    if (!ta) return -1;
    ta->fn = fn; ta->arg = arg;
    *t = CreateThread(NULL, 0, thr_trampoline, ta, 0, NULL);
    if (*t == NULL) { free(ta); return -1; }
    return 0;
}
static inline void thr_join(thr_t *t) { WaitForSingleObject(*t, INFINITE); CloseHandle(*t); }
#else
typedef pthread_t thr_t;
static inline int thr_create(thr_t *t, thr_fn_t fn, void *arg) {
    return pthread_create(t, NULL, fn, arg);
}
static inline void thr_join(thr_t *t) { pthread_join(*t, NULL); }
#endif

/* --- auto-reset event (one pending wakeup) --- */
#ifdef _WIN32
typedef HANDLE evt_t;
static inline void evt_init(evt_t *e)    { *e = CreateEvent(NULL, FALSE, FALSE, NULL); }
static inline void evt_signal(evt_t *e)  { SetEvent(*e); }
static inline void evt_wait(evt_t *e)    { WaitForSingleObject(*e, INFINITE); }
static inline void evt_destroy(evt_t *e) { CloseHandle(*e); }
#else
typedef struct { pthread_mutex_t m; pthread_cond_t c; int flag; } evt_t;
static inline void evt_init(evt_t *e) {
    pthread_mutex_init(&e->m, NULL);
    pthread_cond_init(&e->c, NULL);
    e->flag = 0;
}
static inline void evt_signal(evt_t *e) {
    pthread_mutex_lock(&e->m);
    e->flag = 1;
    pthread_cond_signal(&e->c);
    pthread_mutex_unlock(&e->m);
}
static inline void evt_wait(evt_t *e) {
    pthread_mutex_lock(&e->m);
    while (!e->flag) pthread_cond_wait(&e->c, &e->m);
    e->flag = 0;
    pthread_mutex_unlock(&e->m);
}
static inline void evt_destroy(evt_t *e) {
    pthread_mutex_destroy(&e->m);
    pthread_cond_destroy(&e->c);
}
#endif

#endif /* !HTTP_PLAT_DEFINED */

/* Sleep for ms milliseconds. */
static inline void compat_sleep_ms(int ms) {
#ifdef _WIN32
    Sleep((DWORD)(ms < 0 ? 0 : ms));
#else
    struct timespec ts;
    ts.tv_sec = (ms < 0 ? 0 : ms) / 1000;
    ts.tv_nsec = (long)((ms < 0 ? 0 : ms) % 1000) * 1000000L;
    nanosleep(&ts, NULL);
#endif
}

/* Create a directory (with parents created by the caller).
 * @param path Directory path.
 * @return 0 on success. */
static inline int compat_mkdir(const char *path) {
#ifdef _WIN32
    return _mkdir(path);
#else
    return mkdir(path, 0755);
#endif
}

/* Apply a socket I/O timeout (SO_RCVTIMEO/SO_SNDTIMEO).
 * @param s Socket.
 * @param ms Timeout in milliseconds. */
static inline void compat_set_io_timeout(SOCKET s, int ms) {
#ifdef _WIN32
    int to = ms;
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, (const char *)&to, sizeof(to));
    setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, (const char *)&to, sizeof(to));
#else
    struct timeval to;
    to.tv_sec = ms / 1000;
    to.tv_usec = (ms % 1000) * 1000;
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, &to, sizeof(to));
    setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, &to, sizeof(to));
#endif
}

#ifdef __cplusplus
}
#endif

#endif /* COMPAT_H */