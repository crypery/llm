/*
 * Logging module.
 *
 * Timestamped logging to console, file, both, or none with configurable
 * severity levels. Configure with log_init(), write messages through the
 * LOG_* macros or log_msg(), and shut down with log_close(). Setting
 * LOG_OUTPUT_NONE disables all output.
 */

#ifndef LOG_H
#define LOG_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdarg.h>
#include <stdio.h>
#include <time.h>

/* Default log file name */
#define DEFAULT_LOG_FILE "server.log"

/* Severity levels, from most to least verbose */
typedef enum {
    LOG_LEVEL_DEBUG = 0,
    LOG_LEVEL_INFO  = 1,
    LOG_LEVEL_WARN  = 2,
    LOG_LEVEL_ERROR = 3
} LogLevel;

/* Output destinations */
typedef enum {
    LOG_OUTPUT_NONE    = 0,  /* Disable logging */
    LOG_OUTPUT_BOTH    = 1,  /* Console and file */
    LOG_OUTPUT_CONSOLE = 2,  /* Console only */
    LOG_OUTPUT_FILE    = 3   /* File only */
} LogOutput;

/* Initialize logging.
 * @param output LogOutput destination selector.
 * @param level Minimum LogLevel to record.
 * @param filename Path of the log file (ignored without file output). */
void log_init(int output, int level, const char *filename);

/* Close logging and flush any pending output. */
void log_close(void);

/* Callback invoked with every formatted log line (e.g. WebSocket broadcast) */
typedef void (*log_cb_t)(const char *msg);

/* Install a callback that receives each formatted log line.
 * @param cb Callback, or NULL to remove it. */
void log_set_callback(log_cb_t cb);

/* Write a formatted message at a given level.
 * @param level Severity of the message.
 * @param fmt printf-style format string.
 * @param ... Format arguments. */
void log_msg(LogLevel level, const char *fmt, ...);

/* Write a formatted message at a given level from a va_list.
 * @param level Severity of the message.
 * @param fmt printf-style format string.
 * @param args Format arguments (owned by the caller). */
void log_vmsg(LogLevel level, const char *fmt, va_list args);

/* Convenience macros, one per severity level */
#define LOG_INFO(fmt, ...)  log_msg(LOG_LEVEL_INFO,  fmt, ##__VA_ARGS__)
#define LOG_ERROR(fmt, ...) log_msg(LOG_LEVEL_ERROR, fmt, ##__VA_ARGS__)
#define LOG_WARN(fmt, ...)  log_msg(LOG_LEVEL_WARN,  fmt, ##__VA_ARGS__)
#define LOG_DEBUG(fmt, ...) log_msg(LOG_LEVEL_DEBUG, fmt, ##__VA_ARGS__)

#ifdef __cplusplus
}
#endif

#endif /* LOG_H */