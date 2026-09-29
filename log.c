/*
 * Git: https://github.com/crypery
 * Author: https://crypery.com
 * License: GNU AGPL v3 (Affero GPL)
 */

/*
 * Logging module.
 *
 * Timestamped log output to console, file, both, or none, with configurable
 * severity levels (DEBUG, INFO, WARN, ERROR). Uses thread-safe Win32
 * GetLocalTime instead of locale-dependent CRT localtime/strftime.
 *
 * Example:
 *   log_init(LOG_OUTPUT_FILE, LOG_LEVEL_INFO, "server.log");
 *   LOG_INFO("server started on port %d\n", 443);
 *   LOG_ERROR("request failed: %s\n", reason);
 *   log_close();
 */
#include "log.h"
#include "compat.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Log module globals */
static FILE *g_log_file = NULL;
static LogOutput g_log_output = LOG_OUTPUT_NONE;
static LogLevel g_log_level = LOG_LEVEL_INFO;
static char g_log_path[512] = DEFAULT_LOG_FILE;
static int g_log_open = 0;
static log_cb_t g_log_cb = NULL;

/* Install a callback that receives each formatted log line */
void log_set_callback(log_cb_t cb) {
    g_log_cb = cb;
}

/* Format the current local date/time into buf (thread-safe) */
static void log_timestamp(char *buf, size_t buf_len) {
    time_t now = time(NULL);
    struct tm tmv;
#ifdef _WIN32
    localtime_s(&tmv, &now);
#else
    localtime_r(&now, &tmv);
#endif
    snprintf(buf, buf_len, "%04d-%02d-%02d %02d:%02d:%02d",
             tmv.tm_year + 1900, tmv.tm_mon + 1, tmv.tm_mday,
             tmv.tm_hour, tmv.tm_min, tmv.tm_sec);
}

/* Return the string name of a log level */
static const char *log_level_str(LogLevel level) {
    switch (level) {
        case LOG_LEVEL_DEBUG: return "DEBUG";
        case LOG_LEVEL_INFO:  return "INFO";
        case LOG_LEVEL_WARN:  return "WARN";
        case LOG_LEVEL_ERROR: return "ERROR";
        default:              return "UNKNOWN";
    }
}

/* Initialize logging with output mode, level, and file */
void log_init(int output, int level, const char *filename) {
    if (g_log_open) log_close();

    g_log_output = LOG_OUTPUT_NONE;
    if (output == LOG_OUTPUT_BOTH) g_log_output = LOG_OUTPUT_BOTH;
    else if (output == LOG_OUTPUT_CONSOLE) g_log_output = LOG_OUTPUT_CONSOLE;
    else if (output == LOG_OUTPUT_FILE) g_log_output = LOG_OUTPUT_FILE;

    if (level >= LOG_LEVEL_DEBUG && level <= LOG_LEVEL_ERROR) {
        g_log_level = (LogLevel)level;
    } else {
        g_log_level = LOG_LEVEL_INFO;
    }

    if (filename && filename[0]) {
        strncpy(g_log_path, filename, sizeof(g_log_path) - 1);
        g_log_path[sizeof(g_log_path) - 1] = '\0';
    }

    if (g_log_output == LOG_OUTPUT_FILE || g_log_output == LOG_OUTPUT_BOTH) {
        g_log_file = fopen(g_log_path, "a");
        if (g_log_file) {
            char ts[64];
            log_timestamp(ts, sizeof(ts));
            fprintf(g_log_file, "[%s] [INFO] [main] log started: %s\n", ts, g_log_path);
            fflush(g_log_file);
        }
    }
    g_log_open = 1;
}

/* Close log file and flush */
void log_close(void) {
    if (g_log_file) {
        char ts[64];
        log_timestamp(ts, sizeof(ts));
        fprintf(g_log_file, "[%s] [INFO] [main] log closed\n", ts);
        fflush(g_log_file);
        fclose(g_log_file);
        g_log_file = NULL;
    }
    g_log_open = 0;
}

/* Write a log message at given level */
void log_vmsg(LogLevel level, const char *fmt, va_list args) {
    if ((int)level < (int)g_log_level) return;

    char ts[64];
    log_timestamp(ts, sizeof(ts));

    char msg[1024];
    vsnprintf(msg, sizeof(msg), fmt, args);

    size_t ml = strlen(msg);
    while (ml > 0 && (msg[ml-1] == '\n' || msg[ml-1] == '\r')) msg[--ml] = '\0';

    char full_msg[1200];
    snprintf(full_msg, sizeof(full_msg), "[%s] [%s] %s\n",
             ts, log_level_str(level), msg);

    if (g_log_cb) g_log_cb(full_msg);

    if (g_log_output == LOG_OUTPUT_NONE) return;

    if (g_log_output == LOG_OUTPUT_FILE || g_log_output == LOG_OUTPUT_BOTH) {
        if (g_log_file) {
            fprintf(g_log_file, "%s", full_msg);
            fflush(g_log_file);
        }
    }

    if (g_log_output == LOG_OUTPUT_CONSOLE || g_log_output == LOG_OUTPUT_BOTH) {
        fprintf(stderr, "%s", full_msg);
        fflush(stderr);
    }
}

/* Write a formatted log message at given level */
void log_msg(LogLevel level, const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    log_vmsg(level, fmt, args);
    va_end(args);
}