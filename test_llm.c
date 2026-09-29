/*
 * Git: https://github.com/crypery
 * Author: https://crypery.com
 * License: GNU AGPL v3 (Affero GPL)
 */


// chat.c - example chat client built on the llm module.
//
// Example:
//   chat.exe -m Qwen3.8-2B-Q4_K_M.gguf -th 8 -c 4096 -t 0.1 -l ru -p "Привет, как дела?"
//
// -p text is always handled as UTF-8: on Windows the command line is taken
// from GetCommandLineW() (UTF-16) and each argument is converted to UTF-8,
// so Cyrillic text survives regardless of the console codepage.
//
// --agent <file> points to the initial agent prompt file (default: agent.md).
// Its content is sent to the model as a system message right after
// llm_init(), before any user prompt.

#include "llm.h"
#include "tools.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#endif

#define DEFAULT_AGENT     "agent.md"
#define DEFAULT_N_PREDICT 512

#ifdef _WIN32
static double tnow_ms(void) {
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double) c.QuadPart * 1000.0 / (double) f.QuadPart;
}
#define CHAT_TIMING getenv("CHAT_TIMING") != NULL
#else
#include <time.h>
static double tnow_ms(void) { return (double) clock() * 1000.0 / (double) CLOCKS_PER_SEC; }
#define CHAT_TIMING getenv("CHAT_TIMING") != NULL
#endif

// ---------------------------------------------------------------------------
// Windows: rebuild argv from the UTF-16 command line (UTF-8 output)
// ---------------------------------------------------------------------------
#ifdef _WIN32
static wchar_t ** wsplit(const wchar_t * s, int * n_out) {
    int cap = 32, n = 0;
    wchar_t ** v = (wchar_t **) malloc(sizeof(wchar_t *) * cap);
    if (!v) { *n_out = 0; return NULL; }
    for (;;) {
        while (*s == ' ' || *s == '\t') s++;
        if (!*s) break;
        const wchar_t * start = s;
        if (*s == L'"') {
            s++;
            while (*s && *s != L'"') s++;
            if (*s) s++;
        } else {
            while (*s && *s != ' ' && *s != '\t') s++;
        }
        size_t len = (size_t)(s - start);
        if (n == cap) {
            cap *= 2;
            wchar_t ** nv = (wchar_t **) realloc(v, sizeof(wchar_t *) * cap);
            if (!nv) break;
            v = nv;
        }
        v[n] = (wchar_t *) malloc((len + 1) * sizeof(wchar_t));
        if (!v[n]) break;
        memcpy(v[n], start, len * sizeof(wchar_t));
        v[n][len] = 0;
        n++;
    }
    *n_out = n;
    return v;
}

static char * w2utf8(const wchar_t * w) {
    int n = WideCharToMultiByte(CP_UTF8, 0, w, -1, NULL, 0, NULL, NULL);
    if (n <= 0) return strdup("");
    char * s = (char *) malloc(n);
    if (!s) return strdup("");
    WideCharToMultiByte(CP_UTF8, 0, w, -1, s, n, NULL, NULL);
    return s;
}
#endif

// ---------------------------------------------------------------------------
// UTF-8 stdout helper: write wide characters when attached to a console and
// raw UTF-8 bytes when redirected.
// ---------------------------------------------------------------------------
static void out_write(const char * s, size_t n) {
    if (n == 0) return;
#ifdef _WIN32
    HANDLE h = GetStdHandle(STD_OUTPUT_HANDLE);
    static int is_console = -1;
    if (is_console < 0) is_console = (GetFileType(h) == FILE_TYPE_CHAR);
    if (is_console) {
        int wlen = MultiByteToWideChar(CP_UTF8, 0, s, (int) n, NULL, 0);
        if (wlen <= 0) return;
        static wchar_t * wb = NULL;
        static size_t wcap = 0;
        if ((size_t) wlen > wcap) {
            free(wb);
            wb = (wchar_t *) malloc(((size_t) wlen) * sizeof(wchar_t));
            if (!wb) { wcap = 0; return; }
            wcap = (size_t) wlen;
        }
        if (!MultiByteToWideChar(CP_UTF8, 0, s, (int) n, wb, wlen)) return;
        DWORD wwritten;
        WriteConsoleW(h, wb, wlen, &wwritten, NULL);
    } else {
        DWORD written;
        WriteFile(h, s, (DWORD) n, &written, NULL);
    }
#else
    fwrite(s, 1, n, stdout);
    fflush(stdout);
#endif
}

static void out_str(const char * s) {
    if (s) out_write(s, strlen(s));
}

static void out_printf(const char * fmt, ...) {
    va_list ap, ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    int n = vsnprintf(NULL, 0, fmt, ap2);
    va_end(ap2);
    if (n < 0) { va_end(ap); return; }
    char * buf = (char *) malloc((size_t) n + 1);
    if (!buf) { va_end(ap); return; }
    vsnprintf(buf, (size_t) n + 1, fmt, ap);
    va_end(ap);
    out_write(buf, (size_t) n);
    free(buf);
}

// ---------------------------------------------------------------------------
// NOTE: per-token streaming, thinking-block stripping and end-marker
// handling now live in llm.c (chat_stream_cb / chat_answer_start /
// chat_strip_thinking). This client uses the llm_chat_* API below and only
// prints the final reply.
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// streaming: buffer tokens, strip the leading thinking block and any
// <|im_end|> / <|endoftext|> markers, print only the assistant's answer
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Dialog history and KV-cache reuse are handled by the llm module
// (llm_chat_add / llm_chat_generate / llm_chat_warmup in llm.c). This client
// only adds user messages and prints the returned replies.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// one chat turn: add the user message and generate the assistant reply.
// History accumulation and KV-cache reuse across turns are handled inside
// llm.c (llm_chat_add / llm_chat_generate). Unlike llm_chat_turn() this does
// NOT reset to the warmup state first, so multi-turn dialogs keep context.
// Returns the reply text (caller frees), NULL on error.
// ---------------------------------------------------------------------------
static char * chat_turn(llm_t * l, const char * user_text, int n_predict) {
    llm_chat_add(l, "user", user_text);
    return llm_chat_generate(l, n_predict);
}

// ---------------------------------------------------------------------------
// system tools (enabled with --tools): volume control and app launching
// ---------------------------------------------------------------------------
#define TOOL_NO_ARGS "{\"type\":\"object\",\"properties\":{}}"
#define TOOL_SET_VOL_PARAMS \
    "{\"type\":\"object\",\"properties\":{\"volume\":{\"type\":\"integer\"," \
    "\"minimum\":0,\"maximum\":100}},\"required\":[\"volume\"]}"
#define TOOL_LAUNCH_PARAMS \
    "{\"type\":\"object\",\"properties\":{\"app\":{\"type\":\"string\"," \
    "\"description\":\"app name in English or transliteration, e.g. notepad, calc\"}}," \
    "\"required\":[\"app\"]}"
static llm_tool_t system_tools[] = {
    { "volume_up",   "Increase the system volume by 10%.",            TOOL_NO_ARGS },
    { "volume_down", "Decrease the system volume by 10%.",            TOOL_NO_ARGS },
    { "volume_get",  "Report the current system volume (0-100).",     TOOL_NO_ARGS },
    { "volume_set",  "Set the system volume (volume, 0-100).",        TOOL_SET_VOL_PARAMS },
    { "volume_on",   "Turn the sound on (unmute).",                   TOOL_NO_ARGS },
    { "volume_off",  "Turn the sound off (mute).",                    TOOL_NO_ARGS },
    { "launch_app",  "Launch an application by name (like the Run dialog).", TOOL_LAUNCH_PARAMS },
};

static char * exec_tool(const char * name, const char * args) {
    char res[256];
    int v;
    if (!strcmp(name, "volume_up")) {
        v = tools_volume_step(10);
        if (v < 0) return strdup("error: cannot change volume");
        snprintf(res, sizeof(res), "volume now %d%%", v);
        return strdup(res);
    }
    if (!strcmp(name, "volume_down")) {
        v = tools_volume_step(-10);
        if (v < 0) return strdup("error: cannot change volume");
        snprintf(res, sizeof(res), "volume now %d%%", v);
        return strdup(res);
    }
    if (!strcmp(name, "volume_get")) {
        v = tools_volume_get();
        if (v < 0) return strdup("error: cannot read volume");
        snprintf(res, sizeof(res), "volume is %d%%", v);
        return strdup(res);
    }
    if (!strcmp(name, "volume_set")) {
        json_value_t * root = json_parse(args);
        json_value_t * e = root ? json_obj_get(root, "volume") : NULL;
        int vol = (e && e->type == JSON_NUM) ? (int) e->u.num : -1;
        if (root) json_free(root);
        if (vol < 0 || vol > 100) return strdup("error: volume must be 0..100");
        if (tools_volume_set(vol) != 0) return strdup("error: cannot set volume");
        snprintf(res, sizeof(res), "volume set to %d%%", vol);
        return strdup(res);
    }
    if (!strcmp(name, "volume_on")) {
        if (tools_mute_set(0) != 0) return strdup("error: cannot unmute");
        return strdup("sound is now ON");
    }
    if (!strcmp(name, "volume_off")) {
        if (tools_mute_set(1) != 0) return strdup("error: cannot mute");
        return strdup("sound is now OFF");
    }
    if (!strcmp(name, "launch_app")) {
        json_value_t * root = json_parse(args);
        json_value_t * e = root ? json_obj_get(root, "app") : NULL;
        char app[256];
        int ok = (e && e->type == JSON_STR && e->u.str && e->u.str[0]);
        if (ok) snprintf(app, sizeof(app), "%s", e->u.str);
        if (root) json_free(root);
        if (!ok) return strdup("error: missing app name");
        if (tools_launch(app) != 0) return strdup("error: failed to launch the application");
        snprintf(res, sizeof(res), "application '%s' launched", app);
        return strdup(res);
    }
    return strdup("error: unknown tool");
}

// run the tool-calling loop: while the model emits a tool call, execute it,
// feed the result back and let the model continue, up to 8 iterations. Stops
// early if the model repeats a tool call it already issued (avoids looping).
static char * run_tools(llm_t * l, char * reply, int n_predict) {
    double t0 = tnow_ms();
    // single pass: parse one tool call, execute it, then let the model answer
    llm_tool_call_t tc;
    if (llm_parse_tool_call(reply, &tc) != 0) return reply;
    char * result = exec_tool(tc.name, tc.arguments);
    double t1 = tnow_ms();
    if (CHAT_TIMING) fprintf(stderr, "[t] run_tools: parse+exec=%.0f ms\n", t1 - t0);
    llm_chat_add(l, "tool", result);
    free(result);
    llm_tool_call_free(&tc);
    char * nxt = llm_chat_generate(l, n_predict);
    if (CHAT_TIMING) fprintf(stderr, "[t] run_tools: final generate=%.0f ms (total %.0f ms)\n", tnow_ms() - t1, tnow_ms() - t0);
    free(reply);
    return nxt;
}

static void usage(const char * prog) {
    fprintf(stderr,
        "usage: %s -m <model.gguf> [options] [-p \"prompt\"]\n"
        "  -m <file>             model path (required)\n"
        "  -c <n>                context size\n"
        "  -p <text>             first user prompt (UTF-8)\n"
        "  --threads <n>         cpu threads\n"
        "  --temp <f>            sampling temperature\n"
        "  --top-k <n>           top-k sampling\n"
        "  --top-p <f>           nucleus (top-p) sampling\n"
        "  --predict <n>         max tokens per reply\n"
        "  --lang <code>         response language hint (e.g. ru)\n"
        "  --agent <file>        initial agent prompt file (default: agent.md)\n"
        "  --tools              enable system tools (volume_up/down/get/set, launch_app)\n"
#ifdef GGML_USE_VULKAN
        "  --device <cpu|gpu>    compute device (default: cpu)\n"
        "  --gpu <n>             Vulkan GPU number (0, 1, ... -> Vulkan<n>)\n"
#endif
        "  -h, --help            this help\n",
        prog);
}
#ifdef _WIN32
char* get_utf8_line(char* buffer, int buffer_size) {
    HANDLE hInput = GetStdHandle(STD_INPUT_HANDLE);
    if (hInput == INVALID_HANDLE_VALUE) return NULL;

    // Выделяем временный буфер для UTF-16 (wchat_t)
    int wbuffer_size = buffer_size;
    wchar_t* wbuffer = (wchar_t*)malloc(wbuffer_size * sizeof(wchar_t));
    if (!wbuffer) return NULL;

    DWORD chars_read = 0;
    // Читаем строку из консоли напрямую в UTF-16
    if (!ReadConsoleW(hInput, wbuffer, wbuffer_size - 1, &chars_read, NULL)) {
        // stdin is redirected (pipe/file): fall back to plain byte reads
        free(wbuffer);
        if (!fgets(buffer, buffer_size, stdin)) return NULL;
        return buffer;
    }

    wbuffer[chars_read] = L'\0';

    // Конвертируем UTF-16 (WideChar) в UTF-8 (MultiByte)
    int utf8_len = WideCharToMultiByte(CP_UTF8, 0, wbuffer, chars_read, buffer, buffer_size - 1, NULL, NULL);
    free(wbuffer);

    if (utf8_len == 0 && chars_read > 0) {
        return NULL; // Ошибка конвертации
    }

    buffer[utf8_len] = '\0';
    return buffer;
}
#endif

// discard the remainder of an overlong input line so the next read starts
// with a fresh line instead of the tail of the previous one
static void stdin_drain_line(void) {
#ifdef _WIN32
    HANDLE h = GetStdHandle(STD_INPUT_HANDLE);
    if (h != INVALID_HANDLE_VALUE && GetFileType(h) == FILE_TYPE_CHAR) {
        wchar_t tmp[256];
        DWORD r = 0;
        do {
            if (!ReadConsoleW(h, tmp, 255, &r, NULL) || r == 0) break;
        } while (tmp[r - 1] != L'\n');
        return;
    }
#endif
    int c;
    do { c = fgetc(stdin); } while (c != '\n' && c != EOF);
}

int main(int argc, char ** argv) {
    double t_main0 = tnow_ms(); // total wall time: main entry -> main exit
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
    {
        int wn = 0;
        wchar_t ** wv = wsplit(GetCommandLineW(), &wn);
        if (wv && wn > 0) {
            for (int i = 0; i < wn; i++) { argv[i] = w2utf8(wv[i]); free(wv[i]); }
            free(wv);
            argc = wn;
        }
    }
#endif

    const char * model  = NULL;
    const char * prompt = NULL;
    const char * agent_path = NULL;
    int n_predict = DEFAULT_N_PREDICT;
    int tools_enabled = 0;
    llm_params_t params;
    llm_params_default(&params);

    for (int i = 1; i < argc; i++) {
        const char * a = argv[i];
        if      (!strcmp(a, "-m") && i + 1 < argc)               model = argv[++i];
        else if (!strcmp(a, "-c") && i + 1 < argc) {
            params.n_ctx = atoi(argv[++i]);
            if (params.n_ctx <= 0) {
                fprintf(stderr, "chat: -c must be a positive integer, got '%d'\n", params.n_ctx);
                usage(argv[0]); return 1;
            }
        }
        else if (!strcmp(a, "-p") && i + 1 < argc)               prompt = argv[++i];
        else if (!strcmp(a, "--threads") && i + 1 < argc) {
            params.n_threads = atoi(argv[++i]);
            if (params.n_threads <= 0) {
                fprintf(stderr, "chat: --threads must be a positive integer, got '%d'\n", params.n_threads);
                usage(argv[0]); return 1;
            }
        }
        else if (!strcmp(a, "--temp") && i + 1 < argc) {
            params.sampler.temp = (float) atof(argv[++i]);
            if (params.sampler.temp < 0.0f) {
                fprintf(stderr, "chat: --temp must be >= 0, got '%g'\n", params.sampler.temp);
                usage(argv[0]); return 1;
            }
        }
        else if (!strcmp(a, "--top-k") && i + 1 < argc) {
            params.sampler.top_k = atoi(argv[++i]);
            if (params.sampler.top_k < 1) {
                fprintf(stderr, "chat: --top-k must be a positive integer (>= 1), got '%d'\n", params.sampler.top_k);
                usage(argv[0]); return 1;
            }
        }
        else if (!strcmp(a, "--top-p") && i + 1 < argc) {
            params.sampler.top_p = (float) atof(argv[++i]);
            if (params.sampler.top_p <= 0.0f || params.sampler.top_p > 1.0f) {
                fprintf(stderr, "chat: --top-p must be in (0, 1] (fraction, e.g. 0.95, not 95), got '%g'\n", params.sampler.top_p);
                usage(argv[0]); return 1;
            }
        }
        else if (!strcmp(a, "--predict") && i + 1 < argc) {
            n_predict = atoi(argv[++i]);
            if (n_predict <= 0) {
                fprintf(stderr, "chat: --predict must be a positive integer, got '%d'\n", n_predict);
                usage(argv[0]); return 1;
            }
        }
        else if (!strcmp(a, "--lang"))                          { if (i+1 < argc) params.lang = argv[++i]; }
        else if (!strcmp(a, "--agent") && i + 1 < argc)          agent_path = argv[++i];
        else if (!strcmp(a, "--tools"))                          tools_enabled = 1;
#ifdef GGML_USE_VULKAN
        else if (!strcmp(a, "--device") && i + 1 < argc) {
            const char * d = argv[++i];
            if      (!strcmp(d, "gpu")) params.use_gpu = 1;
            else if (!strcmp(d, "cpu")) params.use_gpu = 0;
            else { fprintf(stderr, "chat: --device expects 'cpu' or 'gpu'\n"); usage(argv[0]); return 1; }
        }
        else if (!strcmp(a, "--gpu") && i + 1 < argc) { params.gpu_index = atoi(argv[++i]); params.use_gpu = 1; }
#endif
        else if (!strcmp(a, "-h") || !strcmp(a, "--help"))       { usage(argv[0]); return 0; }
        else { fprintf(stderr, "chat: unknown option '%s'\n", a); usage(argv[0]); return 1; }
    }

    if (!model) { fprintf(stderr, "chat: -m <model.gguf> is required\n"); usage(argv[0]); return 1; }

    llm_t * l = llm_init(model, &params);
    if (!l) { fprintf(stderr, "chat: llm_init('%s') failed\n", model); return 1; }
    if (tools_enabled) llm_set_tools(l, system_tools, 7);

    // initial agent prompt: read the --agent file (tools.md when --tools is set,
    // otherwise agent.md) and send it to the model as a system message.
    {
        double wa0 = tnow_ms();
        const char * ap = agent_path ? agent_path : (tools_enabled ? "tools.md" : DEFAULT_AGENT);
        if (llm_chat_load_agent(l, ap, params.lang) != 0) {
            fprintf(stderr, "chat: warning: cannot read agent prompt '%s', continuing without it\n", ap);
        }

        // pre-process the constant system(+tools) prefix into the KV cache so the
        // first user turn is a short continuation (gemma3/llama)
        llm_chat_warmup(l);
        if (CHAT_TIMING) fprintf(stderr, "[t] agent+warmup: %.0f ms\n", tnow_ms() - wa0);
    }
    if (prompt) {
        out_printf("you> %s\r\n\r\nai> ", prompt);
        char * reply = run_tools(l, chat_turn(l, prompt, n_predict), n_predict);
        if (reply) out_str(reply);
        out_str("\r\n\r\n");
        free(reply);
    } else {

    // interactive loop: read lines from stdin until EOF or "exit"

    char line[4096];
    while (1) {
        out_str("you> ");

#ifdef _WIN32
        if (!get_utf8_line(line, sizeof(line))) {
            out_str("\r\nexit\r\n");
            break;
        }
#else
        if (!fgets(line,sizeof(line), stdin)) {
            out_str("\r\nexit\r\n");
            break;
        }
#endif
        if (!strchr(line, '\n')) stdin_drain_line(); // overlong line: drop the rest
        size_t len = strlen(line);
        while (len && (line[len-1] == '\n' || line[len-1] == '\r' ||
                       line[len-1] == ' ' || line[len-1] == '\t')) {
            line[--len] = '\0';
        }

        if (len == 0) {
            continue;
        }

        if (!strcmp(line, "exit") || !strcmp(line, "quit")) {
            break;
        }

        out_str("ai> ");
        double tt0 = tnow_ms();
        char * reply = run_tools(l, chat_turn(l, line, n_predict), n_predict);
        double tt1 = tnow_ms();
        if (CHAT_TIMING) fprintf(stderr, "[t] turn total=%.0f ms\n", tt1 - tt0);
        if (reply) out_str(reply);
        out_str("\r\n\r\n");
        free(reply);
    }
    }

    llm_print_stats(l);
    fprintf(stderr, "[llm] total wall time (main): %.2f s\n", (tnow_ms() - t_main0) / 1000.0);
    llm_free(l);
    return 0;
}
