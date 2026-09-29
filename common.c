/*
 * Git: https://github.com/crypery
 * Author: https://crypery.com
 * License: GNU AGPL v3 (Affero GPL)
 */

// common.c — implementation of the shared utilities declared in common.h.
#include <math.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#endif
#include "common.h"

// ---------------------------------------------------------------------------
// UTF-8 codepoint strings
// ---------------------------------------------------------------------------

void ucs_init(ucs_t * u) { u->v = NULL; u->n = 0; u->cap = 0; }

static void ucs_reserve(ucs_t * u, int n) {
    if (u->cap < n) {
        int c = u->cap ? u->cap : 16;
        while (c < n) c *= 2;
        u->v = (uint32_t *) realloc(u->v, (size_t) c * sizeof(uint32_t));
        u->cap = c;
    }
}

void ucs_push(ucs_t * u, uint32_t cp) { ucs_reserve(u, u->n + 1); u->v[u->n++] = cp; }

void ucs_free(ucs_t * u) { free(u->v); u->v = NULL; u->n = u->cap = 0; }

static int utf8_seq_len(uint8_t b0) {
    if (b0 < 0x80) return 1;
    if ((b0 >> 5) == 0x6) return 2;
    if ((b0 >> 4) == 0xE) return 3;
    if ((b0 >> 3) == 0x1E) return 4;
    return 1;
}

void ucs_from_utf8(const char * s, size_t len, ucs_t * u) {
    size_t i = 0;
    while (i < len) {
        uint8_t b0 = (uint8_t) s[i];
        int l = utf8_seq_len(b0);
        if (i + (size_t) l > len) l = 1;
        uint32_t cp = 0;
        if (l == 1) cp = b0;
        else if (l == 2) cp = ((uint32_t)(b0 & 0x1F) << 6) | ((uint32_t)((uint8_t) s[i + 1] & 0x3F));
        else if (l == 3) cp = ((uint32_t)(b0 & 0x0F) << 12) | ((uint32_t)((uint8_t) s[i + 1] & 0x3F) << 6) | ((uint32_t)((uint8_t) s[i + 2] & 0x3F));
        else cp = ((uint32_t)(b0 & 0x07) << 18) | ((uint32_t)((uint8_t) s[i + 1] & 0x3F) << 12) | ((uint32_t)((uint8_t) s[i + 2] & 0x3F) << 6) | ((uint32_t)((uint8_t) s[i + 3] & 0x3F));
        ucs_push(u, cp);
        i += (size_t) l;
    }
}

static int utf8_encode(uint32_t cp, char * out) {
    if (cp < 0x80) { out[0] = (char) cp; return 1; }
    if (cp < 0x800) {
        out[0] = (char) (0xC0 | (cp >> 6));
        out[1] = (char) (0x80 | (cp & 0x3F));
        return 2;
    }
    if (cp < 0x10000) {
        out[0] = (char) (0xE0 | (cp >> 12));
        out[1] = (char) (0x80 | ((cp >> 6) & 0x3F));
        out[2] = (char) (0x80 | (cp & 0x3F));
        return 3;
    }
    out[0] = (char) (0xF0 | (cp >> 18));
    out[1] = (char) (0x80 | ((cp >> 12) & 0x3F));
    out[2] = (char) (0x80 | ((cp >> 6) & 0x3F));
    out[3] = (char) (0x80 | (cp & 0x3F));
    return 4;
}

static size_t ucs_to_utf8(const ucs_t * u, char * buf, size_t cap) {
    size_t i = 0;
    for (int k = 0; k < u->n && i + 4 < cap; k++) i += (size_t) utf8_encode(u->v[k], buf + i);
    buf[i] = 0;
    return i;
}

char * ucs_to_str(const ucs_t * u) {
    char * buf = (char *) malloc((size_t) u->n * 4 + 1);
    ucs_to_utf8(u, buf, (size_t) u->n * 4 + 1);
    return buf;
}

int ucs_eq(const ucs_t * a, const ucs_t * b) {
    if (a->n != b->n) return 0;
    return memcmp(a->v, b->v, (size_t) a->n * sizeof(uint32_t)) == 0;
}

ucs_t ucs_dup(const ucs_t * u) {
    ucs_t d; ucs_init(&d);
    for (int i = 0; i < u->n; i++) ucs_push(&d, u->v[i]);
    return d;
}

// ---------------------------------------------------------------------------
// Unicode character classes (subset sufficient for ru text)
// ---------------------------------------------------------------------------

uint32_t cp_lower(uint32_t cp) {
    if (cp >= 'A' && cp <= 'Z') return cp + 0x20;
    if (cp >= 0x410 && cp <= 0x42F) return cp + 0x020;
    if (cp == 0x401) return 0x451;
    return cp;
}

uint32_t cp_upper(uint32_t cp) {
    if (cp >= 'a' && cp <= 'z') return cp - 0x20;
    if (cp >= 0x430 && cp <= 0x44F) return cp - 0x020;
    if (cp == 0x451) return 0x401;
    return cp;
}

int cp_islower(uint32_t cp) {
    return (cp >= 'a' && cp <= 'z') || (cp >= 0x430 && cp <= 0x44F) || cp == 0x451;
}

int cp_is_cyr(uint32_t cp) { // [А-Яа-яёЁ]
    return (cp >= 0x410 && cp <= 0x42F) || (cp >= 0x430 && cp <= 0x44F) || cp == 0x401 || cp == 0x451;
}

int cp_is_vowel(uint32_t cp) { // аоуыэиеяёю
    switch (cp) {
        case 0x430: case 0x43E: case 0x443: case 0x44B: case 0x44D:
        case 0x438: case 0x435: case 0x44F: case 0x451: case 0x44E: return 1;
        default: return 0;
    }
}

int cp_is_control(uint32_t cp) {
    if (cp == '\t' || cp == '\n' || cp == '\r') return 0;
    if (cp < 0x20) return 1;
    if (cp >= 0x7F && cp <= 0x9F) return 1;
    if (cp >= 0x200B && cp <= 0x200F) return 1;
    if (cp == 0x2028 || cp == 0x2029 || cp == 0x2060 || cp == 0xFEFF || cp == 0x180B) return 1;
    if (cp == 0x061C || cp == 0x06DD) return 1;
    if (cp >= 0xFFF0 && cp <= 0xFFFF) return 1;
    return 0;
}

int cp_is_space(uint32_t cp) { // ' ' \t \n \r + Zs
    if (cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r') return 1;
    if (cp == 0xA0 || cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200A) || cp == 0x202F || cp == 0x205F || cp == 0x3000) return 1;
    return 0;
}

int cp_is_ws_re(uint32_t cp) { // python re \s
    if (cp == '\f' || cp == '\v') return 1;
    return cp_is_space(cp);
}

int cp_is_punct(uint32_t cp) {
    if ((cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64) || (cp >= 91 && cp <= 96) || (cp >= 123 && cp <= 126)) return 1;
    if (cp >= 0x2010 && cp <= 0x2027) return 1;
    if (cp == 0x2030 || cp == 0x2031 || (cp >= 0x2032 && cp <= 0x2037) || (cp >= 0x2038 && cp <= 0x2045) ||
        (cp >= 0x2047 && cp <= 0x2051) || cp == 0x2053 || cp == 0x2055) return 1;
    if (cp == 0xA1 || cp == 0xAB || cp == 0xBB || cp == 0xB7 || cp == 0x37E || cp == 0x387) return 1;
    return 0;
}

int cp_is_chinese(uint32_t cp) {
    return (cp >= 0x4E00 && cp <= 0x9FFF) || (cp >= 0x3400 && cp <= 0x4DBF) ||
           (cp >= 0x20000 && cp <= 0x2A6DF) || (cp >= 0x2A700 && cp <= 0x2B73F) ||
           (cp >= 0x2B740 && cp <= 0x2B81F) || (cp >= 0x2B820 && cp <= 0x2CEAF) ||
           (cp >= 0xF900 && cp <= 0xFAFF) || (cp >= 0x2F800 && cp <= 0x2FA1F);
}

int cp_is_accentor_delim(uint32_t cp) { // [\s.,!?;:<>=()/\\]
    if (cp_is_ws_re(cp)) return 1;
    switch (cp) {
        case '.': case ',': case '!': case '?': case ';': case ':':
        case '<': case '>': case '=': case '(': case ')': case '/': case '\\': return 1;
        default: return 0;
    }
}

int cp_is_homochar(uint32_t cp) { // [а-яё+] IGNORECASE
    return (cp >= 0x430 && cp <= 0x44F) || cp == 0x451 || (cp >= 0x410 && cp <= 0x42F) || cp == 0x401 || cp == '+';
}

// ---------------------------------------------------------------------------
// list of codepoint strings
// ---------------------------------------------------------------------------

void ul_init(ucs_list_t * l) { l->v = NULL; l->n = 0; l->cap = 0; }

void ul_push(ucs_list_t * l, ucs_t t) {
    if (l->n == l->cap) {
        l->cap = l->cap ? l->cap * 2 : 8;
        l->v = (ucs_t *) realloc(l->v, (size_t) l->cap * sizeof(ucs_t));
    }
    l->v[l->n++] = t;
}

void ul_free(ucs_list_t * l) {
    for (int i = 0; i < l->n; i++) ucs_free(&l->v[i]);
    free(l->v);
    l->v = NULL;
    l->n = l->cap = 0;
}

// ---------------------------------------------------------------------------
// string builder (UTF-8)
// ---------------------------------------------------------------------------

void sb_init(sb_t * s) { s->buf = NULL; s->n = 0; s->cap = 0; }

static void sb_reserve(sb_t * s, int n) {
    if (s->cap < n) {
        int c = s->cap ? s->cap : 64;
        while (c < n) c *= 2;
        s->buf = (char *) realloc(s->buf, (size_t) c + 1);
        s->cap = c;
    }
}

void sb_push_cp(sb_t * s, uint32_t cp) {
    sb_reserve(s, s->n + 4);
    s->n += utf8_encode(cp, s->buf + s->n);
}

void sb_append_char(sb_t * s, char c) {
    sb_reserve(s, s->n + 1);
    s->buf[s->n++] = c;
}

void sb_append_n(sb_t * s, const char * str, int n) {
    sb_reserve(s, s->n + n + 1);
    memcpy(s->buf + s->n, str, (size_t) n);
    s->n += n;
}

void sb_append(sb_t * s, const char * str) {
    int l = (int) strlen(str);
    sb_reserve(s, s->n + l + 1);
    memcpy(s->buf + s->n, str, (size_t) l);
    s->n += l;
}

void sb_append_ucs(sb_t * s, const ucs_t * u) {
    for (int i = 0; i < u->n; i++) sb_push_cp(s, u->v[i]);
}

void sb_append_range(sb_t * s, const ucs_t * u, int a, int b) {
    for (int i = a; i < b; i++) sb_push_cp(s, u->v[i]);
}

char * sb_str(sb_t * s) {
    sb_reserve(s, s->n + 1);
    s->buf[s->n] = 0;
    return s->buf;
}

void sb_free(sb_t * s) { free(s->buf); s->buf = NULL; s->n = s->cap = 0; }

// ---------------------------------------------------------------------------
// string -> value hash map (open addressing, FNV-1a)
// ---------------------------------------------------------------------------

static uint64_t fnv1a_len(const char * s, size_t l) {
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < l; i++) { h ^= (uint8_t) s[i]; h *= 1099511628211ULL; }
    return h;
}

void smap_init(strmap_t * m, int64_t expect) {
    int64_t c = 1;
    while (c < expect * 2) c <<= 1;
    if (c < 16) c = 16;
    m->cap = c;
    m->n = 0;
    m->keys = (char **) calloc((size_t) c, sizeof(char *));
    m->hashes = (uint64_t *) calloc((size_t) c, sizeof(uint64_t));
    m->vals = (void **) calloc((size_t) c, sizeof(void *));
}

void smap_put(strmap_t * m, const char * key, size_t len, void * val) {
    if ((m->n + 1) * 10 >= m->cap * 7) {
        strmap_t nm;
        smap_init(&nm, m->n * 2);
        for (int64_t i = 0; i < m->cap; i++) {
            if (m->keys[i]) smap_put(&nm, m->keys[i], strlen(m->keys[i]), m->vals[i]);
        }
        free(m->keys); free(m->hashes); free(m->vals);
        *m = nm;
    }
    uint64_t h = fnv1a_len(key, len);
    int64_t i = (int64_t)(h & (uint64_t)(m->cap - 1));
    while (m->keys[i]) {
        if (m->hashes[i] == h && strlen(m->keys[i]) == len && memcmp(m->keys[i], key, len) == 0) {
            m->vals[i] = val;
            return;
        }
        i = (i + 1) & (m->cap - 1);
    }
    char * k = (char *) malloc(len + 1);
    memcpy(k, key, len);
    k[len] = 0;
    m->keys[i] = k;
    m->hashes[i] = h;
    m->vals[i] = val;
    m->n++;
}

void * smap_get(const strmap_t * m, const char * key, size_t len) {
    uint64_t h = fnv1a_len(key, len);
    int64_t i = (int64_t)(h & (uint64_t)(m->cap - 1));
    while (m->keys[i]) {
        if (m->hashes[i] == h && strlen(m->keys[i]) == len && memcmp(m->keys[i], key, len) == 0) return m->vals[i];
        i = (i + 1) & (m->cap - 1);
    }
    return NULL;
}

void * smap_get_c(const strmap_t * m, const char * key) {
    return smap_get(m, key, strlen(key));
}

void smap_free(strmap_t * m) {
    for (int64_t i = 0; i < m->cap; i++) free(m->keys[i]);
    free(m->keys); free(m->hashes); free(m->vals);
    memset(m, 0, sizeof(*m));
}

// ---------------------------------------------------------------------------
// small helpers
// ---------------------------------------------------------------------------

void softmax(const float * x, int n, float * out) {
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { out[i] = expf(x[i] - mx); sum += out[i]; }
    for (int i = 0; i < n; i++) out[i] /= sum;
}

int argmax(const float * x, int n) {
    int best = 0;
    for (int i = 1; i < n; i++) if (x[i] > x[best]) best = i;
    return best;
}

char * read_file(const char * path, size_t * out_len) {
    FILE * f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char * buf = (char *) malloc((size_t) sz + 1);
    if (!buf) { fclose(f); return NULL; }
    size_t rd = fread(buf, 1, (size_t) sz, f);
    fclose(f);
    buf[rd] = 0;
    if (out_len) *out_len = rd;
    return buf;
}

char * strndup_(const char * s, size_t n) {
    char * d = (char *) malloc(n + 1);
    if (!d) return NULL;
    memcpy(d, s, n);
    d[n] = 0;
    return d;
}

double now_ms(void) {
#ifdef _WIN32
    LARGE_INTEGER q, f;
    QueryPerformanceCounter(&q);
    QueryPerformanceFrequency(&f);
    return (double) q.QuadPart * 1000.0 / (double) f.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec * 1000.0 + (double) ts.tv_nsec / 1e6;
#endif
}

void * mmap_file(const char * path, size_t * out_size) {
#ifdef _WIN32
    HANDLE hf = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hf == INVALID_HANDLE_VALUE) return NULL;
    LARGE_INTEGER sz;
    if (!GetFileSizeEx(hf, &sz)) { CloseHandle(hf); return NULL; }
    size_t size = (size_t) sz.QuadPart;
    HANDLE hmap = CreateFileMappingA(hf, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!hmap) { CloseHandle(hf); return NULL; }
    void * ptr = MapViewOfFile(hmap, FILE_MAP_READ, 0, 0, 0);
    CloseHandle(hf);
    if (!ptr) { CloseHandle(hmap); return NULL; }
    *out_size = size;
    return ptr;
#else
    int fd = open(path, O_RDONLY);
    if (fd < 0) return NULL;
    struct stat st;
    if (fstat(fd, &st) != 0) { close(fd); return NULL; }
    size_t size = (size_t) st.st_size;
    void * ptr = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (ptr == MAP_FAILED) return NULL;
    *out_size = size;
    return ptr;
#endif
}

void munmap_file(void * ptr, size_t size) {
#ifdef _WIN32
    (void) size; UnmapViewOfFile(ptr);
#else
    munmap(ptr, size);
#endif
}

int next_pow2(int x) {
    int p = 1;
    while (p < x) p <<= 1;
    return p;
}

// ---------------------------------------------------------------------------
// minimal JSON parser
// ---------------------------------------------------------------------------

typedef struct { const char * p, * end; } json_ps_t;

static void json_skip_ws(json_ps_t * ps) {
    while (ps->p < ps->end) {
        char c = *ps->p;
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') ps->p++;
        else break;
    }
}

static json_value_t * json_new(json_type_t t) {
    json_value_t * v = (json_value_t *) calloc(1, sizeof(json_value_t));
    v->type = t;
    return v;
}

static uint32_t json_hex4(const char * p) {
    uint32_t v = 0;
    for (int i = 0; i < 4; i++) {
        char c = p[i];
        v <<= 4;
        if (c >= '0' && c <= '9') v |= (uint32_t)(c - '0');
        else if (c >= 'a' && c <= 'f') v |= (uint32_t)(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') v |= (uint32_t)(c - 'A' + 10);
    }
    return v;
}

static json_value_t * json_parse_value(json_ps_t * ps);

static json_value_t * json_parse_string(json_ps_t * ps) {
    ps->p++; // opening quote
    sb_t sb; sb_init(&sb);
    for (;;) {
        if (ps->p >= ps->end) break;
        char c = *ps->p;
        if (c == '"') { ps->p++; break; }
        if (c == '\\') {
            ps->p++;
            if (ps->p >= ps->end) break;
            char e = *ps->p++;
            switch (e) {
                case '"':  sb_append_char(&sb, '"');  break;
                case '\\': sb_append_char(&sb, '\\'); break;
                case '/':  sb_append_char(&sb, '/');  break;
                case 'b':  sb_append_char(&sb, '\b'); break;
                case 'f':  sb_append_char(&sb, '\f'); break;
                case 'n':  sb_append_char(&sb, '\n'); break;
                case 'r':  sb_append_char(&sb, '\r'); break;
                case 't':  sb_append_char(&sb, '\t'); break;
                case 'u': {
                    if (ps->p + 4 <= ps->end) {
                        uint32_t cp = json_hex4(ps->p);
                        ps->p += 4;
                        if (cp >= 0xD800 && cp <= 0xDBFF && ps->p + 5 < ps->end && ps->p[0] == '\\' && ps->p[1] == 'u') {
                            ps->p += 2;
                            uint32_t lo = json_hex4(ps->p);
                            ps->p += 4;
                            if (lo >= 0xDC00 && lo <= 0xDFFF) cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                        }
                        char tmp[4];
                        int l = 0;
                        // local encoder (avoid dependency on common internal name)
                        if (cp < 0x80) { tmp[0] = (char) cp; l = 1; }
                        else if (cp < 0x800) { tmp[0] = (char)(0xC0 | (cp >> 6)); tmp[1] = (char)(0x80 | (cp & 0x3F)); l = 2; }
                        else if (cp < 0x10000) { tmp[0] = (char)(0xE0 | (cp >> 12)); tmp[1] = (char)(0x80 | ((cp >> 6) & 0x3F)); tmp[2] = (char)(0x80 | (cp & 0x3F)); l = 3; }
                        else { tmp[0] = (char)(0xF0 | (cp >> 18)); tmp[1] = (char)(0x80 | ((cp >> 12) & 0x3F)); tmp[2] = (char)(0x80 | ((cp >> 6) & 0x3F)); tmp[3] = (char)(0x80 | (cp & 0x3F)); l = 4; }
                        sb_append_n(&sb, tmp, l);
                    }
                    break;
                }
                default: sb_append_char(&sb, '\\'); sb_append_char(&sb, e); break;
            }
        } else {
            sb_append_char(&sb, c);
            ps->p++;
        }
    }
    json_value_t * v = json_new(JSON_STR);
    v->u.str = sb_str(&sb);
    return v;
}

static json_value_t * json_parse_number(json_ps_t * ps) {
    double num = 0;
    int neg = 0;
    if (*ps->p == '-') { neg = 1; ps->p++; }
    while (ps->p < ps->end && *ps->p >= '0' && *ps->p <= '9') { num = num * 10 + (*ps->p - '0'); ps->p++; }
    if (ps->p < ps->end && *ps->p == '.') {
        ps->p++;
        double f = 1;
        while (ps->p < ps->end && *ps->p >= '0' && *ps->p <= '9') { f *= 0.1; num += (*ps->p - '0') * f; ps->p++; }
    }
    if (ps->p < ps->end && (*ps->p == 'e' || *ps->p == 'E')) {
        ps->p++;
        if (ps->p < ps->end && (*ps->p == '+' || *ps->p == '-')) ps->p++;
        while (ps->p < ps->end && *ps->p >= '0' && *ps->p <= '9') ps->p++;
    }
    json_value_t * v = json_new(JSON_NUM);
    v->u.num = neg ? -num : num;
    return v;
}

static json_value_t * json_parse_array(json_ps_t * ps) {
    json_value_t * v = json_new(JSON_ARR);
    ps->p++; // [
    json_skip_ws(ps);
    if (ps->p < ps->end && *ps->p == ']') { ps->p++; return v; }
    for (;;) {
        json_skip_ws(ps);
        json_value_t * item = json_parse_value(ps);
        if (item) {
            if (v->u.arr.n == v->u.arr.cap) {
                v->u.arr.cap = v->u.arr.cap ? v->u.arr.cap * 2 : 8;
                v->u.arr.v = (json_value_t **) realloc(v->u.arr.v, (size_t) v->u.arr.cap * sizeof(json_value_t *));
            }
            v->u.arr.v[v->u.arr.n++] = item;
        }
        json_skip_ws(ps);
        if (ps->p >= ps->end) break;
        if (*ps->p == ',') { ps->p++; continue; }
        if (*ps->p == ']') { ps->p++; break; }
        break;
    }
    return v;
}

static json_value_t * json_parse_object(json_ps_t * ps) {
    json_value_t * v = json_new(JSON_OBJ);
    ps->p++; // {
    json_skip_ws(ps);
    if (ps->p < ps->end && *ps->p == '}') { ps->p++; return v; }
    for (;;) {
        json_skip_ws(ps);
        json_value_t * key = NULL;
        if (ps->p < ps->end && *ps->p == '"') key = json_parse_string(ps);
        json_skip_ws(ps);
        if (ps->p < ps->end && *ps->p == ':') ps->p++;
        json_value_t * val = json_parse_value(ps);
        if (key && val) {
            if (v->u.obj.n == v->u.obj.cap) {
                v->u.obj.cap = v->u.obj.cap ? v->u.obj.cap * 2 : 8;
                v->u.obj.keys = (char **) realloc(v->u.obj.keys, (size_t) v->u.obj.cap * sizeof(char *));
                v->u.obj.vals = (json_value_t **) realloc(v->u.obj.vals, (size_t) v->u.obj.cap * sizeof(json_value_t *));
            }
            v->u.obj.keys[v->u.obj.n] = key->u.str; // take ownership of the string
            v->u.obj.vals[v->u.obj.n] = val;
            v->u.obj.n++;
            free(key);
        } else {
            if (key) json_free(key);
            if (val) json_free(val);
        }
        json_skip_ws(ps);
        if (ps->p >= ps->end) break;
        if (*ps->p == ',') { ps->p++; continue; }
        if (*ps->p == '}') { ps->p++; break; }
        break;
    }
    return v;
}

static json_value_t * json_parse_value(json_ps_t * ps) {
    json_skip_ws(ps);
    if (ps->p >= ps->end) return NULL;
    char c = *ps->p;
    if (c == '{') return json_parse_object(ps);
    if (c == '[') return json_parse_array(ps);
    if (c == '"') return json_parse_string(ps);
    if (c == 't') { json_value_t * v = json_new(JSON_BOOL); v->u.b = 1; ps->p += 4; return v; }
    if (c == 'f') { json_value_t * v = json_new(JSON_BOOL); v->u.b = 0; ps->p += 5; return v; }
    if (c == 'n') { json_value_t * v = json_new(JSON_NULL); ps->p += 4; return v; }
    if (c == '-' || (c >= '0' && c <= '9')) return json_parse_number(ps);
    return NULL;
}

json_value_t * json_parse(const char * text) {
    json_ps_t ps;
    ps.p = text;
    ps.end = text + strlen(text);
    return json_parse_value(&ps);
}

void json_free(json_value_t * v) {
    if (!v) return;
    switch (v->type) {
        case JSON_STR: free(v->u.str); break;
        case JSON_ARR:
            for (int i = 0; i < v->u.arr.n; i++) json_free(v->u.arr.v[i]);
            free(v->u.arr.v);
            break;
        case JSON_OBJ:
            for (int i = 0; i < v->u.obj.n; i++) {
                free(v->u.obj.keys[i]);
                json_free(v->u.obj.vals[i]);
            }
            free(v->u.obj.keys);
            free(v->u.obj.vals);
            break;
        default: break;
    }
    free(v);
}

json_value_t * json_obj_get(json_value_t * obj, const char * key) {
    if (!obj || obj->type != JSON_OBJ) return NULL;
    for (int i = 0; i < obj->u.obj.n; i++) {
        if (strcmp(obj->u.obj.keys[i], key) == 0) return obj->u.obj.vals[i];
    }
    return NULL;
}

static void json_sb(sb_t * sb, const json_value_t * v) {
    switch (v->type) {
        case JSON_NULL: sb_append(sb, "null"); break;
        case JSON_BOOL: sb_append(sb, v->u.b ? "true" : "false"); break;
        case JSON_NUM: {
            char tmp[64];
            snprintf(tmp, sizeof(tmp), "%.17g", v->u.num);
            sb_append(sb, tmp);
            break;
        }
        case JSON_STR: {
            sb_append_char(sb, '"');
            for (const char * p = v->u.str; *p; p++) {
                if (*p == '"') sb_append(sb, "\\\"");
                else if (*p == '\\') sb_append(sb, "\\\\");
                else if (*p == '\n') sb_append(sb, "\\n");
                else if (*p == '\r') sb_append(sb, "\\r");
                else if (*p == '\t') sb_append(sb, "\\t");
                else sb_append_char(sb, *p);
            }
            sb_append_char(sb, '"');
            break;
        }
        case JSON_ARR: {
            sb_append_char(sb, '[');
            for (int i = 0; i < v->u.arr.n; i++) {
                if (i) sb_append_char(sb, ',');
                json_sb(sb, v->u.arr.v[i]);
            }
            sb_append_char(sb, ']');
            break;
        }
        case JSON_OBJ: {
            sb_append_char(sb, '{');
            for (int i = 0; i < v->u.obj.n; i++) {
                if (i) sb_append_char(sb, ',');
                sb_append_char(sb, '"');
                sb_append(sb, v->u.obj.keys[i]);
                sb_append(sb, "\":");
                json_sb(sb, v->u.obj.vals[i]);
            }
            sb_append_char(sb, '}');
            break;
        }
        default: sb_append(sb, "null"); break;
    }
}

char * json_to_str(const json_value_t * v) {
    if (!v) return strdup("null");
    sb_t sb;
    sb_init(&sb);
    json_sb(&sb, v);
    return sb_str(&sb);
}

// Проверка длины UTF-8 символа
int get_utf8_char_len(unsigned char c) {
    if ((c & 0x80) == 0) return 1;
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return 4;
    return 1;
}

// Проверяет, входит ли ASCII символ в белый список разрешенных знаков
bool is_allowed_punct(char c) {
    return (c == '\'' || c == ',' || c == '.' || c == '!' || 
            c == '?'  || c == '-' || c == '+' || c == ':' || c == ';');
}

void clean_ai_text(char *str) {
    if (!str) return;

    unsigned char *src = (unsigned char *)str;
    unsigned char *dst = (unsigned char *)str;
    
    // Флаг для отслеживания того, был ли последний записанный символ пробелом
    // Инициализируем true, чтобы автоматически обрезать пробелы в самом начале строки
    bool last_was_space = true; 

    while (*src) {
        int len = get_utf8_char_len(*src);
        unsigned char next_char = '\0';
        
        // --- 1. УДАЛЕНИЕ ЭМОДЗИ (4 байта) ---
        if (len == 4) {
            src += 4;
            continue;
        }

        // --- 2. ИСПРАВЛЕНИЕ ПСЕВДОГРАФИКИ (3 байта) ---
        if (len == 3 && src[0] == 0xE2 && src[1] == 0x94 && src[2] == 0x80) {
            next_char = '-';
            src += 3;
        }
        // --- 3. СТАНДАРТИЗАЦИЯ КАВЫЧЕК ---
        // Французские «ёлочки» (2 байта)
        else if (len == 2 && src[0] == 0xC2 && src[1] == 0xAB) {
            next_char = ' ';
            src += 2;
        }
        else if (len == 2 && src[0] == 0xC2 && src[1] == 0xBB) {
            next_char = ' ';
            src += 2;
        }
        // Английские «умные» кавычки (3 байта)
        else if (len == 3 && src[0] == 0xE2 && src[1] == 0x80 && (src[2] == 0x9C || src[2] == 0x9D)) {
            next_char = '"';
            src += 3;
        }
        // --- 4. ОБРАБОТКА ОСТАЛЬНЫХ МНОГОБАЙТОВЫХ СИМВОЛОВ ---
        else if (len > 1) {
            // Кириллица в UTF-8 начинается с байтов 0xD0 и 0xD1
            // Если это кириллица — оставляем как есть, иначе заменяем на пробел (неизвестный спецсимвол)
            if (src[0] == 0xD0 || src[0] == 0xD1) {
                // Копируем кириллический символ целиком
                for (int i = 0; i < len; i++) {
                    *dst++ = *src++;
                }
                last_was_space = false;
                continue;
            } else {
                next_char = ' ';
                src += len;
            }
        }
        // --- 5. ОБРАБОТКА ОДНОБАЙТОВЫХ СИМВОЛОВ (ASCII) ---
        else {
            char c = *src;
            // Если это буква, цифра или разрешенный знак — оставляем символ
            if (isalnum((unsigned char)c) || is_allowed_punct(c)) {
                next_char = c;
            } else {
                // Все остальные ASCII спецсимволы (включая табы, переносы строк, кавычки, бакслеши) превращаем в пробел
                next_char = ' ';
            }
            src++;
        }

        // --- 6. ЗАПИСЬ СИМВОЛА, СХЛОПЫВАНИЕ ПРОБЕЛОВ И НОРМАЛИЗАЦИЯ ПУНКТУАЦИИ ---
        if (next_char != '\0') {
            if (next_char == ' ') {
                // Если текущий символ пробел, и прошлый был пробелом — игнорируем
                if (!last_was_space) {
                    *dst++ = ' ';
                    last_was_space = true;
                }
            } else {
                *dst++ = next_char;
                last_was_space = false;
            }
        }
    }

    // Удаляем возможный пробел на конце строки
    if (dst > (unsigned char *)str && *(dst - 1) == ' ') {
        dst--;
    }
    
    *dst = '\0'; // Закрываем строку
}