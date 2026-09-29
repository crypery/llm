/*
 * Git: https://github.com/crypery
 * Author: https://crypery.com
 * License: GNU AGPL v3 (Affero GPL)
 */

// common.h — shared low-level utilities for the Silero stress C port.
// No model logic here: UTF-8 codepoint strings, string builder, hash map,
// codepoint classes, softmax/argmax, whole-file reader and a minimal JSON parser.
#ifndef COMMON_H
#define COMMON_H

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// ---------------------------------------------------------------------------
// UTF-8 codepoint strings
// ---------------------------------------------------------------------------

typedef struct { uint32_t * v; int n, cap; } ucs_t;

void ucs_init(ucs_t * u);
void ucs_push(ucs_t * u, uint32_t cp);
void ucs_free(ucs_t * u);
ucs_t ucs_dup(const ucs_t * u);
void ucs_from_utf8(const char * s, size_t len, ucs_t * u);
char * ucs_to_str(const ucs_t * u);
int ucs_eq(const ucs_t * a, const ucs_t * b);

// ---------------------------------------------------------------------------
// Unicode character classes (subset sufficient for ru text)
// ---------------------------------------------------------------------------

uint32_t cp_lower(uint32_t cp);
uint32_t cp_upper(uint32_t cp);
int cp_islower(uint32_t cp);
int cp_is_cyr(uint32_t cp);
int cp_is_vowel(uint32_t cp);
int cp_is_control(uint32_t cp);
int cp_is_space(uint32_t cp);
int cp_is_ws_re(uint32_t cp);
int cp_is_punct(uint32_t cp);
int cp_is_chinese(uint32_t cp);
int cp_is_accentor_delim(uint32_t cp);
int cp_is_homochar(uint32_t cp);

// ---------------------------------------------------------------------------
// list of codepoint strings
// ---------------------------------------------------------------------------

typedef struct { ucs_t * v; int n, cap; } ucs_list_t;

void ul_init(ucs_list_t * l);
void ul_push(ucs_list_t * l, ucs_t t);
void ul_free(ucs_list_t * l);

// ---------------------------------------------------------------------------
// string builder (UTF-8)
// ---------------------------------------------------------------------------

typedef struct { char * buf; int n, cap; } sb_t;

void sb_init(sb_t * s);
void sb_push_cp(sb_t * s, uint32_t cp);
void sb_append_char(sb_t * s, char c);
void sb_append_n(sb_t * s, const char * str, int n);
void sb_append(sb_t * s, const char * str);
void sb_append_ucs(sb_t * s, const ucs_t * u);
void sb_append_range(sb_t * s, const ucs_t * u, int a, int b);
char * sb_str(sb_t * s);
void sb_free(sb_t * s);

// ---------------------------------------------------------------------------
// string -> value hash map (open addressing, FNV-1a)
// ---------------------------------------------------------------------------

typedef struct {
    char ** keys;
    uint64_t * hashes;
    void ** vals;
    int64_t n, cap;
} strmap_t;

void smap_init(strmap_t * m, int64_t expect);
void smap_put(strmap_t * m, const char * key, size_t len, void * val);
void * smap_get(const strmap_t * m, const char * key, size_t len);
void * smap_get_c(const strmap_t * m, const char * key);
void smap_free(strmap_t * m); // frees keys + tables, NOT the values

// ---------------------------------------------------------------------------
// small helpers
// ---------------------------------------------------------------------------

void softmax(const float * x, int n, float * out);
int argmax(const float * x, int n);
char * read_file(const char * path, size_t * out_len);

// duplicate at most n bytes of s into a NUL-terminated heap string
char * strndup_(const char * s, size_t n);

// monotonic wall-clock milliseconds (high-resolution where available)
double now_ms(void);

// cross-platform read-only memory map of a whole file
void * mmap_file(const char * path, size_t * out_size);
void   munmap_file(void * ptr, size_t size);

// smallest power of two >= x
int next_pow2(int x);

// ---------------------------------------------------------------------------
// minimal JSON parser
// ---------------------------------------------------------------------------

typedef enum { JSON_NULL, JSON_BOOL, JSON_NUM, JSON_STR, JSON_ARR, JSON_OBJ } json_type_t;

typedef struct json_value json_value_t;
struct json_value {
    json_type_t type;
    union {
        int b;
        double num;
        char * str;
        struct { json_value_t ** v; int n, cap; } arr;
        struct { char ** keys; json_value_t ** vals; int n, cap; } obj;
    } u;
};

json_value_t * json_parse(const char * text);
void json_free(json_value_t * v);
json_value_t * json_obj_get(json_value_t * obj, const char * key);
// serialize a parsed JSON value back into a JSON string (malloc'd, caller frees)
char * json_to_str(const json_value_t * v);

void clean_ai_text(char *str);
#endif // COMMON_H