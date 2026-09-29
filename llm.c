/*
 * Git: https://github.com/crypery
 * Author: https://crypery.com
 * License: GNU AGPL v3 (Affero GPL)
 */

// llm.c - LLM inference module (ggml-backed) for Qwen3-Next style hybrid models.
// Architecture "qwen35": hybrid of Gated DeltaNet (linear/SSM) blocks and full
// attention blocks.
//
// GGUF linear weight layout: [n_in, n_out] (ne[0]=n_in, ne[1]=n_out).
// Matmul: out = ggml_mul_mat(W, xT) where xT is [n_tokens, n_in] -> out [n_out, n_tokens].
// Hidden states stored row-major [n_tokens][n_in] (feature dim fast).
//
// Tokenizer is byte-level BPE (gpt2 style, "qwen35" pre-tokenizer): raw UTF-8
// input bytes are mapped through the GPT-2 byte-to-unicode table before BPE, so
// non-ASCII text (Cyrillic, etc.) tokenizes correctly.

#include "llm.h"
#include "log.h"
#include "compat.h"
#include <omp.h>
#include <ggml.h>
#include <ggml-cpu.h>
#include <ggml-quants.h>
#ifdef GGML_USE_VULKAN
#include <ggml-backend.h>
#endif
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <sys/stat.h>
#if defined(__AVX2__) || defined(__AVX__) || defined(__SSE2__)
#include <immintrin.h>
#endif

// convenience typedefs (ggml.h declares these as plain structs/enums)
typedef struct ggml_tensor ggml_tensor;
typedef struct ggml_context ggml_context;
typedef struct ggml_cgraph ggml_cgraph;

#define LLM_MAX_MSG 64

// ---------------------------------------------------------------------------
// GGUF parsing
// ---------------------------------------------------------------------------
typedef struct {
    char      name[160];
    int       n_dims;
    int64_t   ne[4];
    int64_t   nb[4];
    enum ggml_type type;
    size_t    offset;
    size_t    size;
    ggml_tensor * gt;
} gguf_tensor_t;

typedef struct {
    uint8_t * buf;
    size_t    buf_size;
    int       n_tensors;
    gguf_tensor_t * tensors;
} gguf_file_t;

typedef struct { const uint8_t * p; size_t len, pos; } rd_ctx_t;
static int rd_u32(rd_ctx_t * r, uint32_t * o) { if (r->pos+4 > r->len) return -1; uint32_t v; memcpy(&v, r->p+r->pos, 4); r->pos+=4; *o=v; return 0; }
static int rd_u64(rd_ctx_t * r, uint64_t * o) { if (r->pos+8 > r->len) return -1; uint64_t v; memcpy(&v, r->p+r->pos, 8); r->pos+=8; *o=v; return 0; }
static int rd_str(rd_ctx_t * r, char ** out, size_t * out_len) {
    uint64_t n; if (rd_u64(r, &n) || n > (1u<<30)) return -1;
    if (r->pos + n > r->len) return -1;
    char * s = (char *) malloc(n+1); memcpy(s, r->p + r->pos, n); s[n] = 0;
    r->pos += n; *out = s; *out_len = n; return 0;
}
static int gguf_skip_val(rd_ctx_t * r, uint32_t type, uint64_t n) {
    switch (type) {
        case 0: case 1: case 7: r->pos += n; return r->pos <= r->len ? 0 : -1;
        case 2: case 3: r->pos += n*2; return r->pos <= r->len ? 0 : -1;
        case 4: case 5: case 6: r->pos += n*4; return r->pos <= r->len ? 0 : -1;
        case 10: case 11: case 12: r->pos += n*8; return r->pos <= r->len ? 0 : -1;
        case 8: { for (uint64_t i = 0; i < n; i++) { char * s; size_t l; if (rd_str(r, &s, &l)) return -1; free(s); } return 0; }
        default: return -1;
    }
}
static int gguf_get_num(rd_ctx_t * r, uint32_t type, double * out) {
    switch (type) {
        case 0: case 1: case 7: { uint8_t v; memcpy(&v, r->p+r->pos, 1); r->pos+=1; *out=v; return 0; }
        case 2: case 3: { uint16_t v; memcpy(&v, r->p+r->pos, 2); r->pos+=2; *out=v; return 0; }
        case 4: { uint32_t v; memcpy(&v, r->p+r->pos, 4); r->pos+=4; *out=v; return 0; }
        case 5: { int32_t v; memcpy(&v, r->p+r->pos, 4); r->pos+=4; *out=v; return 0; }
        case 6: { uint32_t v; memcpy(&v, r->p+r->pos, 4); r->pos+=4; *out=*(float*)&v; return 0; }
        case 10: { uint64_t v; memcpy(&v, r->p+r->pos, 8); r->pos+=8; *out=v; return 0; }
        case 11: { int64_t v; memcpy(&v, r->p+r->pos, 8); r->pos+=8; *out=v; return 0; }
        case 12: { double v; memcpy(&v, r->p+r->pos, 8); r->pos+=8; *out=v; return 0; }
        default: return -1;
    }
}

// ---------------------------------------------------------------------------
// tokenizer (byte-level BPE, gpt2/qwen35)
// ---------------------------------------------------------------------------
typedef struct {
    int    n_vocab;
    char ** tokens;
    int   * token_type;
    float * scores;       // token scores (SentencePiece BPE merge order)
    int    * char2id;     // by byte-mapped code point (0..0x1FF)
    uint64_t * mkey; int32_t * mval; int mcap; int n_merges;
    char ** skey; int32_t * sval; int scap;   // string -> id
    char ** merges_raw; int n_merges_raw;
    int    bos, eos, pad, unk;
    int    add_bos;
    int    max_token_len;
    int    n_special;          // <name> / <|name|> tokens kept atomic
    int  ** special_cps;       // per special: byte-mapped code point sequence
    int   * special_len;       // per special: number of code points
    int   * special_id;        // per special: vocab id
    char * chat_template;
} tok_t;

// GPT-2 byte-to-unicode table: raw bytes -> code points.
// Direct bytes (33..126, 161..172, 174..255) map to themselves; every other
// byte maps to 256 + n in order of byte value (so space -> 0x120 = U+0120 "Ġ").
static uint16_t byte_to_uni[256];
static uint8_t  uni_to_byte[0x200];
static int      byte_uni_init = 0;

static void init_byte_uni(void) {
    if (byte_uni_init) return;
    byte_uni_init = 1;
    int n = 0;
    for (int b = 0; b < 256; b++) {
        int direct = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
        if (direct) { byte_to_uni[b] = (uint16_t) b; }
        else { byte_to_uni[b] = (uint16_t)(256 + n); n++; }
    }
    for (int b = 0; b < 256; b++) uni_to_byte[byte_to_uni[b]] = (uint8_t) b;
}

static int tok_hash_u(uint64_t k, int cap) { uint64_t h = k * 0x9E3779B97F4A7C15ULL; return (int)(h & (uint64_t)(cap-1)); }
static int tok_hash_s(const char * s, int cap) {
    uint64_t h = 1469598103934665603ULL;
    for (const unsigned char * p = (const unsigned char *) s; *p; p++) { h ^= *p; h *= 1099511628211ULL; }
    return (int)(h & (uint64_t)(cap-1));
}
static int tok_hash_s_len(const char * s, int len, int cap) {
    uint64_t h = 1469598103934665603ULL;
    for (int i = 0; i < len; i++) { h ^= (unsigned char) s[i]; h *= 1099511628211ULL; }
    return (int)(h & (uint64_t)(cap-1));
}
static int tok_merge_get(const tok_t * t, int a, int b) {
    uint64_t key = ((uint64_t)(uint32_t)a << 32) | (uint32_t)b;
    int i = tok_hash_u(key, t->mcap);
    while (t->mkey[i] != 0) { if (t->mkey[i] == key) return t->mval[i]; i = (i+1) & (t->mcap-1); }
    return -1;
}
static void tok_merge_put(tok_t * t, int a, int b, int pri) {
    uint64_t key = ((uint64_t)(uint32_t)a << 32) | (uint32_t)b;
    int i = tok_hash_u(key, t->mcap);
    while (t->mkey[i] != 0 && t->mkey[i] != key) i = (i+1) & (t->mcap-1);
    t->mkey[i] = key; t->mval[i] = pri;
}
static int tok_str_id(const tok_t * t, const char * s) {
    int i = tok_hash_s(s, t->scap);
    while (t->skey[i] != NULL) { if (strcmp(t->skey[i], s) == 0) return t->sval[i]; i = (i+1) & (t->scap-1); }
    return -1;
}
static int tok_str_id_len(const tok_t * t, const char * s, int len) {
    int i = tok_hash_s_len(s, len, t->scap);
    while (t->skey[i] != NULL) {
        if (strlen(t->skey[i]) == (size_t) len && memcmp(t->skey[i], s, (size_t) len) == 0) return t->sval[i];
        i = (i+1) & (t->scap-1);
    }
    return -1;
}
static void tok_str_put(tok_t * t, const char * s, int id) {
    int i = tok_hash_s(s, t->scap);
    while (t->skey[i] != NULL) i = (i+1) & (t->scap-1);
    t->skey[i] = strdup(s); t->sval[i] = id;
}

static int utf8_decode(const char * s, int len, uint32_t * cps, int max) {
    int n = 0, i = 0;
    while (i < len && n < max) {
        uint8_t c = s[i]; uint32_t cp; int extra;
        if (c < 0x80) { cp = c; extra = 0; }
        else if ((c>>5) == 6) { cp = c & 0x1F; extra = 1; }
        else if ((c>>4) == 14) { cp = c & 0x0F; extra = 2; }
        else if ((c>>3) == 30) { cp = c & 0x07; extra = 3; }
        else { cp = c; extra = 0; }
        if (i + extra >= len && extra > 0) { extra = 0; cp = c; }
        for (int j = 1; j <= extra; j++) cp = (cp << 6) | (s[i+j] & 0x3F);
        cps[n++] = cp; i += extra + 1;
    }
    return n;
}
static int utf8_encode(uint32_t cp, char * out) {
    if (cp < 0x80) { out[0]=cp; return 1; }
    if (cp < 0x800) { out[0]=0xC0|(cp>>6); out[1]=0x80|(cp&0x3F); return 2; }
    if (cp < 0x10000) { out[0]=0xE0|(cp>>12); out[1]=0x80|((cp>>6)&0x3F); out[2]=0x80|(cp&0x3F); return 3; }
    out[0]=0xF0|(cp>>18); out[1]=0x80|((cp>>12)&0x3F); out[2]=0x80|((cp>>6)&0x3F); out[3]=0x80|(cp&0x3F); return 4;
}

// is this byte-mapped code point a whitespace byte?
static int is_ws_cp(uint32_t cp) {
    int b;
    if (cp >= 0x100 && cp < 0x200) b = uni_to_byte[cp];
    else if (cp < 0x80) b = (int) cp;
    else return 0;
    return b == ' ' || b == '\t' || b == '\n' || b == '\r' || b == '\v' || b == '\f';
}
// word-ish char: letters, digits, apostrophe, or any non-whitespace byte-mapped char
static int is_word_cp(uint32_t cp) {
    if (is_ws_cp(cp)) return 0;
    if (cp >= 0x80) return 1;
    return (cp >= 'a' && cp <= 'z') || (cp >= 'A' && cp <= 'Z') || (cp >= '0' && cp <= '9') || cp == '\'';
}
// llama-bpe pre-tokenizer: a digit byte (0-9). Digits are chunked 1..3.
static int is_digit_cp(uint32_t cp) {
    return (cp >= '0' && cp <= '9');
}
// llama-bpe pre-tokenizer: a "letter" byte -- ASCII a-z/A-Z, apostrophe, or any
// non-ASCII byte (>=0x80). This mirrors \p{L} closely enough for the byte-level
// BPE pre-tokenization (Cyrillic/CJK/etc. all live in the >=0x80 byte range).
static int is_letter_cp(uint32_t cp) {
    if (is_ws_cp(cp)) return 0;
    if (cp >= 0x80) return 1;
    return (cp >= 'a' && cp <= 'z') || (cp >= 'A' && cp <= 'Z') || cp == '\'';
}
// llama-bpe pre-tokenizer: a newline byte (\r or \n). Newlines are never folded
// into the leading-punct of the following word.
static int is_newline_cp(uint32_t cp) {
    int b;
    if (cp >= 0x100 && cp < 0x200) b = uni_to_byte[cp];
    else if (cp < 0x80) b = (int) cp;
    else return 0;
    return b == '\n' || b == '\r';
}

// BPE a single "word" given as an array of byte-mapped code points.
static int bpe_encode_word(const tok_t * t, const uint32_t * cps, int ncp, int * out, int max) {
    if (ncp <= 0) return 0;
    int * seq = (int *) malloc(sizeof(int) * (ncp + 1));
    int n = 0;
    for (int i = 0; i < ncp; i++) {
        int id = (cps[i] < 0x200) ? t->char2id[cps[i]] : -1;
        if (id < 0) continue;
        seq[n++] = id;
    }
    if (n == 0) { free(seq); return 0; }
    for (;;) {
        int best = -1, best_idx = -1;
        for (int i = 0; i < n-1; i++) {
            int pri = tok_merge_get(t, seq[i], seq[i+1]);
            if (pri >= 0 && (best < 0 || pri < best)) { best = pri; best_idx = i; }
        }
        if (best < 0) break;
        const char * sa = t->tokens[seq[best_idx]];
        const char * sb = t->tokens[seq[best_idx+1]];
        int la = (int)strlen(sa), lb = (int)strlen(sb);
        char * concat = (char *) malloc(la+lb+1);
        memcpy(concat, sa, la); memcpy(concat+la, sb, lb); concat[la+lb] = 0;
        int mid = tok_str_id(t, concat);
        free(concat);
        if (mid < 0) break;
        for (int i = best_idx+1; i < n-1; i++) seq[i] = seq[i+1];
        seq[best_idx] = mid; n--;
    }
    int cnt = n > max ? max : n;
    for (int i = 0; i < cnt; i++) out[i] = seq[i];
    free(seq);
    return cnt;
}

// build hash tables from the KV-loaded vocab data (tokens/token_type/merges)
static void tok_build_hashes(tok_t * t) {
    init_byte_uni();
    t->char2id = (int *) malloc(sizeof(int) * 0x200);
    memset(t->char2id, 0xFF, sizeof(int) * 0x200);
    if (t->n_vocab > 0) {
        for (int i = 0; i < t->n_vocab; i++) {
            uint32_t cps[8];
            int ncp = utf8_decode(t->tokens[i], (int) strlen(t->tokens[i]), cps, 8);
            if (ncp == 1 && cps[0] < 0x200 && t->char2id[cps[0]] < 0) t->char2id[cps[0]] = i;
        }
    }
    t->scap = next_pow2(t->n_vocab * 2 + 16);
    t->skey = (char **) calloc(t->scap, sizeof(char *));
    t->sval = (int32_t *) malloc(sizeof(int32_t) * t->scap);
    if (t->n_vocab > 0) {
        t->max_token_len = 0;
        for (int i = 0; i < t->n_vocab; i++) {
            int sl = (int) strlen(t->tokens[i]);
            if (sl > t->max_token_len) t->max_token_len = sl;
            tok_str_put(t, t->tokens[i], i);
        }
        if (t->max_token_len < 1) t->max_token_len = 1;
    }
    // special tokens ("<|name|>" / "<name>" control tokens used by chat
    // templates): kept atomic during encoding so they emit as a single id
    t->n_special = 0;
    for (int i = 0; i < t->n_vocab; i++) {
        const char * s = t->tokens[i];
        int sl = (int) strlen(s);
        int ok = (sl >= 3 && s[0] == '<' && s[sl-1] == '>');
        if (ok) {
            for (int j = 1; j < sl-1; j++) {
                if (s[j] == ' ' || s[j] == '\n') { ok = 0; break; }
            }
        }
        if (ok) t->n_special++;
    }
    if (t->n_special > 0) {
        t->special_cps = (int **) malloc(sizeof(int *) * (size_t) t->n_special);
        t->special_len = (int *) malloc(sizeof(int) * (size_t) t->n_special);
        t->special_id = (int *) malloc(sizeof(int) * (size_t) t->n_special);
        int k = 0;
        for (int i = 0; i < t->n_vocab; i++) {
            const char * s = t->tokens[i];
            int sl = (int) strlen(s);
            int ok = (sl >= 3 && s[0] == '<' && s[sl-1] == '>');
            if (ok) {
                for (int j = 1; j < sl-1; j++) {
                    if (s[j] == ' ' || s[j] == '\n') { ok = 0; break; }
                }
            }
            if (!ok) continue;
            int * cps = (int *) malloc(sizeof(int) * (size_t) sl);
            for (int j = 0; j < sl; j++) cps[j] = byte_to_uni[(unsigned char) s[j]];
            t->special_cps[k] = cps;
            t->special_len[k] = sl;
            t->special_id[k] = i;
            k++;
        }
    }
    // merges: "a b" pairs
    if (t->merges_raw && t->n_merges_raw > 0) {
        int n = t->n_merges_raw;
        t->n_merges = n;
        t->mcap = next_pow2(n * 2 + 16);
        t->mkey = (uint64_t *) calloc(t->mcap, sizeof(uint64_t));
        t->mval = (int32_t *) malloc(sizeof(int32_t) * t->mcap);
        for (int i = 0; i < n; i++) {
            char * dup = strdup(t->merges_raw[i]);
            char * sp = strchr(dup, ' ');
            if (sp) {
                *sp = 0;
                int ia = tok_str_id(t, dup);
                int ib = tok_str_id(t, sp+1);
                if (ia >= 0 && ib >= 0) tok_merge_put(t, ia, ib, i);
            }
            free(dup);
        }
        for (int i = 0; i < n; i++) free(t->merges_raw[i]);
        free(t->merges_raw);
        t->merges_raw = NULL;
        t->n_merges_raw = 0;
    }
}

// longest special token starting at cps[i] (index into the special tables), -1 if none
static int tok_special_match(const tok_t * t, const uint32_t * cps, int i, int ncp) {
    int best = -1, best_len = 0;
    for (int s = 0; s < t->n_special; s++) {
        int n = t->special_len[s];
        if (i + n > ncp || n <= best_len) continue;
        if (t->special_cps[s][0] != cps[i]) continue;
        int ok = 1;
        for (int j = 1; j < n; j++) {
            if (t->special_cps[s][j] != cps[i+j]) { ok = 0; break; }
        }
        if (ok) { best = s; best_len = n; }
    }
    return best;
}

// ---------------------------------------------------------------------------
// model structures
// ---------------------------------------------------------------------------
typedef struct {
    ggml_tensor * attn_norm;            // input_layernorm [embd]
    ggml_tensor * post_attention_norm;  // [embd]
    ggml_tensor * ffn_gate, * ffn_up, * ffn_down;
    // full attention
    ggml_tensor * attn_q, * attn_k, * attn_v, * attn_output;
    ggml_tensor * attn_q_norm, * attn_k_norm;
    // gemma3 extra norms (post-attention and around the FFN)
    ggml_tensor * ffn_norm;           // pre-FFN norm [embd]
    ggml_tensor * post_ffw_norm;      // post-FFN norm  [embd]
    // sliding-window attention (gemma3): window size in tokens for this block
    int win;                          // rows allocated in the KV cache (= full n_ctx or sliding window)
    int is_swa;                       // 1 if this layer uses the sliding window
    // GDN (linear attention)
    ggml_tensor * attn_qkv, * attn_gate;
    ggml_tensor * ssm_beta, * ssm_alpha, * ssm_a, * ssm_dt;
    ggml_tensor * ssm_conv1d, * ssm_norm, * ssm_out;
    // LFM2 shortconv (short-convolution) module
    ggml_tensor * shortconv_conv, * shortconv_in_proj, * shortconv_out_proj;
    int is_full_attn;
} llm_block_t;

// post-warmup snapshot of the LLM state (system / tools prompt already in the
// KV cache). Used as the per-request reset point and as the on-disk cache.
typedef struct llm_session_s {
    int n_layer;
    float ** k_cache;                 // per full-attn layer (rows saved)
    float ** v_cache;
    int    * kv_rows;                 // saved row count per layer (0 = no rows)
    float ** gdn_state;               // per GDN layer
    float ** gdn_conv;
    float ** lfm2_state;              // per LFM2 shortconv layer
    int n_past;
    int * chat_toks; int chat_n; int chat_from; int chat_msg_count;
    int n_msg;
    char * msg_role[LLM_MAX_MSG];
    char * msg_content[LLM_MAX_MSG];
} llm_session_t;

typedef struct { float l; int i; } llm_cand_t;

typedef struct llm_s {
    gguf_file_t file;
    tok_t tok;
    // architecture
    llm_arch_t arch; char arch_name[64];
    // config
    int n_embd, n_embd_eff, n_ff, n_layer, n_head, n_head_kv, head_dim;
    float f_norm_rms_eps, f_rope_freq_base, f_rope_freq_swa;
    int n_rope; int rope_sections[4];
    float f_embd_scale;
    // gemma3
    int sliding_window;               // number of tokens for the sliding-window layers
    int swa_period;                   // every k-th layer is full attention (gemma3 = 6)
    int turn_end_id;                  // token id for <end_of_turn>
    // rope tables [n_ctx][n_rope/2]; _swa variant uses the smaller base for SWA layers
    float * rope_cos_swa, * rope_sin_swa;
    int n_layer_main;   // main layers (excludes trailing MTP/nextn layers)
    int ssm_conv_kernel, ssm_state_size, ssm_num_k_heads, ssm_num_v_heads, ssm_head_dim, ssm_inner_size;
    int full_attn_interval;
    int n_ctx, n_ctx_model, n_past;
    // rope tables [n_ctx][n_rope/2]
    float * rope_cos, * rope_sin;
    // llama3-style scaled rope: per-frequency factors from rope_freqs.weight.
    // angle[i] = pos * freq_base^(-2i/n_rope) / rope_freqs[i]
    float * rope_freqs;         // [n_rope/2]
    int     n_rope_freqs;
    // weights
    ggml_tensor * w_token_embd, * w_output, * w_output_norm;
    llm_block_t * blocks;
    // ggml
    ggml_context * ctx_w;       void * w_buf; void * f32_w_buf;
    ggml_context * ctx_compute; void * compute_buf; size_t compute_buf_size;
    ggml_context * ctx_graph;   void * graph_buf;
    ggml_cgraph * graph;
    int n_threads;
#ifdef GGML_USE_VULKAN
    // Vulkan GPU offload (NULL when unavailable): quantized matmul weights are
    // copied to VRAM and mul_mats run on the backend; attention/norms stay CPU.
    ggml_backend_t         vk;        // GPU backend
    ggml_backend_buffer_t  vk_wbuf;   // weights in VRAM
    ggml_backend_buffer_t  vk_cbuf;   // persistent per-call compute buffer (reused)
    ggml_context         * vk_ctx;    // per-call tensor context (no_alloc, reset each call)
    ggml_context         * vk_gctx;   // per-call graph context (not reset)
    ggml_cgraph          * vk_graph;  // per-call graph
    int                    vk_off;    // 1 when a GPU is in use
    void * embd_raw;                 // scratch: raw quantized embedding row read back from GPU
#endif
    // params
    llm_params_t params;
    // caches (full-attn blocks)
    float ** k_cache, ** v_cache;
    float ** gdn_state, ** gdn_conv;  // per GDN block
    // LFM2 shortconv recurrent state (per shortconv block): [d_conv][n_embd]
    float ** lfm2_state;
    int      shortconv_l_cache;      // conv kernel length (LFM2)
    // sampling
    uint64_t rng_state;
    float * work; llm_cand_t * cand;
    int * gen_hist; int n_gen_hist;
    // stats
    int64_t n_tokens; double t_load_ms, t_prompt_ms, t_gen_ms;
    int n_prompt_tokens, n_gen_tokens;
    // messages
    char * msg_role[LLM_MAX_MSG]; char * msg_content[LLM_MAX_MSG]; int n_msg;
    // tools
    llm_tool_t * tools; int n_tools;
    // chat KV-cache state (reused across turns)
    int * chat_toks; int chat_n; int chat_from; int chat_msg_count;
    // mirror of the exact token stream currently held in the KV/GDN caches.
    // Validates llm_generate_from() continuations: a mismatched prefix falls
    // back to a fresh full pass instead of reading foreign cached state.
    int * cache_toks; int cache_n; int cache_cap; int cache_ok;
    // session cache: per-request reset point + on-disk warmup cache
    llm_session_t * session;
    char model_path[512];
    uint64_t model_size, model_mtime;
    // logits
    float * logits; int n_logits;
    // direct-quantized-GEMV scratch (max weight row + quantized row)
    float * mm_xbuf; void * mm_qbuf; size_t mm_cap;
    // scratch (sized for B=1 token)
    float * scratch; size_t scratch_size;
    // batched prompt-processing scratch (llama/gemma3, CPU only)
    int    batch_max;   // max tokens per prompt batch
    float * bx, * br, * bout, * bo;
    float * bq, * bk, * bv;
    float * bfg, * bfu;
    float * bscores;   // [n_ctx] attention scores
    void   * bq8;     // batched quantized-input buffer (B * mm_cap * 2)
    float * bvin;     // batched padded-input buffer for the GPU path (B * mm_cap)
    float * scores_mt; // per-thread attention scores [n_threads * n_ctx]
} llm_t;

// ---------------------------------------------------------------------------
// scratch buffer layout (union of full-attention and GDN block intermediates)
// ---------------------------------------------------------------------------
typedef struct {
    float * x, * r;                 // residual stream + residual
    float * qkv, * z, * conv_out;   // GDN
    float * q, * k, * v;            // GDN (from conv_out)
    float * o, * out;               // attention out / block out
    float * q_full, * k_cur, * v_cur; // full attention
    float * ffn_g, * ffn_u, * ffn_out;
    float * scores;                 // [n_ctx]
    float * beta, * alpha, * u;     // GDN scalars
    // LFM2 shortconv scratch
    float * sproj;                  // [3*n_embd] in_proj output
    float * sgated;                 // [n_embd]   b*x (pre-conv gate)
    float * sconv;                  // [n_embd]   conv output
} llm_bufs_t;

static llm_bufs_t llm_bufs(llm_t * l, int B) {
    llm_bufs_t b;
    float * sc = l->scratch;
    size_t off = 0;
    int key_dim = l->ssm_num_k_heads * l->ssm_head_dim;
    int value_dim = l->ssm_num_v_heads * l->ssm_head_dim;
    int conv_dim = key_dim * 2 + value_dim;
    int q_full_dim = l->n_head * l->head_dim * 2;
    int kv_dim = l->n_head_kv * l->head_dim;
    // the attention / GDN output buffer holds nh*hd or nv*hd floats, which can
    // exceed n_embd (e.g. Qwen3.5-4B: nh*hd = nv*hd = 4096 > embd 2560)
    int o_dim = l->n_embd_eff;
    if (l->n_head * l->head_dim > o_dim) o_dim = l->n_head * l->head_dim;
    if (l->ssm_num_v_heads * l->ssm_head_dim > o_dim) o_dim = l->ssm_num_v_heads * l->ssm_head_dim;
    b.x        = sc; off += (size_t) B * l->n_embd_eff;
    b.r        = sc + off; off += (size_t) B * l->n_embd_eff;
    b.qkv      = sc + off; off += (size_t) B * conv_dim;
    b.z        = sc + off; off += (size_t) B * value_dim;
    b.conv_out = sc + off; off += (size_t) B * conv_dim;
    b.q        = sc + off; off += (size_t) B * key_dim;
    b.k        = sc + off; off += (size_t) B * key_dim;
    b.v        = sc + off; off += (size_t) B * value_dim;
    b.o        = sc + off; off += (size_t) B * o_dim;
    b.out      = sc + off; off += (size_t) B * l->n_embd_eff;
    b.q_full   = sc + off; off += (size_t) B * q_full_dim;
    b.k_cur    = sc + off; off += (size_t) B * kv_dim;
    b.v_cur    = sc + off; off += (size_t) B * kv_dim;
    b.ffn_g    = sc + off; off += (size_t) B * l->n_ff;
    b.ffn_u    = sc + off; off += (size_t) B * l->n_ff;
    b.ffn_out  = sc + off; off += (size_t) B * l->n_embd_eff;
    b.scores   = sc + off; off += (size_t) l->n_ctx;
    b.beta     = sc + off; off += (size_t) B * l->ssm_num_v_heads;
    b.alpha    = sc + off; off += (size_t) B * l->ssm_num_v_heads;
    b.u        = sc + off; off += (size_t) B * l->ssm_head_dim;
    int lc = 3 * l->n_embd;
    b.sproj  = sc + off; off += (size_t) B * lc;
    b.sgated = sc + off; off += (size_t) B * l->n_embd;
    b.sconv  = sc + off; off += (size_t) B * l->n_embd;
    return b;
}

// ---------------------------------------------------------------------------
// small math helpers
// ---------------------------------------------------------------------------
static void llm_rms_norm(const float * x, const float * w, int n, float eps, float * out) {
    float ss = 0.f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float r = 1.0f / sqrtf(ss / (float) n + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * r * w[i];
}

static void llm_l2_norm(float * v, int n, float eps) {
    float ss = 0.f;
    for (int i = 0; i < n; i++) ss += v[i] * v[i];
    float r = 1.0f / sqrtf(ss + eps);
    for (int i = 0; i < n; i++) v[i] *= r;
}

// RoPE over the first n_rope dims, applied to pairs (i, i+half), half = n_rope/2.
// For text-only (rope_sections = [11,11,10,0]) with no mrope section for this
// position this reduces to standard Neox rope over n_rope dims.
static void llm_rope_tables(const float * cos_t, const float * sin_t, float * v, int pos, int n_rope) {
    int half = n_rope / 2;
    for (int i = 0; i < half; i++) {
        float c = cos_t[(size_t) pos * half + i];
        float s = sin_t[(size_t) pos * half + i];
        float a = v[i], b = v[i + half];
        v[i] = a * c - b * s;
        v[i + half] = a * s + b * c;
    }
}
static void llm_rope(llm_t * l, float * v, int pos, int hd) {
    (void) hd;
    llm_rope_tables(l->rope_cos, l->rope_sin, v, pos, l->n_rope);
}

// ---------------------------------------------------------------------------
// SIMD attention helpers (AVX2/FMA when compiled with -mavx2 -mfma; the
// scalar fallback keeps correctness on any x86/ARM build)
// ---------------------------------------------------------------------------
#if defined(__AVX2__)
static inline float llm_hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    __m128 sh = _mm_movehdup_ps(lo);
    lo = _mm_add_ps(lo, sh);
    sh = _mm_movehl_ps(sh, lo);
    lo = _mm_add_ps(lo, sh);
    return _mm_cvtss_f32(lo);
}
#endif

// dot product over a head of length hd
static inline float llm_dot_hd(const float * a, const float * b, int hd) {
#if defined(__AVX2__) && defined(__FMA__)
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= hd; i += 8)
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc);
    float s = llm_hsum256(acc);
    for (; i < hd; i++) s += a[i] * b[i];
    return s;
#elif defined(__AVX2__)
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= hd; i += 8)
        acc = _mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)), acc);
    float s = llm_hsum256(acc);
    for (; i < hd; i++) s += a[i] * b[i];
    return s;
#else
    float s = 0.f;
    for (int i = 0; i < hd; i++) s += a[i] * b[i];
    return s;
#endif
}

// oh[d] = sum_j sc[j] * vc[off(j)*krow + d]. rows==NULL means off(j)=j (a
// contiguous KV cache slice); otherwise rows[j] is the KV slot for score j
// (sliding-window wrap). Used by the single-token attention hot path.
static inline void llm_attn_apply_hd(const float * sc, const float * vc,
                                     const int * rows, int krow, int n, int hd,
                                     float * oh) {
#if defined(__AVX2__) && defined(__FMA__)
    int d = 0;
    for (; d + 8 <= hd; d += 8) {
        __m256 acc = _mm256_setzero_ps();
        for (int j = 0; j < n; j++) {
            size_t off = rows ? (size_t) rows[j] * krow : (size_t) j * krow;
            acc = _mm256_fmadd_ps(_mm256_set1_ps(sc[j]),
                                  _mm256_loadu_ps(vc + off + d), acc);
        }
        _mm256_storeu_ps(oh + d, acc);
    }
    for (; d < hd; d++) {
        float acc = 0.f;
        for (int j = 0; j < n; j++) {
            size_t off = rows ? (size_t) rows[j] * krow : (size_t) j * krow;
            acc += sc[j] * vc[off + d];
        }
        oh[d] = acc;
    }
#elif defined(__AVX2__)
    int d = 0;
    for (; d + 8 <= hd; d += 8) {
        __m256 acc = _mm256_setzero_ps();
        for (int j = 0; j < n; j++) {
            size_t off = rows ? (size_t) rows[j] * krow : (size_t) j * krow;
            acc = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(sc[j]),
                                              _mm256_loadu_ps(vc + off + d)), acc);
        }
        _mm256_storeu_ps(oh + d, acc);
    }
    for (; d < hd; d++) {
        float acc = 0.f;
        for (int j = 0; j < n; j++) {
            size_t off = rows ? (size_t) rows[j] * krow : (size_t) j * krow;
            acc += sc[j] * vc[off + d];
        }
        oh[d] = acc;
    }
#else
    for (int d = 0; d < hd; d++) {
        float acc = 0.f;
        for (int j = 0; j < n; j++) {
            size_t off = rows ? (size_t) rows[j] * krow : (size_t) j * krow;
            acc += sc[j] * vc[off + d];
        }
        oh[d] = acc;
    }
#endif
}

// ---------------------------------------------------------------------------
// matmul helper: out[t*n_out + o] = sum_i x[t*n_in + i] * W[i*n_out + o]
// x: [n_tokens][n_in] row-major (feature fast); W: ggml [n_in, n_out]
// ---------------------------------------------------------------------------
static void llm_gemm_f32(const float * W, const float * x, int n_in, int row_stride,
                         int n_out, float * out, int n_threads);

// direct quantized GEMV: out[o] = sum_i W[i][o] * x[i].
// Uses the same ggml vec_dot kernels as ggml_mul_mat but skips the per-call
// graph/context/thread-pool machinery (weights are always single-token here).
static void llm_mm(llm_t * l, const float * x, int n_tokens, int n_in,
                   ggml_tensor * W, float * out, int n_out);
#ifdef GGML_USE_VULKAN
static void llm_mm_vk_batch(llm_t * l, ggml_tensor * W, const float * x, int n_in, int B,
                            float * out, int n_out);
#endif
static void llm_mm_gemv(llm_t * l, ggml_tensor * W, const float * x, int n_in, float * out, int n_out) {
    const struct ggml_type_traits_cpu * tr = ggml_get_type_traits_cpu(W->type);
    const enum ggml_type vdt = tr->vec_dot_type;
    const ggml_from_float_t from_float = ggml_get_type_traits_cpu(vdt)->from_float;
    const ggml_vec_dot_t vec_dot = tr->vec_dot;

    const int64_t ne0 = W->ne[0]; // padded row length (block-size multiples)
    float * xp = l->mm_xbuf;
    if ((int64_t) n_in < ne0) {
        memcpy(xp, x, sizeof(float) * (size_t) n_in);
        memset(xp + n_in, 0, sizeof(float) * (size_t)(ne0 - n_in));
    } else {
        memcpy(xp, x, sizeof(float) * (size_t) ne0);
    }
    from_float(xp, l->mm_qbuf, ne0); // quantize the input row to the dot type

    const char * wbase = (const char *) W->data;
    const size_t wstride = W->nb[1];
    #pragma omp parallel for if(n_out >= 256) num_threads(l->n_threads)
    for (int o = 0; o < n_out; o++) {
        vec_dot((int) ne0, &out[o], 0, wbase + (size_t) o * wstride, 0, l->mm_qbuf, 0, 1);
    }
}

// batched CPU matmul: out[t*n_out + o] = sum_i W[o][i] * x[t*n_in + i].
// Reuses the ggml vec_dot kernels; the quantized input columns are produced
// once and each weight row is re-used across all B columns (cache-friendly).
static void llm_mm_batch_cpu(llm_t * l, ggml_tensor * W, const float * x, int n_in, int B,
                             float * out, int n_out) {
    if (B <= 1) { llm_mm(l, x, 1, n_in, W, out, n_out); return; }
#ifdef GGML_USE_VULKAN
    if (W->buffer) { llm_mm_vk_batch(l, W, x, n_in, B, out, n_out); return; }
#endif
    if (W->type == GGML_TYPE_F32) {
        for (int t = 0; t < B; t++)
            llm_gemm_f32((const float *) W->data, x + (size_t) t * n_in, n_in,
                         (int) W->ne[0], n_out, out + (size_t) t * n_out, l->n_threads);
        return;
    }
    const struct ggml_type_traits_cpu * tr = ggml_get_type_traits_cpu(W->type);
    const enum ggml_type vdt = tr->vec_dot_type;
    const ggml_from_float_t from_float = ggml_get_type_traits_cpu(vdt)->from_float;
    const ggml_vec_dot_t vec_dot = tr->vec_dot;
    const int64_t ne0 = W->ne[0];
    // column size in the q-buffer: block of 32 floats; q8_K=64B, q8_0=40B
    const size_t nblocks = ((size_t) ne0 + 31) / 32;
    const size_t col_sz = nblocks * (vdt == GGML_TYPE_Q8_K ? 64 : 40);
    const size_t stride = col_sz;
    for (int t = 0; t < B; t++) {
        const float * xt = x + (size_t) t * n_in;
        float * xp = l->mm_xbuf;
        if ((int64_t) n_in < ne0) {
            memcpy(xp, xt, sizeof(float) * (size_t) n_in);
            memset(xp + n_in, 0, sizeof(float) * (size_t)(ne0 - n_in));
        } else {
            memcpy(xp, xt, sizeof(float) * (size_t) ne0);
        }
        from_float(xp, (char *) l->bq8 + (size_t) t * stride, ne0);
    }
    const char * wbase = (const char *) W->data;
    const size_t wstride = W->nb[1];
    #pragma omp parallel for if(B * n_out >= 4096) num_threads(l->n_threads) schedule(static)
    for (int o = 0; o < n_out; o++) {
        const char * wrow = wbase + (size_t) o * wstride;
        for (int t = 0; t < B; t++) {
            vec_dot((int) ne0, &out[(size_t) t * n_out + o], 0, wrow, 0,
                    (const char *) l->bq8 + (size_t) t * stride, 0, 1);
        }
    }
}

#ifdef GGML_USE_VULKAN
// Vulkan offload for a single quantized matmul: out[o] = sum_i W[i][o] * x[i].
// Builds a one-op mul_mat graph on the GPU backend and copies the result back.
static void llm_mm_vk(llm_t * l, ggml_tensor * W, const float * x, int n_in, float * out, int n_out) {
    ggml_reset(l->vk_ctx);
    ggml_backend_buffer_reset(l->vk_cbuf);
    const int64_t ne0 = W->ne[0];
    // zero-pad the input up to the weight's padded row length
    float * xp = l->mm_xbuf;
    if ((int64_t) n_in < ne0) {
        memcpy(xp, x, sizeof(float) * (size_t) n_in);
        memset(xp + n_in, 0, sizeof(float) * (size_t)(ne0 - n_in));
    } else {
        memcpy(xp, x, sizeof(float) * (size_t) ne0);
    }
    ggml_tensor * in = ggml_new_tensor_2d(l->vk_ctx, GGML_TYPE_F32, ne0, 1);
    ggml_tensor * y  = ggml_mul_mat(l->vk_ctx, W, in);
    void * base = ggml_backend_buffer_get_base(l->vk_cbuf);
    ggml_backend_tensor_alloc(l->vk_cbuf, in, base);
    size_t in_sz = ggml_backend_buffer_get_alloc_size(l->vk_cbuf, in);
    ggml_backend_tensor_alloc(l->vk_cbuf, y, (char *) base + in_sz);
    ggml_backend_tensor_set(in, xp, 0, (size_t) ne0 * sizeof(float));
    ggml_graph_clear(l->vk_graph);
    ggml_build_forward_expand(l->vk_graph, y);
    enum ggml_status st = ggml_backend_graph_compute(l->vk, l->vk_graph);
    if (st != GGML_STATUS_SUCCESS) {
        LOG_ERROR("vulkan: graph compute failed: %d\n", (int) st);
        memset(out, 0, sizeof(float) * (size_t) n_out);
        return;
    }
    ggml_backend_tensor_get(y, out, 0, (size_t) n_out * sizeof(float));
}

// batched Vulkan matmul: out[t*n_out + o] = sum_i W[o][i] * x[t*n_in + i].
// One mul_mat graph with a [ne0, B] input tensor; the B padded input columns
// are uploaded once, all B output columns are read back once.
static void llm_mm_vk_batch(llm_t * l, ggml_tensor * W, const float * x, int n_in, int B,
                            float * out, int n_out) {
    ggml_reset(l->vk_ctx);
    ggml_backend_buffer_reset(l->vk_cbuf);
    const int64_t ne0 = W->ne[0];
    float * xp = l->bvin;
    for (int t = 0; t < B; t++) {
        float * dst = xp + (size_t) t * ne0;
        const float * src = x + (size_t) t * n_in;
        if ((int64_t) n_in < ne0) {
            memcpy(dst, src, sizeof(float) * (size_t) n_in);
            memset(dst + n_in, 0, sizeof(float) * (size_t)(ne0 - n_in));
        } else {
            memcpy(dst, src, sizeof(float) * (size_t) ne0);
        }
    }
    ggml_tensor * in = ggml_new_tensor_2d(l->vk_ctx, GGML_TYPE_F32, ne0, B);
    ggml_tensor * y  = ggml_mul_mat(l->vk_ctx, W, in);
    void * base = ggml_backend_buffer_get_base(l->vk_cbuf);
    ggml_backend_tensor_alloc(l->vk_cbuf, in, base);
    size_t in_sz = ggml_backend_buffer_get_alloc_size(l->vk_cbuf, in);
    ggml_backend_tensor_alloc(l->vk_cbuf, y, (char *) base + in_sz);
    ggml_backend_tensor_set(in, xp, 0, (size_t) ne0 * B * sizeof(float));
    ggml_graph_clear(l->vk_graph);
    ggml_build_forward_expand(l->vk_graph, y);
    enum ggml_status st = ggml_backend_graph_compute(l->vk, l->vk_graph);
    if (st != GGML_STATUS_SUCCESS) {
        LOG_ERROR("vulkan: batch graph compute failed: %d\n", (int) st);
        memset(out, 0, sizeof(float) * (size_t) n_out * B);
        return;
    }
    ggml_backend_tensor_get(y, out, 0, (size_t) n_out * B * sizeof(float));
}
#endif // GGML_USE_VULKAN

static void llm_mm(llm_t * l, const float * x, int n_tokens, int n_in,
                   ggml_tensor * W, float * out, int n_out) {
    if (n_tokens == 1) {
#ifdef GGML_USE_VULKAN
        if (W->buffer) { llm_mm_vk(l, W, x, n_in, out, n_out); return; }
#endif
        if (W->type == GGML_TYPE_F32) {
            llm_gemm_f32((const float *) W->data, x, n_in, (int) W->ne[0], n_out, out, l->n_threads);
            return;
        }
        llm_mm_gemv(l, W, x, n_in, out, n_out);
        return;
    }
    // batched fallback (not used in single-token generation)
    ggml_reset(l->ctx_compute);
    ggml_graph_clear(l->graph);
    int n_in_eff = (W->ne[0] > n_in) ? (int) W->ne[0] : n_in;
    struct ggml_tensor * xt = ggml_new_tensor_2d(l->ctx_compute, GGML_TYPE_F32, n_in_eff, n_tokens);
    void * xd = ggml_get_data(xt);
    memset(xd, 0, sizeof(float) * (size_t) n_in_eff * n_tokens);
    memcpy(xd, x, sizeof(float) * (size_t) n_in * n_tokens);
    struct ggml_tensor * y = ggml_mul_mat(l->ctx_compute, W, xt);
    ggml_build_forward_expand(l->graph, y);
    enum ggml_status st = ggml_graph_compute_with_ctx(l->ctx_compute, l->graph, l->n_threads);
    if (st != GGML_STATUS_SUCCESS) { LOG_ERROR("graph compute failed: %d\n", (int) st); return; }
    memcpy(out, ggml_get_data(y), sizeof(float) * (size_t) n_out * n_tokens);
}

// ---------------------------------------------------------------------------
// multi-matmul: several matmuls sharing the same input column, executed as one
// operation. On Vulkan all matmuls go into a single graph submission (one
// input upload, one compute, one readback) instead of one submission each;
// on CPU the shared input is quantized once and reused across all weights.
// This is the per-token generation hot path, where the per-matmul Vulkan
// graph/reset/upload/sync overhead was dominating GPU generation speed.
// ---------------------------------------------------------------------------
static void llm_mm_multi(llm_t * l, const float * x, int n_in,
                         ggml_tensor * const * Ws, float * const * outs, int n_mm);

// CPU batched path: quantize x once, then vec_dot against every weight row.
// All weights must share the same ggml type and padded row length (ne[0]);
// otherwise fall back to one llm_mm per op.
static void llm_mm_multi_cpu(llm_t * l, const float * x, int n_in,
                             ggml_tensor * const * Ws, float * const * outs, int n_mm) {
    if (n_mm == 1) { llm_mm(l, x, 1, n_in, Ws[0], outs[0], (int) Ws[0]->ne[1]); return; }
    for (int i = 1; i < n_mm; i++) {
        if (Ws[i]->type != Ws[0]->type || Ws[i]->ne[0] != Ws[0]->ne[0]) {
            for (int k = 0; k < n_mm; k++) llm_mm(l, x, 1, n_in, Ws[k], outs[k], (int) Ws[k]->ne[1]);
            return;
        }
    }
    const struct ggml_type_traits_cpu * tr = ggml_get_type_traits_cpu(Ws[0]->type);
    const enum ggml_type vdt = tr->vec_dot_type;
    const ggml_from_float_t from_float = ggml_get_type_traits_cpu(vdt)->from_float;
    const ggml_vec_dot_t vec_dot = tr->vec_dot;
    const int64_t ne0 = Ws[0]->ne[0];
    float * xp = l->mm_xbuf;
    if ((int64_t) n_in < ne0) {
        memcpy(xp, x, sizeof(float) * (size_t) n_in);
        memset(xp + n_in, 0, sizeof(float) * (size_t)(ne0 - n_in));
    } else {
        memcpy(xp, x, sizeof(float) * (size_t) ne0);
    }
    from_float(xp, l->mm_qbuf, ne0); // quantize the shared input once
    int total = 0;
    for (int i = 0; i < n_mm; i++) total += (int) Ws[i]->ne[1];
    #pragma omp parallel for if(total >= 256) num_threads(l->n_threads) schedule(static)
    for (int o = 0; o < total; o++) {
        int w = 0, rem = o;
        while (w + 1 < n_mm && rem >= (int) Ws[w]->ne[1]) { rem -= (int) Ws[w]->ne[1]; w++; }
        const char * wrow = (const char *) Ws[w]->data + (size_t) rem * Ws[w]->nb[1];
        vec_dot((int) ne0, &outs[w][rem], 0, wrow, 0, l->mm_qbuf, 0, 1);
    }
}

#ifdef GGML_USE_VULKAN
// Vulkan batched path: one mul_mat graph with a single shared input tensor and
// one output tensor per weight. All weights must share the same ne[0].
static void llm_mm_vk_multi(llm_t * l, const float * x, int n_in,
                            ggml_tensor * const * Ws, float * const * outs, int n_mm) {
    ggml_reset(l->vk_ctx);
    ggml_backend_buffer_reset(l->vk_cbuf);
    const int64_t ne0 = Ws[0]->ne[0];
    float * xp = l->mm_xbuf;
    if ((int64_t) n_in < ne0) {
        memcpy(xp, x, sizeof(float) * (size_t) n_in);
        memset(xp + n_in, 0, sizeof(float) * (size_t)(ne0 - n_in));
    } else {
        memcpy(xp, x, sizeof(float) * (size_t) ne0);
    }
    ggml_tensor * in = ggml_new_tensor_2d(l->vk_ctx, GGML_TYPE_F32, ne0, 1);
    ggml_tensor * ys[8];
    for (int i = 0; i < n_mm; i++) ys[i] = ggml_mul_mat(l->vk_ctx, Ws[i], in);
    void * base = ggml_backend_buffer_get_base(l->vk_cbuf);
    size_t off = 0;
    ggml_backend_tensor_alloc(l->vk_cbuf, in, (char *) base + off);
    off += ggml_backend_buffer_get_alloc_size(l->vk_cbuf, in);
    for (int i = 0; i < n_mm; i++) {
        ggml_backend_tensor_alloc(l->vk_cbuf, ys[i], (char *) base + off);
        off += ggml_backend_buffer_get_alloc_size(l->vk_cbuf, ys[i]);
    }
    ggml_backend_tensor_set(in, xp, 0, (size_t) ne0 * sizeof(float));
    ggml_graph_clear(l->vk_graph);
    for (int i = 0; i < n_mm; i++) ggml_build_forward_expand(l->vk_graph, ys[i]);
    enum ggml_status st = ggml_backend_graph_compute(l->vk, l->vk_graph);
    if (st != GGML_STATUS_SUCCESS) {
        LOG_ERROR("vulkan: multi graph compute failed: %d\n", (int) st);
        for (int i = 0; i < n_mm; i++)
            memset(outs[i], 0, sizeof(float) * (size_t) Ws[i]->ne[1]);
        return;
    }
    for (int i = 0; i < n_mm; i++)
        ggml_backend_tensor_get(ys[i], outs[i], 0, (size_t) Ws[i]->ne[1] * sizeof(float));
}

// fused per-token FFN on GPU: gate/up matmuls + silu/gelu + mul + down matmul in a
// single graph submission. All three weights must be GPU-resident and share the
// gate/up input row length; ffn_down may have its own row length (its input is the
// [ff] activation produced on the GPU, so no CPU roundtrip happens in between).
// Returns 1 when handled (out holds the result), 0 to fall back to the CPU path.
static int llm_ffn_vk(llm_t * l, ggml_tensor * ffn_gate, ggml_tensor * ffn_up,
                      ggml_tensor * ffn_down, const float * x, int n_in,
                      float * out, int n_out, int act_kind) {
    if (!l->vk) return 0;
    if (!ffn_gate || !ffn_up || !ffn_down) return 0;
    if (!ffn_gate->buffer || !ffn_up->buffer || !ffn_down->buffer) return 0;
    if (ffn_gate->ne[0] != ffn_up->ne[0]) return 0;  // must share the input row length
    ggml_reset(l->vk_ctx);
    ggml_backend_buffer_reset(l->vk_cbuf);
    const int64_t ne0 = ffn_gate->ne[0];
    float * xp = l->mm_xbuf;
    if ((int64_t) n_in < ne0) {
        memcpy(xp, x, sizeof(float) * (size_t) n_in);
        memset(xp + n_in, 0, sizeof(float) * (size_t)(ne0 - n_in));
    } else {
        memcpy(xp, x, sizeof(float) * (size_t) ne0);
    }
    ggml_tensor * in  = ggml_new_tensor_2d(l->vk_ctx, GGML_TYPE_F32, ne0, 1);
    ggml_tensor * gate = ggml_mul_mat(l->vk_ctx, ffn_gate, in);
    ggml_tensor * up   = ggml_mul_mat(l->vk_ctx, ffn_up, in);
    ggml_tensor * act  = (act_kind == 1) ? ggml_gelu(l->vk_ctx, gate) : ggml_silu(l->vk_ctx, gate);
    ggml_tensor * g    = ggml_mul(l->vk_ctx, act, up);
    ggml_tensor * y    = ggml_mul_mat(l->vk_ctx, ffn_down, g);
    void * base = ggml_backend_buffer_get_base(l->vk_cbuf);
    size_t off = 0;
    ggml_backend_tensor_alloc(l->vk_cbuf, in, (char *) base + off);
    off += ggml_backend_buffer_get_alloc_size(l->vk_cbuf, in);
    ggml_backend_tensor_alloc(l->vk_cbuf, gate, (char *) base + off);
    off += ggml_backend_buffer_get_alloc_size(l->vk_cbuf, gate);
    ggml_backend_tensor_alloc(l->vk_cbuf, up, (char *) base + off);
    off += ggml_backend_buffer_get_alloc_size(l->vk_cbuf, up);
    ggml_backend_tensor_alloc(l->vk_cbuf, act, (char *) base + off);
    off += ggml_backend_buffer_get_alloc_size(l->vk_cbuf, act);
    ggml_backend_tensor_alloc(l->vk_cbuf, g, (char *) base + off);
    off += ggml_backend_buffer_get_alloc_size(l->vk_cbuf, g);
    ggml_backend_tensor_alloc(l->vk_cbuf, y, (char *) base + off);
    ggml_backend_tensor_set(in, xp, 0, (size_t) ne0 * sizeof(float));
    ggml_graph_clear(l->vk_graph);
    ggml_build_forward_expand(l->vk_graph, y);
    enum ggml_status st = ggml_backend_graph_compute(l->vk, l->vk_graph);
    if (st != GGML_STATUS_SUCCESS) {
        LOG_ERROR("vulkan: ffn graph compute failed: %d\n", (int) st);
        memset(out, 0, sizeof(float) * (size_t) n_out);
        return 1;
    }
    ggml_backend_tensor_get(y, out, 0, (size_t) n_out * sizeof(float));
    return 1;
}
#endif // GGML_USE_VULKAN

static void llm_mm_multi(llm_t * l, const float * x, int n_in,
                         ggml_tensor * const * Ws, float * const * outs, int n_mm) {
    if (n_mm <= 0) return;
#ifdef GGML_USE_VULKAN
    int all_vk = 1;
    for (int i = 0; i < n_mm; i++) {
        if (!Ws[i]->buffer || Ws[i]->ne[0] != Ws[0]->ne[0]) { all_vk = 0; break; }
    }
    if (all_vk) { llm_mm_vk_multi(l, x, n_in, Ws, outs, n_mm); return; }
#endif
    llm_mm_multi_cpu(l, x, n_in, Ws, outs, n_mm);
}

// dequantize a row (token embedding) of a [n_embd, vocab] tensor
#ifdef GGML_USE_VULKAN
static void llm_dequant_row_any(const void * src, enum ggml_type t, float * dst, int64_t n);
#endif
static void llm_get_embd(const llm_t * l, int token_id, float * out) {
    ggml_tensor * e = l->w_token_embd;
#ifdef GGML_USE_VULKAN
    if (e->buffer) {
        // token_embd lives on the GPU: read the quantized row back and dequantize it
        size_t row = (size_t) e->nb[1];
        ggml_backend_tensor_get(e, l->embd_raw, (size_t) token_id * row, row);
        llm_dequant_row_any(l->embd_raw, e->type, out, e->ne[0]);
        return;
    }
#endif
    const void * row = (const char *) e->data + (size_t) token_id * e->nb[1];
    int64_t n = e->ne[0];
    switch (e->type) {
        case GGML_TYPE_F32: memcpy(out, row, (size_t) n*4); break;
        case GGML_TYPE_F16: {
            const uint16_t * s = (const uint16_t *) row;
            for (int64_t i = 0; i < n; i++) {
                uint16_t h = s[i];
                int sign = (h >> 15) & 1; int exp = (h >> 10) & 0x1F; int mant = h & 0x3FF;
                float v;
                if (exp == 0) v = ldexp((float) mant, -24);
                else v = ldexp((float) (mant + 1024), exp - 25);
                out[i] = sign ? -v : v;
            }
            break;
        }
        case GGML_TYPE_Q4_K: dequantize_row_q4_K((const block_q4_K *) row, out, n); break;
        case GGML_TYPE_Q5_K: dequantize_row_q5_K((const block_q5_K *) row, out, n); break;
        case GGML_TYPE_Q6_K: dequantize_row_q6_K((const block_q6_K *) row, out, n); break;
        case GGML_TYPE_Q8_0: dequantize_row_q8_0((const block_q8_0 *) row, out, n); break;
        case GGML_TYPE_Q8_K: dequantize_row_q8_K((const block_q8_K *) row, out, n); break;
        case GGML_TYPE_Q4_0: dequantize_row_q4_0((const block_q4_0 *) row, out, n); break;
        default:
            LOG_ERROR("unsupported embd type %d\n", e->type);
            memset(out, 0, (size_t) n*4);
            break;
    }
    const char * dbg_embd = getenv("LLM_DBG_EMBD");
    if (dbg_embd && atoi(dbg_embd) == token_id) {
        float ss = 0.f; for (int64_t i = 0; i < n; i++) ss += out[i]*out[i];
        fprintf(stderr, "[dbgembd] tok=%d type=%d n=%lld:", token_id, (int) e->type, (long long) n);
        for (int64_t i = 0; i < n; i += 256) {
            float bss = 0.f; int m = i+256 < n ? 256 : (int)(n-i);
            for (int j = 0; j < m; j++) bss += out[i+j]*out[i+j];
            fprintf(stderr, " b%lld rms=%.6f", (long long)(i/256), sqrtf(bss/(float)m));
        }
        fprintf(stderr, " totrms=%.6f\n", sqrtf(ss / (float) n));
        for (int64_t i = 0; i < 32 && i < n; i++) fprintf(stderr, "c[%2lld]=% .9f\n", (long long)i, (double) out[i]);
        fflush(stderr);
    }
}

// ---------------------------------------------------------------------------
// weight dequantization to F32 (bypasses the ggml quantized matmul kernels)
// ---------------------------------------------------------------------------
static void llm_dequant_row_any(const void * src, enum ggml_type t, float * dst, int64_t n) {
    switch (t) {
        case GGML_TYPE_F32: memcpy(dst, src, (size_t) n*4); break;
        case GGML_TYPE_F16: {
            const uint16_t * s = (const uint16_t *) src;
            for (int64_t i = 0; i < n; i++) {
                uint16_t h = s[i];
                int sign = (h >> 15) & 1; int exp = (h >> 10) & 0x1F; int mant = h & 0x3FF;
                float v = (exp == 0) ? ldexp((float) mant, -24) : ldexp((float) (mant + 1024), exp - 25);
                dst[i] = sign ? -v : v;
            }
            break;
        }
        case GGML_TYPE_Q4_0: dequantize_row_q4_0((const block_q4_0 *) src, dst, n); break;
        case GGML_TYPE_Q4_1: dequantize_row_q4_1((const block_q4_1 *) src, dst, n); break;
        case GGML_TYPE_Q5_0: dequantize_row_q5_0((const block_q5_0 *) src, dst, n); break;
        case GGML_TYPE_Q5_1: dequantize_row_q5_1((const block_q5_1 *) src, dst, n); break;
        case GGML_TYPE_Q8_0: dequantize_row_q8_0((const block_q8_0 *) src, dst, n); break;
        case GGML_TYPE_Q2_K: dequantize_row_q2_K((const block_q2_K *) src, dst, n); break;
        case GGML_TYPE_Q3_K: dequantize_row_q3_K((const block_q3_K *) src, dst, n); break;
        case GGML_TYPE_Q4_K: dequantize_row_q4_K((const block_q4_K *) src, dst, n); break;
        case GGML_TYPE_Q5_K: dequantize_row_q5_K((const block_q5_K *) src, dst, n); break;
        case GGML_TYPE_Q6_K: dequantize_row_q6_K((const block_q6_K *) src, dst, n); break;
        case GGML_TYPE_Q8_K: dequantize_row_q8_K((const block_q8_K *) src, dst, n); break;
        default:
            LOG_ERROR("llm_dequant_row_any: unsupported type %d\n", (int) t);
            memset(dst, 0, (size_t) n*4);
            break;
    }
}

static void llm_dequant_tensor_to_f32(const gguf_tensor_t * tt, const char * base, float * dst) {
    int64_t rows = tt->ne[1]*tt->ne[2]*tt->ne[3];
    if (rows < 1) rows = 1;
    const char * src = base + tt->offset;
    size_t row_src = ggml_row_size(tt->type, tt->ne[0]);
    for (int64_t r = 0; r < rows; r++)
        llm_dequant_row_any(src + (size_t) r * row_src, tt->type, dst + (size_t) r * tt->ne[0], tt->ne[0]);
}

// F32 GEMV: out[o] = sum_i W[i][o] * x[i];  W stored [row_stride, n_out] row-major
static void llm_gemm_f32(const float * W, const float * x, int n_in, int row_stride,
                         int n_out, float * out, int n_threads) {
    (void) n_threads;
    #pragma omp parallel for if(n_out >= 256) num_threads(n_threads)
    for (int o = 0; o < n_out; o++) {
        const float * w = W + (size_t) o * row_stride;
        float s = 0.f;
        for (int i = 0; i < n_in; i++) s += w[i] * x[i];
        out[o] = s;
    }
}

// ---------------------------------------------------------------------------
// block forwards
// ---------------------------------------------------------------------------

// precomputed activation lookup tables (sigmoid / gelu-tanh) with linear
// interpolation, replacing per-element expf()/tanhf() calls in the FFN hot
// path. 4096 entries over [-16,16] keep the interpolation error below ~1e-6.
#define LLM_ACT_LO  (-16.0f)
#define LLM_ACT_HI  ( 16.0f)
#define LLM_ACT_N   4096
static float llm_act_sigmoid[LLM_ACT_N + 1];
static float llm_act_gelu[LLM_ACT_N + 1];
static int   llm_act_ready = 0;

static float gemma_gelu_tanh(float x) {
    const float GELU_COEF_A = 0.044715f;
    const float SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;
    return 0.5f * x * (1.0f + tanhf(SQRT_2_OVER_PI * x * (1.0f + GELU_COEF_A * x * x)));
}

static void llm_act_build(void) {
    const float step = (LLM_ACT_HI - LLM_ACT_LO) / (float) LLM_ACT_N;
    for (int i = 0; i <= LLM_ACT_N; i++) {
        float x = LLM_ACT_LO + step * (float) i;
        llm_act_sigmoid[i] = 1.0f / (1.0f + expf(-x));
        llm_act_gelu[i] = gemma_gelu_tanh(x);
    }
    llm_act_ready = 1;
}

static inline float llm_sigmoid_tab(float x) {
    if (x <= LLM_ACT_LO) return 0.0f;
    if (x >= LLM_ACT_HI) return 1.0f;
    float t = (x - LLM_ACT_LO) * ((float) LLM_ACT_N / (LLM_ACT_HI - LLM_ACT_LO));
    int i = (int) t;
    float f = t - (float) i;
    return llm_act_sigmoid[i] + f * (llm_act_sigmoid[i + 1] - llm_act_sigmoid[i]);
}

static inline float llm_gelu_tanh_tab(float x) {
    if (x <= LLM_ACT_LO) return 0.0f;  // gelu_tanh(x) -> 0 as x -> -inf
    if (x >= LLM_ACT_HI) return x;     // gelu_tanh(x) -> x as x -> +inf
    float t = (x - LLM_ACT_LO) * ((float) LLM_ACT_N / (LLM_ACT_HI - LLM_ACT_LO));
    int i = (int) t;
    float f = t - (float) i;
    return llm_act_gelu[i] + f * (llm_act_gelu[i + 1] - llm_act_gelu[i]);
}

static void llm_forward_ffn(llm_t * l, llm_block_t * blk, const float * x, float * out, llm_bufs_t * buf) {
    int embd = l->n_embd, ff = l->n_ff;
#ifdef GGML_USE_VULKAN
    if (llm_ffn_vk(l, blk->ffn_gate, blk->ffn_up, blk->ffn_down, x, embd, out, embd, 0)) return;
#endif
    ggml_tensor * Ws[2] = { blk->ffn_gate, blk->ffn_up };
    float * outs[2] = { buf->ffn_g, buf->ffn_u };
    llm_mm_multi(l, x, embd, Ws, outs, 2);
    for (int i = 0; i < ff; i++) {
        float g = buf->ffn_g[i];
        buf->ffn_g[i] = llm_sigmoid_tab(g) * g * buf->ffn_u[i]; // silu(g)*u
    }
    llm_mm(l, buf->ffn_g, 1, ff, blk->ffn_down, out, embd);
}

static void llm_forward_ffn_gemma(llm_t * l, llm_block_t * blk, const float * x, float * out, llm_bufs_t * buf) {
    int embd = l->n_embd, ff = l->n_ff;
#ifdef GGML_USE_VULKAN
    if (llm_ffn_vk(l, blk->ffn_gate, blk->ffn_up, blk->ffn_down, x, embd, out, embd, 1)) return;
#endif
    ggml_tensor * Ws[2] = { blk->ffn_gate, blk->ffn_up };
    float * outs[2] = { buf->ffn_g, buf->ffn_u };
    llm_mm_multi(l, x, embd, Ws, outs, 2);
    for (int i = 0; i < ff; i++)
        buf->ffn_g[i] = llm_gelu_tanh_tab(buf->ffn_g[i]) * buf->ffn_u[i];
    llm_mm(l, buf->ffn_g, 1, ff, blk->ffn_down, out, embd);
}

// full attention block. x is the rms_norm'd input (residual handled by caller).
// gated: Q = [q | gate] per head; out = sigmoid(gate) * softmax(q kT) v
static void llm_forward_full(llm_t * l, int b, llm_block_t * blk, const float * x, float * out, int pos, llm_bufs_t * buf) {
    int embd = l->n_embd, hd = l->head_dim, nh = l->n_head, nhkv = l->n_head_kv;
    int qfd = nh * hd * 2;
    int kvd = nhkv * hd;
    ggml_tensor * Ws[3] = { blk->attn_q, blk->attn_k, blk->attn_v };
    float * outs[3] = { buf->q_full, buf->k_cur, buf->v_cur };
    llm_mm_multi(l, x, embd, Ws, outs, 3);
    const float * qw = (const float *) blk->attn_q_norm->data;
    const float * kw = (const float *) blk->attn_k_norm->data;
    for (int h = 0; h < nh; h++) {
        float * qh = buf->q_full + h * 2 * hd;
        llm_rms_norm(qh, qw, hd, l->f_norm_rms_eps, qh);
        llm_rope(l, qh, pos, hd);
    }
    for (int hv = 0; hv < nhkv; hv++) {
        float * kh = buf->k_cur + hv * hd;
        llm_rms_norm(kh, kw, hd, l->f_norm_rms_eps, kh);
        llm_rope(l, kh, pos, hd);
    }
    size_t krow = (size_t) nhkv * hd;
    memcpy(l->k_cache[b] + (size_t) pos * krow, buf->k_cur, krow * sizeof(float));
    memcpy(l->v_cache[b] + (size_t) pos * krow, buf->v_cur, krow * sizeof(float));
    float scale = 1.0f / sqrtf((float) hd);
    int g = nh / nhkv;
    #pragma omp parallel for if(nh >= 2) num_threads(l->n_threads)
    for (int h = 0; h < nh; h++) {
        const float * qh = buf->q_full + h * 2 * hd;
        const float * gate = buf->q_full + h * 2 * hd + hd;
        const float * kc = l->k_cache[b] + (size_t) (h / g) * hd;
        const float * vc = l->v_cache[b] + (size_t) (h / g) * hd;
        float * sc = l->scores_mt + (size_t) omp_get_thread_num() * l->n_ctx;
        for (int j = 0; j <= pos; j++)
            sc[j] = llm_dot_hd(qh, kc + (size_t) j * krow, hd) * scale;
        float mx = sc[0];
        for (int j = 1; j <= pos; j++) if (sc[j] > mx) mx = sc[j];
        float sum = 0.f;
        for (int j = 0; j <= pos; j++) { sc[j] = expf(sc[j] - mx); sum += sc[j]; }
        for (int j = 0; j <= pos; j++) sc[j] /= sum;
        float * oh = buf->o + h * hd;
        llm_attn_apply_hd(sc, vc, NULL, (int) krow, pos + 1, hd, oh);
        for (int d = 0; d < hd; d++) oh[d] /= (1.0f + expf(-gate[d]));
    }
    llm_mm(l, buf->o, 1, nh * hd, blk->attn_output, out, embd);
}

static void llm_forward_full_gemma(llm_t * l, int b, llm_block_t * blk, const float * x, float * out, int pos, llm_bufs_t * buf) {
    int embd = l->n_embd, hd = l->head_dim, nh = l->n_head, nhkv = l->n_head_kv;
    int qd = nh * hd;
    int kvd = nhkv * hd;
    int win = blk->win > 0 ? blk->win : l->n_ctx;
    ggml_tensor * Ws[3] = { blk->attn_q, blk->attn_k, blk->attn_v };
    float * outs[3] = { buf->q_full, buf->k_cur, buf->v_cur };
    llm_mm_multi(l, x, embd, Ws, outs, 3);
    const float * qw = (const float *) blk->attn_q_norm->data;
    const float * kw = (const float *) blk->attn_k_norm->data;
    const float * cos_t = blk->is_swa ? l->rope_cos_swa : l->rope_cos;
    const float * sin_t = blk->is_swa ? l->rope_sin_swa : l->rope_sin;
    for (int h = 0; h < nh; h++) {
        float * qh = buf->q_full + h * hd;
        llm_rms_norm(qh, qw, hd, l->f_norm_rms_eps, qh);
        llm_rope_tables(cos_t, sin_t, qh, pos, hd);
    }
    for (int hv = 0; hv < nhkv; hv++) {
        float * kh = buf->k_cur + hv * hd;
        llm_rms_norm(kh, kw, hd, l->f_norm_rms_eps, kh);
        llm_rope_tables(cos_t, sin_t, kh, pos, hd);
    }
    size_t krow = (size_t) nhkv * hd;
    int slot = pos % win;
    memcpy(l->k_cache[b] + (size_t) slot * krow, buf->k_cur, krow * sizeof(float));
    memcpy(l->v_cache[b] + (size_t) slot * krow, buf->v_cur, krow * sizeof(float));
    int start = pos + 1 > win ? pos - win + 1 : 0;
    float scale = 1.0f / sqrtf((float) hd);
    int g = nh / nhkv;
    // rows[] maps each score position to its KV slot (sliding-window wrap).
    // Built only when the window actually wraps (n <= win, win <= 512 for SWA).
    int rows[512];
    int use_rows = (win > 0 && start > 0 && (pos - start + 1) <= 512);
    if (use_rows) {
        for (int j = start; j <= pos; j++) rows[j - start] = (j % win);
    }
    #pragma omp parallel for if(nh >= 2) num_threads(l->n_threads)
    for (int h = 0; h < nh; h++) {
        const float * qh = buf->q_full + h * hd;
        const float * kc = l->k_cache[b] + (h / g) * hd;
        const float * vc = l->v_cache[b] + (h / g) * hd;
        float * sc = l->scores_mt + (size_t) omp_get_thread_num() * l->n_ctx;
        int n = 0;
        for (int j = start; j <= pos; j++)
            sc[n++] = llm_dot_hd(qh, kc + (size_t) (j % win) * krow, hd) * scale;
        float mx = sc[0];
        for (int i = 1; i < n; i++) if (sc[i] > mx) mx = sc[i];
        float sum = 0.f;
        for (int i = 0; i < n; i++) { sc[i] = expf(sc[i] - mx); sum += sc[i]; }
        for (int i = 0; i < n; i++) sc[i] /= sum;
        float * oh = buf->o + h * hd;
        llm_attn_apply_hd(sc, vc, use_rows ? rows : NULL, (int) krow, n, hd, oh);
    }
    llm_mm(l, buf->o, 1, qd, blk->attn_output, out, embd);
}

// Gated DeltaNet (linear attention) block.
// Recurrence per value head (matches ggml gated_delta_net, non-KDA):
//   g  = exp(a * softplus(alpha + dt))          (state decay, per head)
//   S  *= g
//   u[j]  = sum_i S[i][j] * k[i]                (r = S^T k)
//   d[j]  = beta * (v[j] - u[j])
//   S[i][j] += k[i] * d[j]
//   o[j]  = sum_i S[i][j] * q[i] * (1/sqrt(hd))
// q/k/v = per-head l2-normalized slices of the conv output; conv = silu(conv1d).
static void llm_forward_gdn(llm_t * l, int b, llm_block_t * blk, const float * x, float * out, int pos, llm_bufs_t * buf) {
    (void) pos;
    int embd = l->n_embd;
    int hd = l->ssm_head_dim;
    int nk = l->ssm_num_k_heads, nv = l->ssm_num_v_heads;
    int key_dim = nk * hd, value_dim = nv * hd;
    int conv_dim = key_dim * 2 + value_dim;
    int K = l->ssm_conv_kernel;
    ggml_tensor * Ws[4] = { blk->attn_qkv, blk->attn_gate, blk->ssm_beta, blk->ssm_alpha };
    float * outs[4] = { buf->qkv, buf->z, buf->beta, buf->alpha };
    llm_mm_multi(l, x, embd, Ws, outs, 4);
    const float * W = (const float *) blk->ssm_conv1d->data;
    float * cs = l->gdn_conv[b];
    float * co = buf->conv_out;
    for (int c = 0; c < conv_dim; c++) {
        float acc = 0.f;
        for (int k = 0; k < K-1; k++) acc += W[k + c*K] * cs[k*conv_dim + c];
        acc += W[(K-1) + c*K] * buf->qkv[c];
        co[c] = acc / (1.0f + expf(-acc));
    }
    for (int k = 0; k < K-2; k++)
        memcpy(cs + k*conv_dim, cs + (k+1)*conv_dim, (size_t) conv_dim * sizeof(float));
    memcpy(cs + (K-2)*conv_dim, buf->qkv, (size_t) conv_dim * sizeof(float));
    float * q = buf->q, * k = buf->k, * v = buf->v;
    memcpy(q, co, (size_t) key_dim * sizeof(float));
    memcpy(k, co + key_dim, (size_t) key_dim * sizeof(float));
    memcpy(v, co + 2*key_dim, (size_t) value_dim * sizeof(float));
    for (int h = 0; h < nk; h++) llm_l2_norm(q + h*hd, hd, l->f_norm_rms_eps);
    for (int h = 0; h < nk; h++) llm_l2_norm(k + h*hd, hd, l->f_norm_rms_eps);
    float * S = l->gdn_state[b];
    const float * a = (const float *) blk->ssm_a->data;
    const float * dtb = (const float *) blk->ssm_dt->data;
    float scale = 1.0f / sqrtf((float) hd);
    for (int h = 0; h < nv; h++) {
        float beta = 1.0f / (1.0f + expf(-buf->beta[h]));
        float sp = buf->alpha[h] + dtb[h];
        float softplus = (sp > 20.f) ? sp : log1pf(expf(sp));
        float g = expf(softplus * a[h]);
        float * Sh = S + (size_t) h * hd * hd;
        // the model has num_v_heads value heads but only num_k_heads query/key
        // heads; the q/k are repeated (tiled) across the v heads, so v-head h
        // uses q/k head (h % nk)
        const float * kh = k + (size_t) (h % nk) * hd;
        const float * vh = v + (size_t) h * hd;
        const float * qh = q + (size_t) (h % nk) * hd;
        float * oh = buf->o + h*hd;
        for (int i = 0; i < hd*hd; i++) Sh[i] *= g;
        // u[j] = (S^T k)[j];  S[i][j] += k[i]*d[j]
        for (int j = 0; j < hd; j++) {
            float uj = 0.f;
            for (int i = 0; i < hd; i++) uj += Sh[i*hd + j] * kh[i];
            float dj = beta * (vh[j] - uj);
            for (int i = 0; i < hd; i++) Sh[i*hd + j] += kh[i] * dj;
        }
        // o[j] = (S^T q)[j]
        for (int j = 0; j < hd; j++) {
            float acc = 0.f;
            for (int i = 0; i < hd; i++) acc += Sh[i*hd + j] * qh[i];
            oh[j] = acc * scale;
        }
    }
    const float * nw = (const float *) blk->ssm_norm->data;
    for (int h = 0; h < nv; h++) {
        float * oh = buf->o + h*hd;
        llm_rms_norm(oh, nw, hd, l->f_norm_rms_eps, oh);
        const float * zh = buf->z + h*hd;
        for (int i = 0; i < hd; i++) oh[i] *= zh[i] / (1.0f + expf(-zh[i]));
    }
    llm_mm(l, buf->o, 1, value_dim, blk->ssm_out, out, embd);
}

// main model forward for one token at position pos.
// sets l->logits (next-token logits).
static void llm_forward(llm_t * l, int token_id, int pos) {
    int embd = l->n_embd;
    llm_bufs_t buf = llm_bufs(l, 1);
    llm_get_embd(l, token_id, buf.x);
    for (int b = 0; b < l->n_layer_main; b++) {
        llm_block_t * blk = &l->blocks[b];
        llm_rms_norm(buf.x, (const float *) blk->attn_norm->data, embd, l->f_norm_rms_eps, buf.r);
        if (blk->is_full_attn) llm_forward_full(l, b, blk, buf.r, buf.out, pos, &buf);
        else llm_forward_gdn(l, b, blk, buf.r, buf.out, pos, &buf);
        for (int i = 0; i < embd; i++) buf.x[i] += buf.out[i];
        llm_rms_norm(buf.x, (const float *) blk->post_attention_norm->data, embd, l->f_norm_rms_eps, buf.r);
        llm_forward_ffn(l, blk, buf.r, buf.out, &buf);
        for (int i = 0; i < embd; i++) buf.x[i] += buf.out[i];
    }
    llm_rms_norm(buf.x, (const float *) l->w_output_norm->data, embd, l->f_norm_rms_eps, buf.x);
    llm_mm(l, buf.x, 1, embd, l->w_output, l->logits, l->n_logits);
}

static void llm_forward_gemma(llm_t * l, int token_id, int pos) {
    int embd = l->n_embd;
    llm_bufs_t buf = llm_bufs(l, 1);
    llm_get_embd(l, token_id, buf.x);
    if (l->f_embd_scale != 1.0f) {
        float s = l->f_embd_scale;
        for (int i = 0; i < embd; i++) buf.x[i] *= s;
    }
    for (int b = 0; b < l->n_layer_main; b++) {
        llm_block_t * blk = &l->blocks[b];
        llm_rms_norm(buf.x, (const float *) blk->attn_norm->data, embd, l->f_norm_rms_eps, buf.r);
        llm_forward_full_gemma(l, b, blk, buf.r, buf.out, pos, &buf);
        llm_rms_norm(buf.out, (const float *) blk->post_attention_norm->data, embd, l->f_norm_rms_eps, buf.out);
        for (int i = 0; i < embd; i++) buf.x[i] += buf.out[i];
        llm_rms_norm(buf.x, (const float *) blk->ffn_norm->data, embd, l->f_norm_rms_eps, buf.r);
        llm_forward_ffn_gemma(l, blk, buf.r, buf.out, &buf);
        llm_rms_norm(buf.out, (const float *) blk->post_ffw_norm->data, embd, l->f_norm_rms_eps, buf.out);
        for (int i = 0; i < embd; i++) buf.x[i] += buf.out[i];
    }
    llm_rms_norm(buf.x, (const float *) l->w_output_norm->data, embd, l->f_norm_rms_eps, buf.x);
    llm_mm(l, buf.x, 1, embd, l->w_output, l->logits, l->n_logits);
}

// LFM2 shortconv module (per token, causal): a gated short convolution.
//   proj = in_proj @ x                 [3*n_embd]
//   b, c, val = split(proj)            each [n_embd]
//   g = b * val                        [n_embd]
//   conv_out[c] = sum_k conv_w[k,c]*x_tap[k,c]  (causal depthwise, state+current)
//   y = c * conv_out
//   out = out_proj @ y
// recurrent state = last (l_cache-1) gated tokens, shifted by g each step
static void llm_forward_shortconv_lfm2(llm_t * l, int b, llm_block_t * blk, const float * x, float * out, llm_bufs_t * buf) {
    int embd = l->n_embd;
    llm_mm(l, x, 1, embd, blk->shortconv_in_proj, buf->sproj, 3 * embd);
    const float * bb = buf->sproj;
    const float * cc = buf->sproj + embd;
    const float * val = buf->sproj + 2 * embd;
    float * g = buf->sgated;
    for (int i = 0; i < embd; i++) g[i] = bb[i] * val[i];
    const int lc = l->shortconv_l_cache;
    const int d_conv = lc - 1;
    float * st = l->lfm2_state[b];
    const float * convw = (const float *) blk->shortconv_conv->data;
    float * co = buf->sconv;
    for (int c = 0; c < embd; c++) {
        float acc = 0.f;
        for (int k = 0; k < d_conv; k++) acc += convw[c * lc + k] * st[k * embd + c];
        acc += convw[c * lc + d_conv] * g[c];
        co[c] = acc;
    }
    if (d_conv > 1) memmove(st, st + embd, (size_t) (d_conv - 1) * embd * sizeof(float));
    memcpy(st + (d_conv - 1) * embd, g, (size_t) embd * sizeof(float));
    float * y = buf->o;
    for (int i = 0; i < embd; i++) y[i] = cc[i] * co[i];
    llm_mm(l, y, 1, embd, blk->shortconv_out_proj, out, embd);
}

// LFM2 dense forward: RMSNorm -> (attention | shortconv) -> +residual ->
// RMSNorm -> SwiGLU FFN -> +residual. Attention layers reuse the gemma3 full
// attention (q/k norm + NeoX RoPE, full window); shortconv layers run the
// short-convolution module with their recurrent state.
static void llm_forward_lfm2(llm_t * l, int token_id, int pos) {
    int embd = l->n_embd;
    llm_bufs_t buf = llm_bufs(l, 1);
    llm_get_embd(l, token_id, buf.x);
    for (int b = 0; b < l->n_layer_main; b++) {
        llm_block_t * blk = &l->blocks[b];
        llm_rms_norm(buf.x, (const float *) blk->attn_norm->data, embd, l->f_norm_rms_eps, buf.r);
        if (blk->is_full_attn) llm_forward_full_gemma(l, b, blk, buf.r, buf.out, pos, &buf);
        else llm_forward_shortconv_lfm2(l, b, blk, buf.r, buf.out, &buf);
        for (int i = 0; i < embd; i++) buf.x[i] += buf.out[i];
        llm_rms_norm(buf.x, (const float *) blk->ffn_norm->data, embd, l->f_norm_rms_eps, buf.r);
        llm_forward_ffn(l, blk, buf.r, buf.out, &buf);
        for (int i = 0; i < embd; i++) buf.x[i] += buf.out[i];
    }
    llm_rms_norm(buf.x, (const float *) l->w_output_norm->data, embd, l->f_norm_rms_eps, buf.x);
    llm_mm(l, buf.x, 1, embd, l->w_output, l->logits, l->n_logits);
}

// batched prompt forward (lfm2): process `n` tokens at positions
// [start_pos, start_pos+n) in one pass. Attention layers and the FFN are
// batched; shortconv layers stay per-token (their conv state is recurrent).
// Only the last token's logits are produced (next-token prediction).
static void llm_forward_batch_lfm2(llm_t * l, const int * tokens, int n, int start_pos) {
    const int embd = l->n_embd, hd = l->head_dim, nh = l->n_head, nhkv = l->n_head_kv;
    const int qd = nh * hd, kvd = nhkv * hd, ff = l->n_ff;
    const int B = n;
    const size_t krow = (size_t) nhkv * hd;
    const int g = nh / nhkv;
    const float scale = 1.0f / sqrtf((float) hd);
    float * x = l->bx, * r = l->br, * out = l->bout, * o = l->bo;
    for (int t = 0; t < B; t++) llm_get_embd(l, tokens[t], x + (size_t) t * embd);
    for (int b = 0; b < l->n_layer_main; b++) {
        llm_block_t * blk = &l->blocks[b];
        for (int t = 0; t < B; t++)
            llm_rms_norm(x + (size_t) t * embd, (const float *) blk->attn_norm->data, embd, l->f_norm_rms_eps, r + (size_t) t * embd);
        if (blk->is_full_attn) {
            // batched attention (same as gemma3 full attention, full window)
            llm_mm_batch_cpu(l, blk->attn_q, r, embd, B, l->bq, qd);
            llm_mm_batch_cpu(l, blk->attn_k, r, embd, B, l->bk, kvd);
            llm_mm_batch_cpu(l, blk->attn_v, r, embd, B, l->bv, kvd);
            const float * qw = (const float *) blk->attn_q_norm->data;
            const float * kw = (const float *) blk->attn_k_norm->data;
            for (int t = 0; t < B; t++) {
                const int pos = start_pos + t;
                float * q_t = l->bq + (size_t) t * qd;
                float * k_t = l->bk + (size_t) t * kvd;
                float * v_t = l->bv + (size_t) t * kvd;
                for (int h = 0; h < nh; h++) {
                    float * qh = q_t + h * hd;
                    llm_rms_norm(qh, qw, hd, l->f_norm_rms_eps, qh);
                    llm_rope_tables(l->rope_cos, l->rope_sin, qh, pos, hd);
                }
                for (int hv = 0; hv < nhkv; hv++) {
                    float * kh = k_t + hv * hd;
                    llm_rms_norm(kh, kw, hd, l->f_norm_rms_eps, kh);
                    llm_rope_tables(l->rope_cos, l->rope_sin, kh, pos, hd);
                }
                memcpy(l->k_cache[b] + (size_t) pos * krow, k_t, kvd * sizeof(float));
                memcpy(l->v_cache[b] + (size_t) pos * krow, v_t, kvd * sizeof(float));
                float * o_t = o + (size_t) t * embd;
                for (int h = 0; h < nh; h++) {
                    const float * qh = q_t + h * hd;
                    const float * kc = l->k_cache[b] + (h / g) * hd;
                    const float * vc = l->v_cache[b] + (h / g) * hd;
                    float * sc = l->bscores;
                    for (int j = 0; j <= pos; j++)
                        sc[j] = llm_dot_hd(qh, kc + (size_t) j * krow, hd) * scale;
                    float mx = sc[0];
                    for (int j = 1; j <= pos; j++) if (sc[j] > mx) mx = sc[j];
                    float sum = 0.f;
                    for (int j = 0; j <= pos; j++) { sc[j] = expf(sc[j] - mx); sum += sc[j]; }
                    for (int j = 0; j <= pos; j++) sc[j] /= sum;
                    float * oh = o_t + h * hd;
                    llm_attn_apply_hd(sc, vc, NULL, (int) krow, pos + 1, hd, oh);
                }
            }
            llm_mm_batch_cpu(l, blk->attn_output, o, embd, B, out, embd);
        } else {
            // shortconv: sequential over tokens (recurrent conv state); each
            // token uses its own per-token scratch via llm_bufs
            for (int t = 0; t < B; t++) {
                llm_bufs_t sb = llm_bufs(l, 1);
                llm_forward_shortconv_lfm2(l, b, blk, r + (size_t) t * embd, out + (size_t) t * embd, &sb);
            }
        }
        for (int t = 0; t < B; t++) {
            float * xt = x + (size_t) t * embd, * ot = out + (size_t) t * embd;
            for (int i = 0; i < embd; i++) xt[i] += ot[i];
            llm_rms_norm(xt, (const float *) blk->ffn_norm->data, embd, l->f_norm_rms_eps, r + (size_t) t * embd);
        }
        llm_mm_batch_cpu(l, blk->ffn_gate, r, embd, B, l->bfg, ff);
        llm_mm_batch_cpu(l, blk->ffn_up, r, embd, B, l->bfu, ff);
        for (int t = 0; t < B; t++) {
            float * gt = l->bfg + (size_t) t * ff, * ut = l->bfu + (size_t) t * ff;
            for (int i = 0; i < ff; i++) gt[i] = gt[i] / (1.0f + expf(-gt[i])) * ut[i];
        }
        llm_mm_batch_cpu(l, blk->ffn_down, l->bfg, ff, B, out, embd);
        for (int t = 0; t < B; t++) {
            float * xt = x + (size_t) t * embd, * ot = out + (size_t) t * embd;
            for (int i = 0; i < embd; i++) xt[i] += ot[i];
        }
    }
    int last = B - 1;
    llm_rms_norm(x + (size_t) last * embd, (const float *) l->w_output_norm->data, embd, l->f_norm_rms_eps, r + (size_t) last * embd);
    llm_mm(l, r + (size_t) last * embd, 1, embd, l->w_output, l->logits, l->n_logits);
}

// batched prompt forward (gemma3): same idea, with q/k norm, SWA and
// post-attention / post-FFN norms.
static void llm_forward_batch_gemma(llm_t * l, const int * tokens, int n, int start_pos) {
    const int embd = l->n_embd, hd = l->head_dim, nh = l->n_head, nhkv = l->n_head_kv;
    const int qd = nh * hd, kvd = nhkv * hd, ff = l->n_ff;
    const int B = n;
    const size_t krow = (size_t) nhkv * hd;
    const int g = nh / nhkv;
    const float scale = 1.0f / sqrtf((float) hd);
    float * x = l->bx, * r = l->br, * out = l->bout, * o = l->bo;
    for (int t = 0; t < B; t++) {
        llm_get_embd(l, tokens[t], x + (size_t) t * embd);
        if (l->f_embd_scale != 1.0f) {
            float s = l->f_embd_scale;
            float * xt = x + (size_t) t * embd;
            for (int i = 0; i < embd; i++) xt[i] *= s;
        }
    }
    for (int b = 0; b < l->n_layer_main; b++) {
        llm_block_t * blk = &l->blocks[b];
        int win = blk->win > 0 ? blk->win : l->n_ctx;
        const float * cos_t = blk->is_swa ? l->rope_cos_swa : l->rope_cos;
        const float * sin_t = blk->is_swa ? l->rope_sin_swa : l->rope_sin;
        const float * qw = (const float *) blk->attn_q_norm->data;
        const float * kw = (const float *) blk->attn_k_norm->data;
        for (int t = 0; t < B; t++)
            llm_rms_norm(x + (size_t) t * embd, (const float *) blk->attn_norm->data, embd, l->f_norm_rms_eps, r + (size_t) t * embd);
        llm_mm_batch_cpu(l, blk->attn_q, r, embd, B, l->bq, qd);
        llm_mm_batch_cpu(l, blk->attn_k, r, embd, B, l->bk, kvd);
        llm_mm_batch_cpu(l, blk->attn_v, r, embd, B, l->bv, kvd);
        for (int t = 0; t < B; t++) {
            const int pos = start_pos + t;
            float * q_t = l->bq + (size_t) t * qd;
            float * k_t = l->bk + (size_t) t * kvd;
            float * v_t = l->bv + (size_t) t * kvd;
            for (int h = 0; h < nh; h++) {
                float * qh = q_t + h * hd;
                llm_rms_norm(qh, qw, hd, l->f_norm_rms_eps, qh);
                llm_rope_tables(cos_t, sin_t, qh, pos, hd);
            }
            for (int hv = 0; hv < nhkv; hv++) {
                float * kh = k_t + hv * hd;
                llm_rms_norm(kh, kw, hd, l->f_norm_rms_eps, kh);
                llm_rope_tables(cos_t, sin_t, kh, pos, hd);
            }
            int slot = pos % win;
            memcpy(l->k_cache[b] + (size_t) slot * krow, k_t, kvd * sizeof(float));
            memcpy(l->v_cache[b] + (size_t) slot * krow, v_t, kvd * sizeof(float));
            int start = pos + 1 > win ? pos - win + 1 : 0;
            float * o_t = o + (size_t) t * embd;
            for (int h = 0; h < nh; h++) {
                const float * qh = q_t + h * hd;
                const float * kc = l->k_cache[b] + (h / g) * hd;
                const float * vc = l->v_cache[b] + (h / g) * hd;
                float * sc = l->bscores;
                int nc = 0;
                for (int j = start; j <= pos; j++)
                    sc[nc++] = llm_dot_hd(qh, kc + (size_t) (j % win) * krow, hd) * scale;
                float mx = sc[0];
                for (int i = 1; i < nc; i++) if (sc[i] > mx) mx = sc[i];
                float sum = 0.f;
                for (int i = 0; i < nc; i++) { sc[i] = expf(sc[i] - mx); sum += sc[i]; }
                for (int i = 0; i < nc; i++) sc[i] /= sum;
                float * oh = o_t + h * hd;
                if (win > 0 && start > 0) {
                    int rows[512];
                    int nw = nc <= 512 ? nc : 0;
                    for (int i = 0; i < nw; i++) rows[i] = (start + i) % win;
                    llm_attn_apply_hd(sc, vc, nw ? rows : NULL, (int) krow, nc, hd, oh);
                } else {
                    llm_attn_apply_hd(sc, vc, NULL, (int) krow, nc, hd, oh);
                }
            }
        }
        llm_mm_batch_cpu(l, blk->attn_output, o, embd, B, out, embd);
        for (int t = 0; t < B; t++) {
            float * xt = x + (size_t) t * embd, * ot = out + (size_t) t * embd;
            llm_rms_norm(ot, (const float *) blk->post_attention_norm->data, embd, l->f_norm_rms_eps, ot);
            for (int i = 0; i < embd; i++) xt[i] += ot[i];
            llm_rms_norm(xt, (const float *) blk->ffn_norm->data, embd, l->f_norm_rms_eps, r + (size_t) t * embd);
        }
        llm_mm_batch_cpu(l, blk->ffn_gate, r, embd, B, l->bfg, ff);
        llm_mm_batch_cpu(l, blk->ffn_up, r, embd, B, l->bfu, ff);
        for (int t = 0; t < B; t++) {
            float * gt = l->bfg + (size_t) t * ff, * ut = l->bfu + (size_t) t * ff;
            for (int i = 0; i < ff; i++) gt[i] = gemma_gelu_tanh(gt[i]) * ut[i];
        }
        llm_mm_batch_cpu(l, blk->ffn_down, l->bfg, ff, B, out, embd);
        for (int t = 0; t < B; t++) {
            float * xt = x + (size_t) t * embd, * ot = out + (size_t) t * embd;
            llm_rms_norm(ot, (const float *) blk->post_ffw_norm->data, embd, l->f_norm_rms_eps, ot);
            for (int i = 0; i < embd; i++) xt[i] += ot[i];
        }
    }
    int last = B - 1;
    llm_rms_norm(x + (size_t) last * embd, (const float *) l->w_output_norm->data, embd, l->f_norm_rms_eps, r + (size_t) last * embd);
    llm_mm(l, r + (size_t) last * embd, 1, embd, l->w_output, l->logits, l->n_logits);
}

// batched prompt forward (qwen35 hybrid): process `n` tokens at positions
// [start_pos, start_pos+n) in one pass. Matmuls are batched; the recurrent
// GDN state / conv / full-attention stay per token. Reuses llm_bufs(B).
static void llm_forward_batch_qwen35(llm_t * l, const int * tokens, int n, int start_pos) {
    const int embd = l->n_embd, hd = l->head_dim, nh = l->n_head, nhkv = l->n_head_kv;
    const int qfd = nh * hd * 2, kvd = nhkv * hd;
    const int B = n;
    int o_dim = l->n_embd_eff;
    if (nh * hd > o_dim) o_dim = nh * hd;
    if (l->ssm_num_v_heads * l->ssm_head_dim > o_dim) o_dim = l->ssm_num_v_heads * l->ssm_head_dim;
    llm_bufs_t buf = llm_bufs(l, B);
    for (int t = 0; t < B; t++) llm_get_embd(l, tokens[t], buf.x + (size_t) t * embd);
    for (int b = 0; b < l->n_layer_main; b++) {
        llm_block_t * blk = &l->blocks[b];
        for (int t = 0; t < B; t++)
            llm_rms_norm(buf.x + (size_t) t * embd, (const float *) blk->attn_norm->data, embd, l->f_norm_rms_eps, buf.r + (size_t) t * embd);
        if (blk->is_full_attn) {
            llm_mm_batch_cpu(l, blk->attn_q, buf.r, embd, B, buf.q_full, qfd);
            llm_mm_batch_cpu(l, blk->attn_k, buf.r, embd, B, buf.k_cur, kvd);
            llm_mm_batch_cpu(l, blk->attn_v, buf.r, embd, B, buf.v_cur, kvd);
            const float * qw = (const float *) blk->attn_q_norm->data;
            const float * kw = (const float *) blk->attn_k_norm->data;
            const size_t krow = (size_t) nhkv * hd;
            const int g = nh / nhkv;
            const float scale = 1.0f / sqrtf((float) hd);
            for (int t = 0; t < B; t++) {
                const int pos = start_pos + t;
                float * q_t = buf.q_full + (size_t) t * qfd;
                float * k_t = buf.k_cur + (size_t) t * kvd;
                float * v_t = buf.v_cur + (size_t) t * kvd;
                for (int h = 0; h < nh; h++) {
                    float * qh = q_t + h * 2 * hd;
                    llm_rms_norm(qh, qw, hd, l->f_norm_rms_eps, qh);
                    llm_rope(l, qh, pos, hd);
                }
                for (int hv = 0; hv < nhkv; hv++) {
                    float * kh = k_t + hv * hd;
                    llm_rms_norm(kh, kw, hd, l->f_norm_rms_eps, kh);
                    llm_rope(l, kh, pos, hd);
                }
                memcpy(l->k_cache[b] + (size_t) pos * krow, k_t, kvd * sizeof(float));
                memcpy(l->v_cache[b] + (size_t) pos * krow, v_t, kvd * sizeof(float));
                float * o_t = buf.o + (size_t) t * o_dim;
                for (int h = 0; h < nh; h++) {
                    const float * qh = q_t + h * 2 * hd;
                    const float * gate = q_t + h * 2 * hd + hd;
                    const float * kc = l->k_cache[b] + (h / g) * hd;
                    const float * vc = l->v_cache[b] + (h / g) * hd;
                    float * sc = buf.scores;
                    for (int j = 0; j <= pos; j++)
                        sc[j] = llm_dot_hd(qh, kc + (size_t) j * krow, hd) * scale;
                    float mx = sc[0];
                    for (int j = 1; j <= pos; j++) if (sc[j] > mx) mx = sc[j];
                    float sum = 0.f;
                    for (int j = 0; j <= pos; j++) { sc[j] = expf(sc[j] - mx); sum += sc[j]; }
                    for (int j = 0; j <= pos; j++) sc[j] /= sum;
                    float * oh = o_t + h * hd;
                    llm_attn_apply_hd(sc, vc, NULL, (int) krow, pos + 1, hd, oh);
                    for (int d = 0; d < hd; d++) oh[d] /= (1.0f + expf(-gate[d]));
                }
            }
            llm_mm_batch_cpu(l, blk->attn_output, buf.o, nh * hd, B, buf.out, embd);
        } else {
            const int shd = l->ssm_head_dim, nk = l->ssm_num_k_heads, nv = l->ssm_num_v_heads;
            const int key_dim = nk * shd, value_dim = nv * shd, conv_dim = key_dim * 2 + value_dim;
            const int K = l->ssm_conv_kernel;
            llm_mm_batch_cpu(l, blk->attn_qkv, buf.r, embd, B, buf.qkv, conv_dim);
            llm_mm_batch_cpu(l, blk->attn_gate, buf.r, embd, B, buf.z, value_dim);
            llm_mm_batch_cpu(l, blk->ssm_beta, buf.r, embd, B, buf.beta, nv);
            llm_mm_batch_cpu(l, blk->ssm_alpha, buf.r, embd, B, buf.alpha, nv);
            const float * W = (const float *) blk->ssm_conv1d->data;
            const float * a = (const float *) blk->ssm_a->data;
            const float * dtb = (const float *) blk->ssm_dt->data;
            const float * nw = (const float *) blk->ssm_norm->data;
            float * cs = l->gdn_conv[b];
            float * S = l->gdn_state[b];
            const float scale = 1.0f / sqrtf((float) shd);
            for (int t = 0; t < B; t++) {
                const float * qkv_t = buf.qkv + (size_t) t * conv_dim;
                float * co = buf.conv_out + (size_t) t * conv_dim;
                for (int c = 0; c < conv_dim; c++) {
                    float acc = 0.f;
                    for (int k = 0; k < K-1; k++) acc += W[k + c*K] * cs[k*conv_dim + c];
                    acc += W[(K-1) + c*K] * qkv_t[c];
                    co[c] = acc / (1.0f + expf(-acc));
                }
                for (int k = 0; k < K-2; k++)
                    memcpy(cs + k*conv_dim, cs + (k+1)*conv_dim, (size_t) conv_dim * sizeof(float));
                memcpy(cs + (K-2)*conv_dim, qkv_t, (size_t) conv_dim * sizeof(float));
                float * q = buf.q + (size_t) t * key_dim;
                float * k = buf.k + (size_t) t * key_dim;
                float * v = buf.v + (size_t) t * value_dim;
                memcpy(q, co, (size_t) key_dim * sizeof(float));
                memcpy(k, co + key_dim, (size_t) key_dim * sizeof(float));
                memcpy(v, co + 2*key_dim, (size_t) value_dim * sizeof(float));
                for (int h = 0; h < nk; h++) llm_l2_norm(q + h*shd, shd, l->f_norm_rms_eps);
                for (int h = 0; h < nk; h++) llm_l2_norm(k + h*shd, shd, l->f_norm_rms_eps);
                const float * beta_t = buf.beta + (size_t) t * nv;
                const float * alpha_t = buf.alpha + (size_t) t * nv;
                const float * z_t = buf.z + (size_t) t * value_dim;
                float * o_t = buf.o + (size_t) t * o_dim;
                for (int h = 0; h < nv; h++) {
                    float beta = 1.0f / (1.0f + expf(-beta_t[h]));
                    float sp = alpha_t[h] + dtb[h];
                    float softplus = (sp > 20.f) ? sp : log1pf(expf(sp));
                    float g = expf(softplus * a[h]);
                    float * Sh = S + (size_t) h * shd * shd;
                    const float * kh = k + (size_t) (h % nk) * shd;
                    const float * vh = v + (size_t) h * shd;
                    const float * qh = q + (size_t) (h % nk) * shd;
                    float * oh = o_t + h * shd;
                    for (int i = 0; i < shd*shd; i++) Sh[i] *= g;
                    for (int j = 0; j < shd; j++) {
                        float uj = 0.f;
                        for (int i = 0; i < shd; i++) uj += Sh[i*shd + j] * kh[i];
                        float dj = beta * (vh[j] - uj);
                        for (int i = 0; i < shd; i++) Sh[i*shd + j] += kh[i] * dj;
                    }
                    for (int j = 0; j < shd; j++) {
                        float acc = 0.f;
                        for (int i = 0; i < shd; i++) acc += Sh[i*shd + j] * qh[i];
                        oh[j] = acc * scale;
                    }
                }
                for (int h = 0; h < nv; h++) {
                    float * oh = o_t + h * shd;
                    llm_rms_norm(oh, nw, shd, l->f_norm_rms_eps, oh);
                    const float * zh = z_t + h * shd;
                    for (int i = 0; i < shd; i++) oh[i] *= zh[i] / (1.0f + expf(-zh[i]));
                }
            }
            llm_mm_batch_cpu(l, blk->ssm_out, buf.o, value_dim, B, buf.out, embd);
        }
        for (int t = 0; t < B; t++) {
            float * xt = buf.x + (size_t) t * embd, * ot = buf.out + (size_t) t * embd;
            for (int i = 0; i < embd; i++) xt[i] += ot[i];
            llm_rms_norm(xt, (const float *) blk->post_attention_norm->data, embd, l->f_norm_rms_eps, buf.r + (size_t) t * embd);
        }
        llm_mm_batch_cpu(l, blk->ffn_gate, buf.r, embd, B, buf.ffn_g, l->n_ff);
        llm_mm_batch_cpu(l, blk->ffn_up, buf.r, embd, B, buf.ffn_u, l->n_ff);
        for (int t = 0; t < B; t++) {
            float * g = buf.ffn_g + (size_t) t * l->n_ff, * u = buf.ffn_u + (size_t) t * l->n_ff;
            for (int i = 0; i < l->n_ff; i++) g[i] = g[i] / (1.0f + expf(-g[i])) * u[i];
        }
        llm_mm_batch_cpu(l, blk->ffn_down, buf.ffn_g, l->n_ff, B, buf.out, embd);
        for (int t = 0; t < B; t++) {
            float * xt = buf.x + (size_t) t * embd, * ot = buf.out + (size_t) t * embd;
            for (int i = 0; i < embd; i++) xt[i] += ot[i];
        }
    }
    int last = B - 1;
    llm_rms_norm(buf.x + (size_t) last * embd, (const float *) l->w_output_norm->data, embd, l->f_norm_rms_eps, buf.r + (size_t) last * embd);
    llm_mm(l, buf.r + (size_t) last * embd, 1, embd, l->w_output, l->logits, l->n_logits);
}

// ---------------------------------------------------------------------------
// model load
// ---------------------------------------------------------------------------
static ggml_tensor * gguf_get(gguf_file_t * f, const char * name) {
    for (int i = 0; i < f->n_tensors; i++)
        if (strcmp(f->tensors[i].name, name) == 0) return f->tensors[i].gt;
    return NULL;
}

static int gguf_has_tensor(gguf_file_t * f, const char * name) {
    for (int i = 0; i < f->n_tensors; i++)
        if (strcmp(f->tensors[i].name, name) == 0) return 1;
    return 0;
}

static int gguf_load(gguf_file_t * f, const char * path) {
    size_t sz;
    f->buf = (uint8_t *) mmap_file(path, &sz);
    if (!f->buf) return -1;
    f->buf_size = sz;
    rd_ctx_t r = { f->buf, sz, 0 };
    char magic[4];
    if (r.pos + 4 > r.len || memcmp(r.p, "GGUF", 4) != 0) { munmap_file(f->buf, sz); f->buf=NULL; return -1; }
    r.pos += 4;
    uint32_t version; rd_u32(&r, &version);
    uint64_t n_tensors, n_kv;
    rd_u64(&r, &n_tensors); rd_u64(&r, &n_kv);
    uint32_t alignment = 32;
    for (uint64_t i = 0; i < n_kv; i++) {
        char * key; size_t klen; if (rd_str(&r, &key, &klen)) return -1;
        uint32_t type; if (rd_u32(&r, &type)) { free(key); return -1; }
        uint64_t n = 1; int is_arr = 0;
        if (type == 9) { is_arr = 1; if (rd_u32(&r, &type)) { free(key); return -1; } if (rd_u64(&r, &n)) { free(key); return -1; } }
        if (strcmp(key, "general.alignment") == 0 && type == 4 && !is_arr) {
            if (rd_u32(&r, &alignment)) { free(key); return -1; }
        } else if (is_arr) { if (gguf_skip_val(&r, type, n)) { free(key); return -1; } }
        else { if (gguf_skip_val(&r, type, 1)) { free(key); return -1; } }
        free(key);
    }
    f->n_tensors = (int) n_tensors;
    f->tensors = (gguf_tensor_t *) calloc(n_tensors, sizeof(gguf_tensor_t));
    for (uint64_t i = 0; i < n_tensors; i++) {
        gguf_tensor_t * tt = &f->tensors[i];
        char * name; size_t nlen; if (rd_str(&r, &name, &nlen)) return -1;
        snprintf(tt->name, sizeof(tt->name), "%s", name); free(name);
        uint32_t nd; if (rd_u32(&r, &nd)) return -1;
        tt->n_dims = nd;
        for (uint32_t d = 0; d < nd && d < 4; d++) { uint64_t v; if (rd_u64(&r, &v)) return -1; tt->ne[d] = v; }
        for (uint32_t d = nd; d < 4; d++) tt->ne[d] = 1;
        uint32_t ttype; if (rd_u32(&r, &ttype)) return -1;
        tt->type = (enum ggml_type) ttype;
        uint64_t off; if (rd_u64(&r, &off)) return -1;
        tt->offset = off;
        size_t row = ggml_row_size(tt->type, tt->ne[0]);
        tt->nb[0] = (tt->ne[0] > 0) ? row / tt->ne[0] : ggml_type_size(tt->type);
        tt->nb[1] = row;
        tt->nb[2] = (tt->ne[1] > 0) ? row * tt->ne[1] : 0;
        tt->nb[3] = (tt->ne[2] > 0) ? tt->nb[2] * tt->ne[2] : 0;
        tt->size = (tt->ne[0] > 0) ? row * (size_t) tt->ne[1] * (size_t) tt->ne[2] * (size_t) tt->ne[3] : 0;
    }
    uint64_t data_start = (r.pos + alignment - 1) / alignment * alignment;
    for (uint64_t i = 0; i < n_tensors; i++) f->tensors[i].offset += data_start;
    return 0;
}

static ggml_tensor * gguf_make_tensor(ggml_context * ctx, gguf_file_t * f, gguf_tensor_t * tt) {
    int64_t ne[4];
    memcpy(ne, tt->ne, sizeof(ne));
    int blck = ggml_blck_size(tt->type);
    if (blck > 1 && ne[0] > 0) {
        int64_t padded = (ne[0] + blck - 1) / blck * blck;
        if (padded != ne[0]) ne[0] = padded;
    }
    ggml_tensor * t = ggml_new_tensor(ctx, tt->type, tt->n_dims, ne);
    ggml_set_name(t, tt->name);
    t->data = (void *) (f->buf + tt->offset);
    tt->gt = t;
    return t;
}

// metadata extraction (single pass over the KV section)
static void gguf_read_meta(gguf_file_t * f, llm_t * l) {
    rd_ctx_t r = { f->buf, f->buf_size, 0 };
    r.pos += 4; // magic
    uint32_t version; rd_u32(&r, &version);
    uint64_t n_tensors, n_kv; rd_u64(&r, &n_tensors); rd_u64(&r, &n_kv);
    l->tok.bos = l->tok.eos = l->tok.pad = -1;
    for (uint64_t i = 0; i < n_kv; i++) {
        char * key; size_t klen; if (rd_str(&r, &key, &klen)) break;
        uint32_t type; if (rd_u32(&r, &type)) { free(key); break; }
        uint64_t n = 1; int is_arr = 0;
        if (type == 9) { is_arr = 1; if (rd_u32(&r, &type)) { free(key); break; } if (rd_u64(&r, &n)) { free(key); break; } }
        if (is_arr) {
            if (strcmp(key, "lfm2.attention.head_count_kv") == 0 && type == 5) {
                int32_t maxkv = 0;
                for (int s = 0; s < (int) n; s++) { int32_t v; memcpy(&v, r.p + r.pos + s*4, 4); if (v > maxkv) maxkv = v; }
                l->n_head_kv = maxkv;   // all attention layers share this kv head count
            } else if (strcmp(key, "qwen35.rope.dimension_sections") == 0 && type == 5) {
                for (int s = 0; s < 4; s++) {
                    if (s < (int) n) { int32_t v; memcpy(&v, r.p + r.pos + s*4, 4); l->rope_sections[s] = v; }
                    else l->rope_sections[s] = 0;
                }
            } else if (strcmp(key, "tokenizer.ggml.tokens") == 0 && type == 8) {
                l->tok.n_vocab = (int) n;
                l->tok.tokens = (char **) calloc(n, sizeof(char *));
                size_t pos = r.pos;
                for (uint64_t s2 = 0; s2 < n; s2++) {
                    uint64_t slen; memcpy(&slen, r.p + pos, 8);
                    char * str = (char *) malloc(slen + 1); memcpy(str, r.p + pos + 8, slen); str[slen] = 0;
                    l->tok.tokens[s2] = str;
                    pos += 8 + slen;
                }
            } else if (strcmp(key, "tokenizer.ggml.token_type") == 0 && type == 5) {
                l->tok.token_type = (int *) malloc(sizeof(int) * n);
                for (uint64_t s2 = 0; s2 < n; s2++) { int32_t v; memcpy(&v, r.p + r.pos + s2*4, 4); l->tok.token_type[s2] = v; }
            } else if (strcmp(key, "tokenizer.ggml.scores") == 0 && type == 6) {
                l->tok.scores = (float *) malloc(sizeof(float) * n);
                for (uint64_t s2 = 0; s2 < n; s2++) { float v; memcpy(&v, r.p + r.pos + s2*4, 4); l->tok.scores[s2] = v; }
            } else if (strcmp(key, "tokenizer.ggml.merges") == 0 && type == 8) {
                l->tok.n_merges_raw = (int) n;
                l->tok.merges_raw = (char **) calloc(n, sizeof(char *));
                size_t pos = r.pos;
                for (uint64_t s2 = 0; s2 < n; s2++) {
                    uint64_t slen; memcpy(&slen, r.p + pos, 8);
                    char * str = (char *) malloc(slen + 1); memcpy(str, r.p + pos + 8, slen); str[slen] = 0;
                    l->tok.merges_raw[s2] = str;
                    pos += 8 + slen;
                }
            }
            if (gguf_skip_val(&r, type, n)) { free(key); break; }
        } else {
            if (strcmp(key, "tokenizer.chat_template") == 0 && type == 8) {
                char * s; size_t sl;
                if (rd_str(&r, &s, &sl) == 0) {
                    if (l->tok.chat_template) { free(l->tok.chat_template); l->tok.chat_template = NULL; }
                    l->tok.chat_template = s;
                }
            } else if (type == 8) {
                char * s; size_t sl;
                if (rd_str(&r, &s, &sl)) { free(key); break; }
                if (strcmp(key, "general.architecture") == 0) {
                    if (strcmp(s, "gemma3") == 0) l->arch = LLM_ARCH_GEMMA3;
                    else if (strcmp(s, "qwen35") == 0) l->arch = LLM_ARCH_QWEN35;
                    else if (strcmp(s, "lfm2") == 0) l->arch = LLM_ARCH_LFM2;
                    snprintf(l->arch_name, sizeof(l->arch_name), "%s", s);
                } else if (strcmp(key, "general.name") == 0) {
                    snprintf(l->arch_name, sizeof(l->arch_name), "%s", s);
                }
                free(s);
            } else {
                double v;
                if (gguf_get_num(&r, type, &v)) { free(key); break; }
                if (strcmp(key, "tokenizer.ggml.bos_token_id") == 0) l->tok.bos = (int) v;
                else if (strcmp(key, "tokenizer.ggml.eos_token_id") == 0) l->tok.eos = (int) v;
                else if (strcmp(key, "tokenizer.ggml.padding_token_id") == 0) l->tok.pad = (int) v;
                else if (strcmp(key, "tokenizer.ggml.unknown_token_id") == 0) l->tok.unk = (int) v;
                else if (strcmp(key, "tokenizer.ggml.add_bos_token") == 0) l->tok.add_bos = (int) v;
                else if (strcmp(key, "gemma3.context_length") == 0) l->n_ctx_model = (int) v;
                else if (strcmp(key, "gemma3.embedding_length") == 0) l->n_embd = (int) v;
                else if (strcmp(key, "gemma3.feed_forward_length") == 0) l->n_ff = (int) v;
                else if (strcmp(key, "gemma3.block_count") == 0) l->n_layer = (int) v;
                else if (strcmp(key, "gemma3.attention.head_count") == 0) l->n_head = (int) v;
                else if (strcmp(key, "gemma3.attention.head_count_kv") == 0) l->n_head_kv = (int) v;
                else if (strcmp(key, "gemma3.attention.key_length") == 0) l->head_dim = (int) v;
                else if (strcmp(key, "gemma3.attention.value_length") == 0) { (void)v; }
                else if (strcmp(key, "gemma3.attention.layer_norm_rms_epsilon") == 0) l->f_norm_rms_eps = (float) v;
                else if (strcmp(key, "gemma3.attention.sliding_window") == 0) l->sliding_window = (int) v;
                else if (strcmp(key, "gemma3.rope.freq_base") == 0) l->f_rope_freq_base = (float) v;
                else if (strcmp(key, "qwen35.context_length") == 0) l->n_ctx_model = (int) v;
                else if (strcmp(key, "qwen35.embedding_length") == 0) l->n_embd = (int) v;
                else if (strcmp(key, "qwen35.feed_forward_length") == 0) l->n_ff = (int) v;
                else if (strcmp(key, "qwen35.block_count") == 0) l->n_layer = (int) v;
                else if (strcmp(key, "qwen35.attention.head_count") == 0) l->n_head = (int) v;
                else if (strcmp(key, "qwen35.attention.head_count_kv") == 0) l->n_head_kv = (int) v;
                else if (strcmp(key, "qwen35.attention.key_length") == 0) l->head_dim = (int) v;
                else if (strcmp(key, "qwen35.attention.layer_norm_rms_epsilon") == 0) l->f_norm_rms_eps = (float) v;
                else if (strcmp(key, "qwen35.rope.freq_base") == 0) l->f_rope_freq_base = (float) v;
                else if (strcmp(key, "qwen35.rope.dimension_count") == 0) l->n_rope = (int) v;
                else if (strcmp(key, "qwen35.ssm.conv_kernel") == 0) l->ssm_conv_kernel = (int) v;
                else if (strcmp(key, "qwen35.ssm.state_size") == 0) l->ssm_state_size = (int) v;
                else if (strcmp(key, "qwen35.ssm.group_count") == 0) l->ssm_num_k_heads = (int) v;
                else if (strcmp(key, "qwen35.ssm.time_step_rank") == 0) l->ssm_num_v_heads = (int) v;
                else if (strcmp(key, "qwen35.ssm.inner_size") == 0) l->ssm_inner_size = (int) v;
                else if (strcmp(key, "qwen35.full_attention_interval") == 0) l->full_attn_interval = (int) v;
                else if (strcmp(key, "lfm2.context_length") == 0) l->n_ctx_model = (int) v;
                else if (strcmp(key, "lfm2.embedding_length") == 0) l->n_embd = (int) v;
                else if (strcmp(key, "lfm2.feed_forward_length") == 0) l->n_ff = (int) v;
                else if (strcmp(key, "lfm2.block_count") == 0) l->n_layer = (int) v;
                else if (strcmp(key, "lfm2.attention.head_count") == 0) l->n_head = (int) v;
                else if (strcmp(key, "lfm2.attention.key_length") == 0) l->head_dim = (int) v;
                else if (strcmp(key, "lfm2.attention.layer_norm_rms_epsilon") == 0) l->f_norm_rms_eps = (float) v;
                else if (strcmp(key, "lfm2.rope.freq_base") == 0) l->f_rope_freq_base = (float) v;
                else if (strcmp(key, "lfm2.rope.dimension_count") == 0) l->n_rope = (int) v;
                else if (strcmp(key, "lfm2.vocab_size") == 0) { (void) v; }
                else if (strcmp(key, "lfm2.shortconv.l_cache") == 0) l->shortconv_l_cache = (int) v;
            }
        }
        free(key);
    }
}

// ---------------------------------------------------------------------------
// tokenizer public API
// ---------------------------------------------------------------------------
int llm_vocab_size(const llm_t * l) { return l->tok.n_vocab; }

llm_arch_t llm_get_arch(const llm_t * l) { return l->arch; }

// ---------------------------------------------------------------------------
// SentencePiece BPE (merge-by-score), matching llama.cpp llm_tokenizer_spm.
// Greedy longest-match is NOT equivalent to SentencePiece BPE: it yields a
// different token sequence than the model was trained on, which corrupts the
// hidden states and garbles output (missing letters / CJK). The merge order is
// highest token score first, ties broken by the smaller left index.
// ---------------------------------------------------------------------------
typedef struct {
    const char * text;   // pointer into the escaped buffer
    int n;               // byte length (grows on merge)
    int prev, next;      // symbol indices (-1 = none)
} spm_sym_t;

typedef struct {
    int left, right;
    float score;
    int size;            // byte length of the bigram at creation
} spm_bg_t;

typedef struct { spm_bg_t * a; int n, cap; } spm_heap_t;

static int spm_bg_hi(const spm_bg_t * x, const spm_bg_t * y) {
    if (x->score != y->score) return x->score > y->score;
    return x->left < y->left;
}
static void spm_heap_push(spm_heap_t * h, spm_bg_t b) {
    if (h->n >= h->cap) { h->cap = h->cap ? h->cap * 2 : 64; h->a = (spm_bg_t *) realloc(h->a, (size_t) h->cap * sizeof(spm_bg_t)); }
    int i = h->n++;
    h->a[i] = b;
    while (i > 0) {
        int p = (i - 1) / 2;
        if (spm_bg_hi(&h->a[i], &h->a[p])) { spm_bg_t t = h->a[i]; h->a[i] = h->a[p]; h->a[p] = t; i = p; }
        else break;
    }
}
static spm_bg_t spm_heap_pop(spm_heap_t * h) {
    spm_bg_t top = h->a[0];
    h->a[0] = h->a[--h->n];
    int i = 0;
    for (;;) {
        int l = 2 * i + 1, r = 2 * i + 2, m = i;
        if (l < h->n && spm_bg_hi(&h->a[l], &h->a[m])) m = l;
        if (r < h->n && spm_bg_hi(&h->a[r], &h->a[m])) m = r;
        if (m == i) break;
        spm_bg_t t = h->a[i]; h->a[i] = h->a[m]; h->a[m] = t; i = m;
    }
    return top;
}
static void spm_try_add(const tok_t * t, spm_sym_t * syms, spm_heap_t * heap, int left, int right) {
    if (left < 0 || right < 0) return;
    spm_sym_t * L = &syms[left], * R = &syms[right];
    if (L->n == 0 || R->n == 0) return;
    int size = L->n + R->n;
    int id = tok_str_id_len(t, L->text, size);
    if (id < 0 || id >= t->n_vocab) return;
    spm_bg_t bg; bg.left = left; bg.right = right; bg.score = t->scores[id]; bg.size = size;
    spm_heap_push(heap, bg);
}
// BPE-merge a single raw fragment (no special tokens) into tokens
static int spm_encode_frag(const tok_t * t, const char * text, int len, int * out, int max) {
    if (len <= 0 || max <= 0) return 0;
    spm_sym_t * syms = (spm_sym_t *) malloc(sizeof(spm_sym_t) * (size_t)(len + 1));
    if (!syms) return 0;
    int nsym = 0, i = 0;
    while (i < len) {
        unsigned char c = (unsigned char) text[i];
        int n;
        if (c < 0x80) n = 1;
        else if ((c >> 5) == 6) n = 2;
        else if ((c >> 4) == 14) n = 3;
        else if ((c >> 3) == 30) n = 4;
        else n = 1;
        if (i + n > len) n = len - i;
        int idx = nsym;
        syms[idx].text = text + i;
        syms[idx].n = n;
        syms[idx].prev = idx - 1;
        syms[idx].next = -1;
        if (idx > 0) syms[idx - 1].next = idx;
        nsym++;
        i += n;
    }
    if (nsym == 0) { free(syms); return 0; }
    spm_heap_t heap; memset(&heap, 0, sizeof(heap));
    for (int k = 1; k < nsym; k++) spm_try_add(t, syms, &heap, k - 1, k);
    while (heap.n > 0) {
        spm_bg_t bg = spm_heap_pop(&heap);
        spm_sym_t * L = &syms[bg.left], * R = &syms[bg.right];
        if (L->n == 0 || R->n == 0 || L->n + R->n != bg.size) continue;
        L->n += R->n;
        R->n = 0;
        L->next = R->next;
        if (R->next >= 0) syms[R->next].prev = bg.left;
        spm_try_add(t, syms, &heap, L->prev, bg.left);
        spm_try_add(t, syms, &heap, bg.left, L->next);
    }
    int nout = 0;
    for (int s = 0; s != -1 && nout < max; s = syms[s].next) {
        spm_sym_t * S = &syms[s];
        if (S->n == 0) continue;
        int id = tok_str_id_len(t, S->text, S->n);
        if (id < 0) { if (t->unk >= 0) out[nout++] = t->unk; }
        else out[nout++] = id;
    }
    free(heap.a);
    free(syms);
    return nout;
}
// longest special token ("<...>") starting at s[pos]; returns token id or -1
static int spm_special_at(const tok_t * t, const char * s, int pos, int rem, int * out_len) {
    int best = -1, best_len = 0;
    for (int k = 0; k < t->n_special; k++) {
        const char * st = t->tokens[t->special_id[k]];
        int sl = (int) strlen(st);
        if (sl > rem || sl <= best_len) continue;
        if (memcmp(st, s + pos, (size_t) sl) == 0) { best = t->special_id[k]; best_len = sl; }
    }
    if (out_len) *out_len = best_len;
    return best;
}

int llm_tokenize(const llm_t * l, const char * text, int add_bos, int * tokens, int n_max) {
    const tok_t * t = &l->tok;
    int n = 0;
    if (add_bos && t->bos >= 0) tokens[n++] = t->bos;
    int len = (int) strlen(text);
    if (len == 0) return n;
    if (l->arch == LLM_ARCH_GEMMA3) {
        char * exp = (char *) malloc((size_t) len * 6 + 8);
        if (exp) {
            int elen = 0, i = 0;
            while (i < len && elen + 6 < len * 6 + 8) {
                unsigned char b = (unsigned char) text[i++];
                if (b == 0x20) {
                    exp[elen++] = (char) 0xE2;
                    exp[elen++] = (char) 0x96;
                    exp[elen++] = (char) 0x81;
                } else if (b >= 0x80 || b == 0x0A || (b >= 33 && b <= 126)) {
                    exp[elen++] = (char) b;
                } else {
                    uint16_t cp = byte_to_uni[b];
                    int cl = utf8_encode(cp, exp + elen);
                    elen += cl;
                }
            }
            exp[elen] = 0;
            if (t->scores) {
                // SentencePiece BPE (merge-by-score), matching llama.cpp:
                // partition on special tokens, BPE-merge each raw fragment.
                int pos = 0;
                while (pos < elen && n < n_max) {
                    int slen = 0;
                    int sid = spm_special_at(t, exp, pos, elen - pos, &slen);
                    if (sid >= 0) {
                        tokens[n++] = sid;
                        pos += slen;
                    } else {
                        int end = elen;
                        for (int q = pos + 1; q < elen; q++) {
                            if (exp[q] != '<') continue;
                            int tl = 0;
                            if (spm_special_at(t, exp, q, elen - q, &tl) >= 0) { end = q; break; }
                        }
                        n += spm_encode_frag(t, exp + pos, end - pos, tokens + n, n_max - n);
                        pos = end;
                    }
                }
                free(exp);
                return n;
            }
            // fallback: greedy longest-match (only if scores were not loaded)
            int pos = 0;
            while (pos < elen && n < n_max) {
                int rem = elen - pos;
                int maxl = rem < t->max_token_len ? rem : t->max_token_len;
                int id = -1;
                int consumed = 1;
                for (int L = maxl; L >= 1; L--) {
                    id = tok_str_id_len(t, exp + pos, L);
                    if (id >= 0) { consumed = L; break; }
                }
                if (id >= 0) {
                    tokens[n++] = id;
                    pos += consumed;
                } else {
                    if (t->unk >= 0) tokens[n++] = t->unk;
                    pos++;
                }
            }
            free(exp);
            return n;
        }
    }
    uint32_t * cps = (uint32_t *) malloc(sizeof(uint32_t) * (size_t)(len + 1));
    if (!cps) return n;
    int ncp = 0;
    for (int i = 0; i < len; i++) cps[ncp++] = byte_to_uni[(unsigned char) text[i]];
    // llama-bpe chunks digits into groups of 1..3; qwen35 pre-tokenizes each
    // digit on its own (\p{N}). Match the per-arch pre-tokenizer exactly.
    int digit_chunk = (l->arch == LLM_ARCH_LFM2) ? 3 : 1;
    int i = 0;
    while (i < ncp && n < n_max) {
        int sp = tok_special_match(t, cps, i, ncp);
        if (sp >= 0) {
            tokens[n++] = t->special_id[sp];
            i += t->special_len[sp];
            continue;
        }
        uint32_t c = cps[i];
        if (c == 0x120) { // space byte: becomes the Ġ prefix of the next word
            if (i+1 < ncp && is_word_cp(cps[i+1])) { i++; continue; }
            int id = (0x120 < 0x200) ? t->char2id[0x120] : -1;
            if (id >= 0 && n < n_max) tokens[n++] = id;
            i++;
        } else if (is_digit_cp(c)) {
            // a run of digits, chunked per the arch pre-tokenizer (1..3 llama, 1 qwen35)
            int j = i; while (j < ncp && is_digit_cp(cps[j])) j++;
            int pos = i;
            while (pos < j) {
                int chunk = j - pos; if (chunk > digit_chunk) chunk = digit_chunk;
                int has_lead = (pos > 0 && cps[pos-1] == 0x120);
                uint32_t * w = (uint32_t *) malloc(sizeof(uint32_t) * (size_t)(chunk + 1));
                int wn = 0;
                if (has_lead) w[wn++] = 0x120;
                memcpy(w + wn, cps + pos, sizeof(uint32_t) * (size_t) chunk);
                wn += chunk;
                n += bpe_encode_word(t, w, wn, tokens + n, n_max - n);
                free(w);
                pos += chunk;
            }
            i = j;
        } else if (is_letter_cp(c)) {
            // llama-bpe: a run of letters (stops at digits / punct / space)
            int j = i; while (j < ncp && is_letter_cp(cps[j])) j++;
            int cnt = j - i; if (cnt > 1000) cnt = 1000;
            int has_lead = (i > 0 && cps[i-1] == 0x120);
            uint32_t * w = (uint32_t *) malloc(sizeof(uint32_t) * (size_t)(cnt + 1));
            int wn = 0;
            if (has_lead) w[wn++] = 0x120;
            memcpy(w + wn, cps + i, sizeof(uint32_t) * (size_t) cnt);
            wn += cnt;
            n += bpe_encode_word(t, w, wn, tokens + n, n_max - n);
            free(w);
            i = j;
        } else {
            // punct / other. llama-bpe folds a single leading punct into the
            // following word ("[^\r\n\p{L}\p{N}]?[\p{L}]+"), so if the next char
            // is a letter (and this is not a newline) fold it into that word.
            if (!is_newline_cp(c) && i+1 < ncp && is_letter_cp(cps[i+1])) {
                int j = i + 1; while (j < ncp && is_letter_cp(cps[j])) j++;
                int cnt = j - i; if (cnt > 1000) cnt = 1000;
                int has_lead = (i > 0 && cps[i-1] == 0x120);
                uint32_t * w = (uint32_t *) malloc(sizeof(uint32_t) * (size_t)(cnt + 1));
                int wn = 0;
                if (has_lead) w[wn++] = 0x120;
                memcpy(w + wn, cps + i, sizeof(uint32_t) * (size_t) cnt);
                wn += cnt;
                n += bpe_encode_word(t, w, wn, tokens + n, n_max - n);
                free(w);
                i = j;
            } else {
                // stop the run as soon as a special token ("<...>") starts:
                // otherwise a special preceded by punctuation (e.g. ".<|eot_id|>")
                // would be swallowed by the BPE chunk and split into pieces.
                int j = i;
                while (j < ncp && !is_word_cp(cps[j]) && cps[j] != 0x120 &&
                       !(cps[j] == 0x3C && tok_special_match(t, cps, j, ncp) >= 0)) j++;
                int cnt = j - i; if (cnt > 1000) cnt = 1000;
                n += bpe_encode_word(t, cps + i, cnt, tokens + n, n_max - n);
                i = j;
            }
        }
    }
    free(cps);
    return n;
}

char * llm_detokenize(const llm_t * l, const int * tokens, int n) {
    const tok_t * t = &l->tok;
    size_t cap = 0, len = 0; char * out = NULL;
    int is_spm = (l->arch == LLM_ARCH_GEMMA3);
    for (int i = 0; i < n; i++) {
        int id = tokens[i];
        if (id < 0 || id >= t->n_vocab) continue;
        const char * s = t->tokens[id]; int sl = (int) strlen(s);
        if (is_spm) {
            // SentencePiece: token text is real UTF-8. Only U+2581 (E2 96 81)
            // marks a leading space; every other byte is copied verbatim. This
            // avoids truncating long tokens and avoids mis-mapping real Unicode
            // (e.g. U+0100..U+01FF) to raw bytes.
            int k = 0;
            while (k < sl) {
                if (k + 2 < sl && (unsigned char)s[k] == 0xE2 &&
                    (unsigned char)s[k+1] == 0x96 && (unsigned char)s[k+2] == 0x81) {
                    if (cap < len + 9) { cap = (len + 9) * 2; out = (char *) realloc(out, cap); }
                    out[len++] = 0x20; k += 3;
                } else {
                    if (cap < len + 9) { cap = (len + 9) * 2; out = (char *) realloc(out, cap); }
                    out[len++] = s[k]; k += 1;
                }
            }
            continue;
        }
        // byte-level BPE: decode to byte-mapped code points, map back to bytes
        uint32_t cps[128]; int nc = utf8_decode(s, sl, cps, 128);
        for (int j = 0; j < nc; j++) {
            uint32_t cp = cps[j];
            unsigned char ch[4]; int cl;
            if (cp >= 0x100 && cp < 0x200) { ch[0] = uni_to_byte[cp]; cl = 1; }
            else if (cp < 0x80 || (cp >= 0xA1 && cp <= 0xAC) || (cp >= 0xAE && cp <= 0xFF)) { ch[0] = (unsigned char) cp; cl = 1; }
            else { cl = utf8_encode(cp, (char *) ch); }
            if (cap < len + cl + 8) { cap = (len + cl + 8) * 2; out = (char *) realloc(out, cap); }
            memcpy(out + len, ch, cl); len += cl;
        }
    }
    if (cap < len + 1) { cap = len + 1; out = (char *) realloc(out, cap); }
    out[len] = 0; return out;
}

void llm_params_default(llm_params_t * p) {
    memset(p, 0, sizeof(*p));
    p->n_ctx = 4096;
    p->n_threads = 4;
    p->n_predict = 512;
    p->sampler.temp = 0.0f;
    p->sampler.top_k = 40;
    p->sampler.top_p = 0.9f;
    p->lang = NULL;
#ifdef GGML_USE_VULKAN
    p->use_gpu = 0;   // CPU by default
    p->gpu_index = 0;
#endif
}

static int llm_load_model(llm_t * l) {
    if (l->n_embd <= 0) l->n_embd = 4096;
    if (l->n_ff <= 0) l->n_ff = 12288;
    if (l->n_layer <= 0) l->n_layer = 33;
    if (l->n_head <= 0) l->n_head = 16;
    if (l->n_head_kv <= 0) l->n_head_kv = 4;
    if (l->head_dim <= 0) l->head_dim = 256;
    if (l->f_norm_rms_eps <= 0) l->f_norm_rms_eps = 1e-6f;
    if (l->f_rope_freq_base <= 0) l->f_rope_freq_base = 1e7f;
    if (l->n_rope <= 0) l->n_rope = l->head_dim;
    if (l->ssm_conv_kernel <= 0) l->ssm_conv_kernel = 4;
    if (l->ssm_state_size <= 0) l->ssm_state_size = 128;
    if (l->ssm_num_k_heads <= 0) l->ssm_num_k_heads = 16;
    if (l->ssm_num_v_heads <= 0) l->ssm_num_v_heads = 16;
    if (l->ssm_head_dim <= 0) l->ssm_head_dim = l->ssm_state_size;
    if (l->ssm_inner_size <= 0) l->ssm_inner_size = l->ssm_num_v_heads * l->ssm_head_dim;
    if (l->full_attn_interval <= 0) l->full_attn_interval = 4;
    if (l->arch == LLM_ARCH_GEMMA3) {
        if (l->sliding_window <= 0) l->sliding_window = 512;
        if (l->swa_period <= 0) l->swa_period = 6;
        if (l->f_rope_freq_swa <= 0) l->f_rope_freq_swa = 10000.0f;
        l->turn_end_id = tok_str_id(&l->tok, "<end_of_turn>");
        if (l->turn_end_id < 0) l->turn_end_id = 106;
    } else if (l->arch == LLM_ARCH_LFM2) {
        l->turn_end_id = tok_str_id(&l->tok, "<|im_end|>");
        if (l->turn_end_id < 0) l->turn_end_id = 0;
        if (l->shortconv_l_cache <= 0) l->shortconv_l_cache = 3;
    }
    // qwen35 hybrid models carry a trailing "nextn" (MTP) block used only for
    // speculative prediction; it is excluded from the main forward.
    int n_mtp = 0;
    for (int b = 0; b < l->n_layer; b++) {
        char nm[80];
        snprintf(nm, sizeof(nm), "blk.%d.nextn.eh_proj.weight", b);
        if (gguf_has_tensor(&l->file, nm)) n_mtp++;
    }
    l->n_layer_main = l->n_layer - n_mtp;
    if (l->n_layer_main <= 0) l->n_layer_main = l->n_layer;
    if (l->n_ctx_model > 0 && l->n_ctx > l->n_ctx_model) l->n_ctx = l->n_ctx_model;
    if (l->n_ctx < 8) l->n_ctx = 8;

    size_t wbuf = (size_t) l->file.n_tensors * ggml_tensor_overhead() + (size_t) (1 << 20);
    l->w_buf = malloc(wbuf);
    struct ggml_init_params wp = { .mem_size = wbuf, .mem_buffer = l->w_buf, .no_alloc = true };
    l->ctx_w = ggml_init(wp);

// Quantized weights are used as-is through the ggml mul_mat kernels (fast
// path); this covers Q4_K_M (Q4_K + Q6_K + F32) and the other common types
// (Q5_0/Q8_0/F16, as used by gemma3). Only types the ggml kernels cannot
// handle fall back to the F32 GEMV path. Set LLM_DBG_ALLF32=1 to force ALL
// quantized weights through the F32 GEMV path (debugging/fallback).
    const int all_f32 = getenv("LLM_DBG_ALLF32") != NULL;
    // helper: does this tensor stay quantized (used via the ggml mul_mat kernel)?
    #define LLM_STAYS_QUANTIZED(t) \
        ((t) == GGML_TYPE_F32 || \
         (!all_f32 && ((t) == GGML_TYPE_Q4_K || (t) == GGML_TYPE_Q6_K || \
                       (t) == GGML_TYPE_Q5_0 || (t) == GGML_TYPE_Q5_1 || \
                       (t) == GGML_TYPE_Q8_0 || (t) == GGML_TYPE_Q8_K || \
                       (t) == GGML_TYPE_Q2_K || (t) == GGML_TYPE_Q3_K || \
                       (t) == GGML_TYPE_Q5_K || (t) == GGML_TYPE_F16)))
    size_t f32_bytes = 0;
    for (int i = 0; i < l->file.n_tensors; i++) {
        gguf_tensor_t * tt = &l->file.tensors[i];
        if (LLM_STAYS_QUANTIZED(tt->type)) continue;
        int64_t rows = tt->ne[1]*tt->ne[2]*tt->ne[3];
        if (rows < 1) rows = 1;
        f32_bytes += (size_t) tt->ne[0] * rows * 4;
    }
    l->f32_w_buf = f32_bytes > 0 ? malloc(f32_bytes) : NULL;
    if (f32_bytes > 0 && !l->f32_w_buf) { LOG_ERROR("failed to alloc %zu MB for F32 weights\n", f32_bytes/1048576); return -1; }
    char * f32_cur = (char *) l->f32_w_buf;

    for (int i = 0; i < l->file.n_tensors; i++) {
        gguf_tensor_t * tt = &l->file.tensors[i];
        if (LLM_STAYS_QUANTIZED(tt->type)) {
            gguf_make_tensor(l->ctx_w, &l->file, tt);
        } else {
            int64_t ne[4]; memcpy(ne, tt->ne, sizeof(ne));
            ggml_tensor * t = ggml_new_tensor(l->ctx_w, GGML_TYPE_F32, tt->n_dims, ne);
            ggml_set_name(t, tt->name);
            t->data = f32_cur;
            int64_t rows = tt->ne[1]*tt->ne[2]*tt->ne[3];
            if (rows < 1) rows = 1;
            llm_dequant_tensor_to_f32(tt, (const char *) l->file.buf, (float *) f32_cur);
            f32_cur += (size_t) tt->ne[0] * rows * 4;
            tt->gt = t;
        }
    }

    l->w_token_embd = gguf_get(&l->file, "token_embd.weight");
    l->w_output = gguf_get(&l->file, "output.weight");
    if (!l->w_output) l->w_output = l->w_token_embd;
    l->w_output_norm = gguf_get(&l->file, "output_norm.weight");
    if (!l->w_output_norm) l->w_output_norm = gguf_get(&l->file, "token_embd_norm.weight");
    if (!l->w_token_embd || !l->w_output_norm) { LOG_ERROR("missing token_embd or output_norm\n"); return -1; }
    l->n_embd_eff = l->n_embd;
    if (l->w_token_embd && l->w_token_embd->ne[0] > l->n_embd_eff) l->n_embd_eff = (int) l->w_token_embd->ne[0];

    l->blocks = (llm_block_t *) calloc(l->n_layer, sizeof(llm_block_t));
    char nm[160];
    for (int b = 0; b < l->n_layer; b++) {
        llm_block_t * blk = &l->blocks[b];
        snprintf(nm, sizeof(nm), "blk.%d.attn_norm.weight", b);      blk->attn_norm = gguf_get(&l->file, nm);
        snprintf(nm, sizeof(nm), "blk.%d.post_attention_norm.weight", b); blk->post_attention_norm = gguf_get(&l->file, nm);
        snprintf(nm, sizeof(nm), "blk.%d.ffn_gate.weight", b);      blk->ffn_gate = gguf_get(&l->file, nm);
        snprintf(nm, sizeof(nm), "blk.%d.ffn_up.weight", b);        blk->ffn_up = gguf_get(&l->file, nm);
        snprintf(nm, sizeof(nm), "blk.%d.ffn_down.weight", b);      blk->ffn_down = gguf_get(&l->file, nm);
        snprintf(nm, sizeof(nm), "blk.%d.attn_q.weight", b);        blk->attn_q = gguf_get(&l->file, nm);
        snprintf(nm, sizeof(nm), "blk.%d.attn_k.weight", b);        blk->attn_k = gguf_get(&l->file, nm);
        snprintf(nm, sizeof(nm), "blk.%d.attn_v.weight", b);        blk->attn_v = gguf_get(&l->file, nm);
        snprintf(nm, sizeof(nm), "blk.%d.attn_output.weight", b);   blk->attn_output = gguf_get(&l->file, nm);
        snprintf(nm, sizeof(nm), "blk.%d.attn_q_norm.weight", b);   blk->attn_q_norm = gguf_get(&l->file, nm);
        snprintf(nm, sizeof(nm), "blk.%d.attn_k_norm.weight", b);   blk->attn_k_norm = gguf_get(&l->file, nm);
        snprintf(nm, sizeof(nm), "blk.%d.attn_qkv.weight", b);      blk->attn_qkv = gguf_get(&l->file, nm);
        snprintf(nm, sizeof(nm), "blk.%d.attn_gate.weight", b);     blk->attn_gate = gguf_get(&l->file, nm);
        snprintf(nm, sizeof(nm), "blk.%d.ssm_beta.weight", b);      blk->ssm_beta = gguf_get(&l->file, nm);
        snprintf(nm, sizeof(nm), "blk.%d.ssm_alpha.weight", b);     blk->ssm_alpha = gguf_get(&l->file, nm);
        snprintf(nm, sizeof(nm), "blk.%d.ssm_a", b);                blk->ssm_a = gguf_get(&l->file, nm);
        snprintf(nm, sizeof(nm), "blk.%d.ssm_dt.bias", b);          blk->ssm_dt = gguf_get(&l->file, nm);
        snprintf(nm, sizeof(nm), "blk.%d.ssm_conv1d.weight", b);    blk->ssm_conv1d = gguf_get(&l->file, nm);
        snprintf(nm, sizeof(nm), "blk.%d.ssm_norm.weight", b);      blk->ssm_norm = gguf_get(&l->file, nm);
        snprintf(nm, sizeof(nm), "blk.%d.ssm_out.weight", b);       blk->ssm_out = gguf_get(&l->file, nm);
        snprintf(nm, sizeof(nm), "blk.%d.ffn_norm.weight", b); blk->ffn_norm = gguf_get(&l->file, nm);
        snprintf(nm, sizeof(nm), "blk.%d.post_ffw_norm.weight", b); blk->post_ffw_norm = gguf_get(&l->file, nm);
        snprintf(nm, sizeof(nm), "blk.%d.shortconv.conv.weight", b);    blk->shortconv_conv = gguf_get(&l->file, nm);
        snprintf(nm, sizeof(nm), "blk.%d.shortconv.in_proj.weight", b); blk->shortconv_in_proj = gguf_get(&l->file, nm);
        snprintf(nm, sizeof(nm), "blk.%d.shortconv.out_proj.weight", b); blk->shortconv_out_proj = gguf_get(&l->file, nm);
        blk->is_full_attn = (blk->attn_q != NULL);
        blk->is_swa = 0;
        blk->win = l->n_ctx;
    }
    if (l->arch == LLM_ARCH_LFM2) {
        // head_dim is not always present in lfm2 metadata; derive it from the
        // q projection shape (ne[1] = n_head * head_dim). RoPE covers the full
        // head dim.
        for (int b = 0; b < l->n_layer; b++) {
            llm_block_t * blk = &l->blocks[b];
            if (blk->attn_q && blk->attn_q->ne[1] > 0 && l->n_head > 0) {
                l->head_dim = (int) (blk->attn_q->ne[1] / l->n_head);
                break;
            }
        }
        l->n_rope = l->head_dim;
    }
    if (l->arch == LLM_ARCH_GEMMA3) {
        for (int b = 0; b < l->n_layer; b++) {
            llm_block_t * blk = &l->blocks[b];
            blk->is_swa = (l->swa_period > 0 && (b + 1) % l->swa_period != 0);
            blk->win = blk->is_swa ? (l->sliding_window < l->n_ctx ? l->sliding_window : l->n_ctx) : l->n_ctx;
            if (!blk->ffn_norm || !blk->post_ffw_norm) {
                LOG_ERROR("gemma3 block %d missing ffn/post_ffw norm\n", b);
                return -1;
            }
        }
    }
    for (int b = 0; b < l->n_layer; b++) {
        llm_block_t * blk = &l->blocks[b];
        if (blk->attn_q && blk->attn_q->ne[0] > l->n_embd_eff) l->n_embd_eff = (int) blk->attn_q->ne[0];
        if (blk->attn_k && blk->attn_k->ne[0] > l->n_embd_eff) l->n_embd_eff = (int) blk->attn_k->ne[0];
        if (blk->attn_v && blk->attn_v->ne[0] > l->n_embd_eff) l->n_embd_eff = (int) blk->attn_v->ne[0];
        if (blk->ffn_gate && blk->ffn_gate->ne[0] > l->n_embd_eff) l->n_embd_eff = (int) blk->ffn_gate->ne[0];
        if (blk->ffn_up && blk->ffn_up->ne[0] > l->n_embd_eff) l->n_embd_eff = (int) blk->ffn_up->ne[0];
    }

    #ifdef GGML_USE_VULKAN
    // Vulkan GPU offload: copy quantized matmul weights to VRAM (when available).
    // F32 tensors (norms / SSM params) stay on CPU; the embedding reads its rows
    // back through the backend when token_embd lives on the GPU.
    if (l->vk) {
        int n_off = 0;
        for (int i = 0; i < l->file.n_tensors; i++) {
            ggml_tensor * t = l->file.tensors[i].gt;
            if (t && t->type != GGML_TYPE_F32 && !t->buffer) n_off++;
        }
        if (n_off > 0) {
            const void ** srcs = (const void **) malloc(sizeof(void *) * (size_t) l->file.n_tensors);
            for (int i = 0; i < l->file.n_tensors; i++) {
                ggml_tensor * t = l->file.tensors[i].gt;
                if (t && t->type != GGML_TYPE_F32 && !t->buffer) { srcs[i] = t->data; t->data = NULL; }
                else srcs[i] = NULL;
            }
            l->vk_wbuf = ggml_backend_alloc_ctx_tensors(l->ctx_w, l->vk);
            if (!l->vk_wbuf) {
                LOG_ERROR("vulkan: failed to allocate weights in VRAM, falling back to CPU\n");
            } else {
                for (int i = 0; i < l->file.n_tensors; i++) {
                    ggml_tensor * t = l->file.tensors[i].gt;
                    if (srcs[i]) ggml_backend_tensor_set(t, srcs[i], 0, (size_t) ggml_nbytes(t));
                }
                LOG_INFO("vulkan: %d tensors offloaded to VRAM (%.1f MB)\n", n_off,
                         (double) ggml_backend_buffer_get_size(l->vk_wbuf) / 1048576.0);
            }
            free(srcs);
        }
        if (l->vk_wbuf) {
            struct ggml_init_params vp = { .mem_size = (size_t) 16 << 20, .mem_buffer = NULL, .no_alloc = true };
            l->vk_ctx = ggml_init(vp);
            struct ggml_init_params vgp = { .mem_size = (size_t) 1 << 20, .mem_buffer = NULL, .no_alloc = false };
            l->vk_gctx = ggml_init(vgp);
            l->vk_graph = ggml_new_graph_custom(l->vk_gctx, 16, false);
            // persistent compute buffer for the per-call matmul (input + output).
            // llama/gemma3 prompt batching processes up to 32 tokens at once.
            int vk_batch = (l->arch == LLM_ARCH_LFM2 || l->arch == LLM_ARCH_GEMMA3 ||
                    l->arch == LLM_ARCH_QWEN35) ? 32 : 1;
            {
                size_t max_in = 0, max_out = 0;
                for (int i = 0; i < l->file.n_tensors; i++) {
                    ggml_tensor * t = l->file.tensors[i].gt;
                    if (t && t->buffer) {
                        if ((size_t) t->ne[0] * 4 > max_in)  max_in  = (size_t) t->ne[0] * 4;
                        if ((size_t) t->ne[1] * 4 > max_out) max_out = (size_t) t->ne[1] * 4;
                    }
                }
                l->vk_cbuf = ggml_backend_alloc_buffer(l->vk,
                    (size_t) vk_batch * (max_in + max_out) + ((size_t) 1 << 20));
                if (!l->vk_cbuf) {
                    LOG_ERROR("vulkan: failed to allocate compute buffer, falling back to CPU\n");
                } else {
                    l->vk_off = 1;
                }
            }
        }
        if (!l->vk_off) {
            if (l->vk_wbuf) { ggml_backend_buffer_free(l->vk_wbuf); l->vk_wbuf = NULL; }
            if (l->vk_cbuf) { ggml_backend_buffer_free(l->vk_cbuf); l->vk_cbuf = NULL; }
            if (l->vk_ctx)  { ggml_free(l->vk_ctx); l->vk_ctx = NULL; }
            if (l->vk_gctx) { ggml_free(l->vk_gctx); l->vk_gctx = NULL; }
            ggml_backend_free(l->vk);
            l->vk = NULL;
        }
    }

    // scratch for reading back the quantized embedding row from the GPU
    l->embd_raw = malloc((size_t) l->w_token_embd->nb[1]);
    if (!l->embd_raw) { LOG_ERROR("failed to alloc embd_raw\n"); return -1; }
#endif // GGML_USE_VULKAN

    size_t cbuf = (size_t) 512 << 20;
    l->compute_buf = malloc(cbuf);
    l->compute_buf_size = cbuf;
    struct ggml_init_params cp = { .mem_size = cbuf, .mem_buffer = l->compute_buf, .no_alloc = false };
    l->ctx_compute = ggml_init(cp);

    const int graph_size = 16384;
    size_t gbuf = ggml_graph_overhead_custom(graph_size, false) + (size_t) (1 << 20);
    l->graph_buf = malloc(gbuf);
    struct ggml_init_params gp = { .mem_size = gbuf, .mem_buffer = l->graph_buf, .no_alloc = false };
    l->ctx_graph = ggml_init(gp);
    l->graph = ggml_new_graph_custom(l->ctx_graph, graph_size, false);

    l->k_cache = (float **) calloc(l->n_layer, sizeof(float *));
    l->v_cache = (float **) calloc(l->n_layer, sizeof(float *));
    l->gdn_state = (float **) calloc(l->n_layer, sizeof(float *));
    l->gdn_conv = (float **) calloc(l->n_layer, sizeof(float *));
    l->lfm2_state = (float **) calloc(l->n_layer, sizeof(float *));
    for (int b = 0; b < l->n_layer; b++) {
        if (l->blocks[b].is_full_attn) {
            int win = l->blocks[b].win > 0 ? l->blocks[b].win : l->n_ctx;
            size_t sz = (size_t) win * l->n_head_kv * l->head_dim;
            l->k_cache[b] = (float *) calloc(sz, sizeof(float));
            l->v_cache[b] = (float *) calloc(sz, sizeof(float));
        } else if (l->arch == LLM_ARCH_LFM2) {
            int d_conv = l->shortconv_l_cache - 1;
            size_t ssz = (size_t) d_conv * l->n_embd;
            l->lfm2_state[b] = (float *) calloc(ssz, sizeof(float));
        } else {
            size_t ssz = (size_t) l->ssm_num_v_heads * l->ssm_head_dim * l->ssm_head_dim;
            l->gdn_state[b] = (float *) calloc(ssz, sizeof(float));
            int conv_dim = 2 * l->ssm_num_k_heads * l->ssm_head_dim + l->ssm_num_v_heads * l->ssm_head_dim;
            l->gdn_conv[b] = (float *) calloc((size_t) (l->ssm_conv_kernel - 1) * conv_dim, sizeof(float));
        }
    }
    l->logits = (float *) malloc(sizeof(float) * (size_t) l->tok.n_vocab);
    l->n_logits = l->tok.n_vocab;

    // scratch for the direct quantized GEMV (largest weight row + quantized row)
    l->mm_cap = 0;
    for (int i = 0; i < l->file.n_tensors; i++) {
        ggml_tensor * gt = l->file.tensors[i].gt;
        if (gt && (size_t) gt->ne[0] > l->mm_cap) l->mm_cap = (size_t) gt->ne[0];
    }
    l->mm_xbuf = (float *) malloc((l->mm_cap + 64) * sizeof(float));
    l->mm_qbuf = malloc(l->mm_cap * 4 + 4096);
    if (!l->mm_xbuf || !l->mm_qbuf) { LOG_ERROR("failed to alloc GEMV scratch\n"); return -1; }
    l->rng_state = ((uint64_t) time(NULL) ^ 0x9E3779B97F4A7C15ULL);

    int half = l->n_rope / 2;
    // llama3-style scaled rope: rope_freqs.weight holds per-frequency factors
    // that damp the high-frequency rotation angles (angle /= freq_factor).
    l->rope_freqs = NULL;
    l->n_rope_freqs = 0;
    ggml_tensor * rope_freqs_t = gguf_get(&l->file, "rope_freqs.weight");
    if (rope_freqs_t && rope_freqs_t->type == GGML_TYPE_F32) {
        int n = (int) rope_freqs_t->ne[0];
        if (n > 0) {
            l->rope_freqs = (float *) malloc(sizeof(float) * (size_t) n);
            memcpy(l->rope_freqs, rope_freqs_t->data, sizeof(float) * (size_t) n);
            l->n_rope_freqs = n;
        }
    }
    l->rope_cos = (float *) malloc(sizeof(float) * (size_t) l->n_ctx * half);
    l->rope_sin = (float *) malloc(sizeof(float) * (size_t) l->n_ctx * half);
    for (int p = 0; p < l->n_ctx; p++) {
        for (int i = 0; i < half; i++) {
            float ff = (l->rope_freqs && i < l->n_rope_freqs) ? l->rope_freqs[i] : 1.0f;
            if (ff == 0.0f) ff = 1.0f;
            float f = powf(l->f_rope_freq_base, -2.0f * i / (float) l->n_rope) * (float) p / ff;
            l->rope_cos[(size_t) p * half + i] = cosf(f);
            l->rope_sin[(size_t) p * half + i] = sinf(f);
        }
    }
    if (l->arch == LLM_ARCH_GEMMA3) {
        float base = l->f_rope_freq_swa > 0 ? l->f_rope_freq_swa : 10000.0f;
        l->rope_cos_swa = (float *) malloc(sizeof(float) * (size_t) l->n_ctx * half);
        l->rope_sin_swa = (float *) malloc(sizeof(float) * (size_t) l->n_ctx * half);
        for (int p = 0; p < l->n_ctx; p++) {
            for (int i = 0; i < half; i++) {
                float ff = (l->rope_freqs && i < l->n_rope_freqs) ? l->rope_freqs[i] : 1.0f;
                if (ff == 0.0f) ff = 1.0f;
                float f = powf(base, -2.0f * i / (float) l->n_rope) * (float) p / ff;
                l->rope_cos_swa[(size_t) p * half + i] = cosf(f);
                l->rope_sin_swa[(size_t) p * half + i] = sinf(f);
            }
        }
    }

    l->work = (float *) malloc(sizeof(float) * (size_t) l->n_logits);
    l->cand = (llm_cand_t *) malloc(sizeof(llm_cand_t) * (size_t) l->n_logits);
    l->gen_hist = (int *) malloc(sizeof(int) * (size_t) l->n_ctx);

    int key_dim = l->ssm_num_k_heads * l->ssm_head_dim;
    int value_dim = l->ssm_num_v_heads * l->ssm_head_dim;
    int conv_dim = key_dim * 2 + value_dim;
    int q_full_dim = l->n_head * l->head_dim * 2;
    int kv_dim = l->n_head_kv * l->head_dim;
    // the attention / GDN output buffer holds nh*hd or nv*hd floats, which can
    // exceed n_embd (e.g. Qwen3.5-4B: nh*hd = nv*hd = 4096 > embd 2560)
    int o_dim = l->n_embd_eff;
    if (l->n_head * l->head_dim > o_dim) o_dim = l->n_head * l->head_dim;
    if (l->ssm_num_v_heads * l->ssm_head_dim > o_dim) o_dim = l->ssm_num_v_heads * l->ssm_head_dim;
    size_t per_tok = (size_t) (l->n_embd_eff + l->n_embd_eff
                                  + conv_dim + value_dim + conv_dim
                                  + key_dim + key_dim + value_dim
                                  + o_dim + l->n_embd_eff
                                  + q_full_dim + kv_dim + kv_dim
                                  + l->n_ff + l->n_ff + l->n_embd_eff
                                  + l->ssm_num_v_heads + l->ssm_num_v_heads + l->ssm_head_dim
                                  + 3 * l->n_embd + l->n_embd + l->n_embd);
    // batch factor for the shared scratch (qwen35 reuses llm_bufs(B) layout)
    int bch = (l->arch == LLM_ARCH_LFM2 || l->arch == LLM_ARCH_GEMMA3 ||
               l->arch == LLM_ARCH_QWEN35) ? 32 : 1;
    size_t sc = per_tok * (size_t) bch + (size_t) l->n_ctx + 1024;
    l->scratch = (float *) malloc(sc * sizeof(float));
    l->scratch_size = sc;
    int n_th = l->n_threads > 0 ? l->n_threads : 4;
    l->scores_mt = (float *) malloc((size_t) n_th * (size_t) l->n_ctx * sizeof(float));
    if (!l->scores_mt) { LOG_ERROR("failed to alloc scores_mt\n"); return -1; }

    // batched prompt-processing scratch (llama/gemma3 use dedicated buffers,
    // qwen35 reuses llm_bufs(B) from l->scratch)
    l->batch_max = 0;
    if (bch > 1) l->batch_max = bch;
    if (l->arch == LLM_ARCH_LFM2 || l->arch == LLM_ARCH_GEMMA3) {
        const int B = 32;
        int qd = l->n_head * l->head_dim * 2;
        int kvd = l->n_head_kv * l->head_dim;
        size_t e = (size_t) l->n_embd_eff;
        l->bx = (float *) malloc(B * e * sizeof(float));
        l->br = (float *) malloc(B * e * sizeof(float));
        l->bout = (float *) malloc(B * e * sizeof(float));
        l->bo = (float *) malloc(B * e * sizeof(float));
        l->bq = (float *) malloc((size_t) B * qd * sizeof(float));
        l->bk = (float *) malloc((size_t) B * kvd * sizeof(float));
        l->bv = (float *) malloc((size_t) B * kvd * sizeof(float));
        l->bfg = (float *) malloc((size_t) B * l->n_ff * sizeof(float));
        l->bfu = (float *) malloc((size_t) B * l->n_ff * sizeof(float));
        l->bscores = (float *) malloc((size_t) l->n_ctx * sizeof(float));
        l->bq8 = malloc((size_t) B * l->mm_cap * 2 + 4096);
        l->bvin = (float *) malloc((size_t) B * l->mm_cap * sizeof(float));
        if (!l->bx || !l->br || !l->bout || !l->bo || !l->bq || !l->bk || !l->bv ||
            !l->bfg || !l->bfu || !l->bscores || !l->bq8 || !l->bvin) {
            LOG_ERROR("failed to alloc batched prompt scratch\n");
            return -1;
        }
    }
    if (l->arch == LLM_ARCH_QWEN35) {
        const int B = 32;
        l->bscores = (float *) malloc((size_t) l->n_ctx * sizeof(float));
        l->bq8 = malloc((size_t) B * l->mm_cap * 2 + 4096);
        l->bvin = (float *) malloc((size_t) B * l->mm_cap * sizeof(float));
        if (!l->bscores || !l->bq8 || !l->bvin) {
            LOG_ERROR("failed to alloc qwen35 batched prompt scratch\n");
            return -1;
        }
    }

    l->f_embd_scale = 1.0f;
    if (l->arch == LLM_ARCH_GEMMA3) {
        int probe[4] = { l->tok.bos, 105, 2, 0 };
        float max_rms = 0.f;
        for (int i = 0; i < 4; i++) {
            int id = probe[i];
            if (id < 0 || id >= l->tok.n_vocab) continue;
            llm_get_embd(l, id, l->scratch);
            float ss = 0.f;
            for (int j = 0; j < l->n_embd; j++) ss += l->scratch[j] * l->scratch[j];
            float rms = sqrtf(ss / (float) l->n_embd);
            if (rms > max_rms) max_rms = rms;
        }
        if (max_rms > 1e-4f && max_rms < 0.1f) l->f_embd_scale = sqrtf((float) l->n_embd);
    }
    return 0;
}

// ---------------------------------------------------------------------------
// chat / template (Qwen3 ChatML)
// ---------------------------------------------------------------------------
void llm_add_message(llm_t * l, const char * role, const char * content) {
    if (l->n_msg >= LLM_MAX_MSG) { LOG_ERROR("too many messages (max %d)\n", LLM_MAX_MSG); return; }
    l->msg_role[l->n_msg] = strdup(role ? role : "user");
    l->msg_content[l->n_msg] = strdup(content ? content : "");
    l->n_msg++;
}

void llm_clear_messages(llm_t * l) {
    for (int i = 0; i < l->n_msg; i++) {
        free(l->msg_role[i]);
        free(l->msg_content[i]);
        l->msg_role[i] = NULL;
        l->msg_content[i] = NULL;
    }
    l->n_msg = 0;
}

// ---------------------------------------------------------------------------
// tools
// ---------------------------------------------------------------------------
void llm_set_tools(llm_t * l, const llm_tool_t * tools, int n_tools) {
    if (l->tools) {
        for (int i = 0; i < l->n_tools; i++) {
            free((void *) l->tools[i].name);
            free((void *) l->tools[i].description);
            free((void *) l->tools[i].parameters);
        }
        free(l->tools);
        l->tools = NULL;
        l->n_tools = 0;
    }
    if (n_tools > 0 && tools) {
        l->tools = (llm_tool_t *) calloc((size_t) n_tools, sizeof(llm_tool_t));
        for (int i = 0; i < n_tools; i++) {
            l->tools[i].name        = strdup(tools[i].name        ? tools[i].name        : "");
            l->tools[i].description = strdup(tools[i].description ? tools[i].description : "");
            l->tools[i].parameters  = strdup(tools[i].parameters  ? tools[i].parameters  : "{}");
        }
        l->n_tools = n_tools;
    }
}

// "[{\"type\":\"function\",\"function\":{...}}, ...]"
static void sb_append_tools_json(sb_t * sb, const llm_t * l) {
    sb_append(sb, "[");
    for (int i = 0; i < l->n_tools; i++) {
        if (i) sb_append(sb, ",");
        sb_append(sb, "{\"type\":\"function\",\"function\":{\"name\":\"");
        sb_append(sb, l->tools[i].name);
        sb_append(sb, "\",\"description\":\"");
        sb_append(sb, l->tools[i].description);
        sb_append(sb, "\",\"parameters\":");
        sb_append(sb, l->tools[i].parameters);
        sb_append(sb, "}}");
    }
    sb_append(sb, "]");
}

void llm_tool_call_free(llm_tool_call_t * c) {
    if (c) { free(c->arguments); c->arguments = NULL; }
}

// parse a JSON object that represents a tool call and fill *out.
// Only {"name": ..., "arguments": {...}} (with the "arguments" key present) is
// accepted; tool *definitions* ({"type":"function","function":{...}}) have no
// "arguments" and must NOT be treated as calls.
static int tool_json_parse(const char * obj, int len, llm_tool_call_t * out) {
    char * buf = strndup_(obj, (size_t) len);
    json_value_t * root = json_parse(buf);
    if (!root || root->type != JSON_OBJ) {
        if (root) json_free(root);
        free(buf);
        return -1;
    }
    json_value_t * fn = json_obj_get(root, "function");
    if (!fn) fn = root;
    json_value_t * nm = json_obj_get(fn, "name");
    json_value_t * args = json_obj_get(fn, "arguments");
    if (!nm || nm->type != JSON_STR || !args) {
        json_free(root);
        free(buf);
        return -1;
    }
    snprintf(out->name, sizeof(out->name), "%s", nm->u.str);
    if (args->type == JSON_OBJ)      out->arguments = json_to_str(args);
    else if (args->type == JSON_STR) out->arguments = strdup(args->u.str);
    else                             out->arguments = strdup("{}");
    json_free(root);
    free(buf);
    return 0;
}

int llm_parse_tool_call(const char * text, llm_tool_call_t * out) {
    if (!text || !out) return -1;
    out->arguments = NULL;
    // qwen3/gemma3: <tool_call> ... </tool_call>
    const char * b = strstr(text, "<tool_call>");
    if (b) {
        b += strlen("<tool_call>");
        while (*b == '\n' || *b == '\r' || *b == ' ' || *b == '\t') b++;
        const char * e = strstr(b, "</tool_call>");
        if (e) {
            while (e > b && (e[-1] == '\n' || e[-1] == '\r' || e[-1] == ' ' || e[-1] == '\t')) e--;
            return tool_json_parse(b, (int)(e - b), out);
        }
    }
    // llama: plain {"name":...,"arguments":{...}} JSON object somewhere in the text
    for (const char * p = text; (p = strchr(p, '{')) != NULL; p++) {
        int depth = 0;
        const char * q = p;
        for (; *q; q++) {
            if (*q == '{') depth++;
            else if (*q == '}') { depth--; if (depth == 0) break; }
        }
        if (depth == 0 && q > p) {
            if (tool_json_parse(p, (int)(q - p + 1), out) == 0) return 0;
        }
    }
    return -1;
}

static void sb_append_trimmed(sb_t * sb, const char * s) {
    while (*s && (*s == ' ' || *s == '\t' || *s == '\n' || *s == '\r')) s++;
    const char * e = s + strlen(s);
    while (e > s && (e[-1] == ' ' || e[-1] == '\t' || e[-1] == '\n' || e[-1] == '\r')) e--;
    sb_append_n(sb, s, (int)(e - s));
}

char * llm_apply_chat_template(llm_t * l, int add_generation_prompt) {
    sb_t sb;
    sb_init(&sb);
    if (l->arch == LLM_ARCH_GEMMA3) {
        sb_append_n(&sb, "<bos>", 5);
        int loop_start = 0;
        if (l->n_msg > 0 && !strcmp(l->msg_role[0], "system")) {
            loop_start = 1;
            const char * s = l->msg_content[0];
            while (*s && (*s == ' ' || *s == '\t' || *s == '\n' || *s == '\r')) s++;
            const char * e = s + strlen(s);
            while (e > s && (e[-1] == ' ' || e[-1] == '\t' || e[-1] == '\n' || e[-1] == '\r')) e--;
            sb_append_n(&sb, "<start_of_turn>system\n", 22);
            sb_append_n(&sb, s, (int)(e - s));
            sb_append_n(&sb, "\n<end_of_turn>\n", 15);
        }
        if (l->n_tools > 0) {
            sb_append_n(&sb, "<start_of_turn>system\n<tools>\n", 30);
            sb_append_tools_json(&sb, l);
            sb_append_n(&sb, "\n<end_of_turn>\n", 15);
        }
        for (int i = loop_start; i < l->n_msg; i++) {
            const char * role = l->msg_role[i];
            const char * content = l->msg_content[i];
            const char * r = (role && !strcmp(role, "assistant")) ? "model" : (role ? role : "user");
            sb_append_n(&sb, "<start_of_turn>", 15);
            sb_append(&sb, r);
            sb_append(&sb, "\n");
            // assistant content is re-tokenized from the generated reply; render
            // it verbatim so the KV cache prefix can be reused across turns
            if (role && !strcmp(role, "assistant")) sb_append(&sb, content);
            else sb_append_trimmed(&sb, content);
            sb_append_n(&sb, "<end_of_turn>", 13);
            sb_append(&sb, "\n");
        }
        if (add_generation_prompt) {
            sb_append_n(&sb, "<start_of_turn>", 15);
            sb_append(&sb, "model\n");
        }
        return sb_str(&sb);
    }
    if (l->arch == LLM_ARCH_LFM2) {
        // LFM2 ChatML-style template: bos + <|im_start|>role\ncontent<|im_end|>\n
        if (l->tok.bos >= 0 && l->tok.bos < l->tok.n_vocab && l->tok.tokens[l->tok.bos])
            sb_append(&sb, l->tok.tokens[l->tok.bos]);
        for (int i = 0; i < l->n_msg; i++) {
            const char * role = l->msg_role[i];
            const char * content = l->msg_content[i];
            if (!strcmp(role, "system")) {
                sb_append(&sb, "<|im_start|>system\n");
                sb_append_trimmed(&sb, content);
                sb_append(&sb, "<|im_end|>\n");
                if (l->n_tools > 0) {
                    sb_append(&sb, "<|im_start|>system\n<|tools|>\n");
                    sb_append_tools_json(&sb, l);
                    sb_append(&sb, "<|im_end|>\n");
                }
            } else if (!strcmp(role, "user")) {
                sb_append(&sb, "<|im_start|>user\n");
                sb_append_trimmed(&sb, content);
                sb_append(&sb, "<|im_end|>\n");
            } else if (!strcmp(role, "tool")) {
                sb_append(&sb, "<|im_start|>tool\n");
                sb_append_trimmed(&sb, content);
                sb_append(&sb, "<|im_end|>\n");
            } else if (!strcmp(role, "assistant")) {
                sb_append(&sb, "<|im_start|>assistant\n");
                sb_append(&sb, content);
                sb_append(&sb, "<|im_end|>\n");
            }
        }
        if (add_generation_prompt) {
            sb_append(&sb, "<|im_start|>assistant\n");
        }
        return sb_str(&sb);
    }
    for (int i = 0; i < l->n_msg; i++) {
        const char * role = l->msg_role[i];
        const char * content = l->msg_content[i];
        if (!strcmp(role, "system")) {
            if (i == 0) {
                sb_append(&sb, "<|im_start|>system\n");
                sb_append_trimmed(&sb, content);
                sb_append(&sb, "<|im_end|>\n");
                if (l->n_tools > 0) {
                    sb_append(&sb, "<|im_start|>system\n<|tools|>\n");
                    sb_append_tools_json(&sb, l);
                    sb_append(&sb, "<|im_end|>\n");
                }
            }
        } else if (!strcmp(role, "user")) {
            sb_append(&sb, "<|im_start|>user\n");
            sb_append_trimmed(&sb, content);
            sb_append(&sb, "<|im_end|>\n");
        } else if (!strcmp(role, "tool")) {
            sb_append(&sb, "<|im_start|>tool\n");
            sb_append_trimmed(&sb, content);
            sb_append(&sb, "<|im_end|>\n");
        } else if (!strcmp(role, "assistant")) {
            if (strstr(content, "<tool_call>")) {
                // tool-call message: emit verbatim (no thinking/response wrapper)
                sb_append(&sb, "<|im_start|>assistant\n");
                sb_append(&sb, content);
                sb_append(&sb, "<|im_end|>\n");
            } else {
                // history holds the raw reply (reasoning + answer); render it
                // verbatim so the KV-cache prefix matches the cached tokens
                sb_append(&sb, "<|im_start|>assistant\n thinking\n");
                sb_append(&sb, content);
                sb_append(&sb, "<|im_end|>\n");
            }
        }
    }
    if (add_generation_prompt) {
        sb_append(&sb, "<|im_start|>assistant\n thinking\n");
    }
    return sb_str(&sb);
}

// render only the tail of the conversation for an incremental prompt: close the
// turn of message[from-1] and append messages[from .. n_msg-1] plus the
// generation prompt. from == 0 falls back to the full template.
char * llm_apply_chat_template_tail(llm_t * l, int from, int add_generation_prompt) {
    if (from <= 0 || from >= l->n_msg) return llm_apply_chat_template(l, add_generation_prompt);
    sb_t sb;
    sb_init(&sb);
    // close the previous turn (the assistant reply that is already in the cache)
    if (l->arch == LLM_ARCH_GEMMA3)        sb_append_n(&sb, "<end_of_turn>\n", 14);
    else                                   sb_append_n(&sb, "<|im_end|>\n", 11);
    for (int i = from; i < l->n_msg; i++) {
        const char * role = l->msg_role[i];
        const char * content = l->msg_content[i];
        if (l->arch == LLM_ARCH_GEMMA3) {
            const char * r = (role && !strcmp(role, "assistant")) ? "model" : (role ? role : "user");
            sb_append_n(&sb, "<start_of_turn>", 15);
            sb_append(&sb, r);
            sb_append(&sb, "\n");
            if (role && !strcmp(role, "assistant")) sb_append(&sb, content);
            else sb_append_trimmed(&sb, content);
            sb_append_n(&sb, "<end_of_turn>\n", 14);
        } else {
            // ChatML (qwen35 / lfm2). lfm2 has no "thinking" reasoning block.
            if (!strcmp(role, "tool")) {
                sb_append(&sb, "<|im_start|>tool\n");
                sb_append_trimmed(&sb, content);
                sb_append(&sb, "<|im_end|>\n");
            } else if (!strcmp(role, "user")) {
                sb_append(&sb, "<|im_start|>user\n");
                sb_append_trimmed(&sb, content);
                sb_append(&sb, "<|im_end|>\n");
            } else {
                sb_append(&sb, "<|im_start|>assistant\n");
                if (l->arch != LLM_ARCH_LFM2) {
                    // qwen35 history holds the raw reply (reasoning + answer);
                    // render it verbatim so the KV-cache prefix matches
                    if (!strstr(content, "<tool_call>")) sb_append(&sb, " thinking\n");
                }
                sb_append(&sb, content);
                sb_append(&sb, "<|im_end|>\n");
            }
        }
    }
    if (add_generation_prompt) {
        if (l->arch == LLM_ARCH_GEMMA3)        sb_append_n(&sb, "<start_of_turn>model\n", 21);
        else if (l->arch == LLM_ARCH_LFM2)     sb_append(&sb, "<|im_start|>assistant\n");
        else                                   sb_append(&sb, "<|im_start|>assistant\n thinking\n");
    }
    return sb_str(&sb);
}

// ---------------------------------------------------------------------------
// sampling
// ---------------------------------------------------------------------------
static uint64_t rng_next(llm_t * l) {
    uint64_t x = l->rng_state;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    l->rng_state = x;
    return x * 0x2545F4914F6CDD1DULL;
}
static float rng_float(llm_t * l) {
    return (float)((rng_next(l) >> 40) * (1.0 / 16777216.0));
}

// descending sort so the highest-probability candidate is first
static int cand_cmp(const void * pa, const void * pb) {
    const llm_cand_t * a = (const llm_cand_t *) pa, * b = (const llm_cand_t *) pb;
    return (b->l > a->l) - (b->l < a->l);
}

// quickselect: rearrange a[0..n) so that the first k entries are the k
// largest (unsorted). Average O(n). Used to apply top-k BEFORE the full
// sort and before any expf, so sampling costs O(V + k log k) instead of
// O(V log V + V expf) per token (V ~ 262k for gemma3).
// Lomuto partition with a median-of-three pivot (every iteration strictly
// shrinks the window, so termination is guaranteed).
static void cand_topk_select(llm_cand_t * a, int n, int k) {
    int lo = 0, hi = n - 1;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (a[lo].l < a[mid].l) { llm_cand_t t = a[lo]; a[lo] = a[mid]; a[mid] = t; }
        if (a[mid].l < a[hi].l) { llm_cand_t t = a[mid]; a[mid] = a[hi]; a[hi] = t; }
        { llm_cand_t t = a[mid]; a[mid] = a[hi]; a[hi] = t; } // median -> hi
        float pivot = a[hi].l;
        int i = lo;
        for (int m = lo; m < hi; m++) {
            if (a[m].l > pivot) {
                llm_cand_t t = a[i]; a[i] = a[m]; a[m] = t;
                i++;
            }
        }
        { llm_cand_t t = a[i]; a[i] = a[hi]; a[hi] = t; } // pivot -> i
        // [lo..i-1] > pivot == a[i], [i+1..hi] <= pivot
        if (i == k - 1) break;
        else if (i < k - 1) lo = i + 1;
        else hi = i - 1;
    }
}

// fixed repeat-penalty: already-generated tokens have their logits scaled
// down by this factor so the model is discouraged from repeating itself
#define LLM_REPEAT_PENALTY 1.1f

// sample a token from the given logits buffer
static int llm_sample_from(llm_t * l, const float * logits) {
    const llm_sampler_t * s = &l->params.sampler;
    int n = l->n_logits;
    float * buf = l->work;
    memcpy(buf, logits, sizeof(float) * (size_t) n);
    // apply repeat penalty over the generated history (llama.cpp convention:
    // divide positive logits by penalty, multiply negative ones by it)
    if (LLM_REPEAT_PENALTY > 1.0f && l->n_gen_hist > 0) {
        float inv_pen = 1.0f / LLM_REPEAT_PENALTY;
        for (int h = 0; h < l->n_gen_hist; h++) {
            int id = l->gen_hist[h];
            if (id < 0 || id >= n) continue;
            buf[id] = buf[id] > 0.f ? buf[id] * inv_pen : buf[id] * LLM_REPEAT_PENALTY;
        }
    }
    if (s->temp < 1e-4f) {
        int best = 0;
        for (int i = 1; i < n; i++) if (buf[i] > buf[best]) best = i;
        return best;
    }
    float mx = buf[0];
    for (int i = 1; i < n; i++) if (buf[i] > mx) mx = buf[i];
    float inv_t = 1.0f / s->temp;
    for (int i = 0; i < n; i++) buf[i] = (buf[i] - mx) * inv_t;
    int nc = 0;
    for (int i = 0; i < n; i++) {
        if (buf[i] <= -50.f) continue;
        l->cand[nc].l = buf[i];
        l->cand[nc].i = i;
        nc++;
    }
    if (nc == 0) {
        int best = 0;
        for (int i = 1; i < n; i++) if (buf[i] > buf[best]) best = i;
        return best;
    }
    // top-k BEFORE sort/expf (was: qsort the whole vocab, then cut to top-k
    // and run expf over all V). Now only k entries are sorted and exponentiated.
    if (s->top_k > 0 && s->top_k < nc) {
        cand_topk_select(l->cand, nc, s->top_k);
        nc = s->top_k;
    }
    qsort(l->cand, (size_t) nc, sizeof(llm_cand_t), cand_cmp);
    if (s->top_p < 1.0f) {
        float total = 0.f;
        for (int i = 0; i < nc; i++) total += expf(l->cand[i].l);
        float acc = 0.f;
        for (int i = 0; i < nc; i++) {
            acc += expf(l->cand[i].l);
            if (acc >= s->top_p * total) { nc = i + 1; break; }
        }
        if (nc < 1) nc = 1;
    }
    float total = 0.f;
    for (int i = 0; i < nc; i++) { l->cand[i].l = expf(l->cand[i].l); total += l->cand[i].l; }
    float r = rng_float(l) * total;
    for (int i = 0; i < nc; i++) {
        r -= l->cand[i].l;
        if (r <= 0.f) return l->cand[i].i;
    }
    return l->cand[nc-1].i;
}

static void llm_forward_token(llm_t * l, int token_id, int pos) {
    if (l->arch == LLM_ARCH_GEMMA3) llm_forward_gemma(l, token_id, pos);
    else if (l->arch == LLM_ARCH_LFM2) llm_forward_lfm2(l, token_id, pos);
    else llm_forward(l, token_id, pos);
}

// ---------------------------------------------------------------------------
// generation
// ---------------------------------------------------------------------------
// clear the KV / GDN caches (fresh generation)
static void llm_cache_discard(llm_t * l) {
    free(l->cache_toks);
    l->cache_toks = NULL;
    l->cache_n = 0;
    l->cache_cap = 0;
    l->cache_ok = 0;
}

// record one more token as resident in the KV/GDN caches. On OOM the mirror
// is discarded (cache_ok=0) so later continuations safely fall back to a
// fresh full pass instead of trusting a partial mirror.
static int llm_cache_append(llm_t * l, int id) {
    if (!l->cache_ok) return -1;
    if (l->cache_n == l->cache_cap) {
        int nc = l->cache_cap ? l->cache_cap * 2 : 256;
        int * nt = (int *) realloc(l->cache_toks, sizeof(int) * (size_t) nc);
        if (!nt) { llm_cache_discard(l); return -1; }
        l->cache_toks = nt;
        l->cache_cap = nc;
    }
    l->cache_toks[l->cache_n++] = id;
    return 0;
}

static void llm_reset_caches(llm_t * l) {
    for (int b = 0; b < l->n_layer; b++) {
        if (l->blocks[b].is_full_attn) {
            int win = l->blocks[b].win > 0 ? l->blocks[b].win : l->n_ctx;
            size_t sz = (size_t) win * l->n_head_kv * l->head_dim * sizeof(float);
            memset(l->k_cache[b], 0, sz);
            memset(l->v_cache[b], 0, sz);
        } else if (l->arch == LLM_ARCH_LFM2) {
            int d_conv = l->shortconv_l_cache - 1;
            memset(l->lfm2_state[b], 0, (size_t) d_conv * l->n_embd * sizeof(float));
        } else {
            size_t ssz = (size_t) l->ssm_num_v_heads * l->ssm_head_dim * l->ssm_head_dim * sizeof(float);
            memset(l->gdn_state[b], 0, ssz);
            int conv_dim = 2 * l->ssm_num_k_heads * l->ssm_head_dim + l->ssm_num_v_heads * l->ssm_head_dim;
            memset(l->gdn_conv[b], 0, (size_t) (l->ssm_conv_kernel - 1) * conv_dim * sizeof(float));
        }
    }
}

// shared implementation. n_past is the number of tokens already present in the
// caches; only prompt_tokens[n_past .. n_prompt-1] are newly processed.
static int llm_generate_impl(llm_t * l, const int * prompt_tokens, int n_prompt, int n_past,
                             int n_max_tokens, llm_token_cb cb, void * user_data) {
    // validate a cache continuation: prompt_tokens[0..n_past) must be exactly
    // what is already resident in the KV/GDN caches. On any mismatch fall
    // back to a fresh full pass (reads of foreign cached state silently
    // corrupt attention/GDN recurrence, so never trust the caller blindly).
    if (n_past == 0) {
        llm_cache_discard(l);
        l->cache_ok = 1;
    } else if (!l->cache_ok || n_past > l->cache_n || !l->cache_toks ||
               memcmp(prompt_tokens, l->cache_toks, sizeof(int) * (size_t) n_past) != 0) {
        LOG_ERROR("llm_generate_from: cache prefix mismatch (n_past=%d cached=%d), full recompute\n",
                  n_past, l->cache_ok ? l->cache_n : -1);
        llm_reset_caches(l);
        llm_cache_discard(l);
        l->cache_ok = 1;
        n_past = 0;
    }
    l->n_past = n_past;
    l->n_gen_hist = 0;
    l->n_prompt_tokens = n_prompt - n_past;
    l->n_gen_tokens = 0;
    if (getenv("LLM_DEBUG")) {
        fprintf(stderr, "[dbg] prompt (+%d):\n", n_prompt - n_past);
        for (int i = n_past; i < n_prompt; i++) {
            int id = prompt_tokens[i];
            const char * s = (id >= 0 && id < l->tok.n_vocab) ? l->tok.tokens[id] : "<?>";
            fprintf(stderr, " %d|%s", id, s ? s : "");
        }
        fprintf(stderr, "\n");
    }
    double t0 = now_ms();
    // process the newly added prompt tokens (batched for llama/gemma3 on CPU)
    int use_batch = (l->batch_max > 0) &&
                    (l->arch == LLM_ARCH_LFM2 || l->arch == LLM_ARCH_GEMMA3 ||
                     l->arch == LLM_ARCH_QWEN35);
    int i = n_past;
    while (i < n_prompt) {
        if (l->n_past >= l->n_ctx) { LOG_ERROR("prompt exceeds context (%d)\n", l->n_ctx); break; }
        int B = l->batch_max > 0 ? l->batch_max : 1;
        if (i + B > n_prompt) B = n_prompt - i;
        if (l->n_past + B > l->n_ctx) B = l->n_ctx - l->n_past;
        if (B <= 0) break;
        if (use_batch && B > 1) {
            if (l->arch == LLM_ARCH_LFM2)       llm_forward_batch_lfm2(l, prompt_tokens + i, B, l->n_past);
            else if (l->arch == LLM_ARCH_GEMMA3) llm_forward_batch_gemma(l, prompt_tokens + i, B, l->n_past);
            else                                llm_forward_batch_qwen35(l, prompt_tokens + i, B, l->n_past);
        } else {
            for (int j = 0; j < B; j++) llm_forward_token(l, prompt_tokens[i + j], l->n_past + j);
        }
        l->n_past += B;
        l->n_tokens += B;
        for (int j = 0; j < B; j++) llm_cache_append(l, prompt_tokens[i + j]);
        i += B;
    }
    l->t_prompt_ms = now_ms() - t0;
    t0 = now_ms();
    int n = 0;
    while (n < n_max_tokens && l->n_past < l->n_ctx) {
        int t1 = llm_sample_from(l, l->logits);
        if (t1 < 0 || t1 >= l->tok.n_vocab) break;
        if (t1 == l->tok.eos || (l->turn_end_id > 0 && t1 == l->turn_end_id)) break;
        if (cb && cb(t1, user_data)) break;
        llm_forward_token(l, t1, l->n_past);
        l->n_past++;
        l->n_tokens++;
        llm_cache_append(l, t1); // t1 is now resident in the caches (see cb order above)
        if (l->n_gen_hist < l->n_ctx) l->gen_hist[l->n_gen_hist++] = t1;
        n++;
        l->n_gen_tokens++;
    }
    l->t_gen_ms = now_ms() - t0;
    if (getenv("CHAT_TIMING")) {
        fprintf(stderr, "[t] llm_generate: total_prompt=%d new=%d gen=%d  prompt=%.0fms gen=%.0fms\n",
                n_prompt, l->n_prompt_tokens, l->n_gen_tokens, l->t_prompt_ms, l->t_gen_ms);
    }
    return n;
}

// fresh generation: reset the caches and process the whole prompt
int llm_generate(llm_t * l, const int * prompt_tokens, int n_prompt,
                 int n_max_tokens, llm_token_cb cb, void * user_data) {
    llm_reset_caches(l);
    return llm_generate_impl(l, prompt_tokens, n_prompt, 0, n_max_tokens, cb, user_data);
}

// continue generation reusing the caches: prompt_tokens[0..n_past-1] must already
// be cached (identical to what was fed before); only prompt_tokens[n_past..] are
// newly processed. n_past is the number of tokens already in the KV cache.
int llm_generate_from(llm_t * l, const int * prompt_tokens, int n_prompt, int n_past,
                      int n_max_tokens, llm_token_cb cb, void * user_data) {
    return llm_generate_impl(l, prompt_tokens, n_prompt, n_past, n_max_tokens, cb, user_data);
}

typedef struct { int * toks; int n, cap; } tok_collect_t;
static int tok_collect_cb(int id, void * ud) {
    tok_collect_t * t = (tok_collect_t *) ud;
    if (t->n == t->cap) {
        t->cap = t->cap ? t->cap * 2 : 256;
        t->toks = (int *) realloc(t->toks, sizeof(int) * (size_t) t->cap);
    }
    t->toks[t->n++] = id;
    return 0;
}

char * llm_complete(llm_t * l, const char * prompt_text, int n_max_tokens) {
    int cap = (int) strlen(prompt_text) * 2 + 64;
    int * toks = (int *) malloc(sizeof(int) * (size_t) cap);
    int n = llm_tokenize(l, prompt_text, 0, toks, cap);
    tok_collect_t tc;
    memset(&tc, 0, sizeof tc);
    llm_generate(l, toks, n, n_max_tokens, tok_collect_cb, &tc);
    free(toks);
    char * out = llm_detokenize(l, tc.toks, tc.n);
    free(tc.toks);
    return out;
}

void llm_print_stats(const llm_t * l) {
    fprintf(stderr, "\n[llm] tokens: %lld (prompt %d, generated %d)\n",
            (long long) l->n_tokens, l->n_prompt_tokens, l->n_gen_tokens);
    if (l->t_prompt_ms > 1.0 && l->n_prompt_tokens > 0)
        fprintf(stderr, "[llm] prompt: %d tok in %.2f s = %.1f tok/s\n",
                l->n_prompt_tokens, l->t_prompt_ms / 1000.0, l->n_prompt_tokens / (l->t_prompt_ms / 1000.0));
    if (l->t_gen_ms > 1.0 && l->n_gen_tokens > 0)
        fprintf(stderr, "[llm] generate: %d tok in %.2f s = %.1f tok/s\n",
                l->n_gen_tokens, l->t_gen_ms / 1000.0, l->n_gen_tokens / (l->t_gen_ms / 1000.0));
}

// ---------------------------------------------------------------------------
// high-level chat helpers (dialog history + KV-cache reuse). These keep the
// "already rendered" token prefix in l->chat_toks and reuse the model caches
// across turns, so a multi-turn dialog does not re-process the full prompt.
// ---------------------------------------------------------------------------

// locate the start of the answer, skipping a qwen3 thinking block
static int chat_ci_cmp(const char * a, const char * b, size_t n); // defined below
static const char * chat_answer_start(const char * s) {
    // The qwen3/thinking reply looks like "[thinking...]\n<marker>\n\nanswer"
    // where <marker> is a separator line. Match it EXACTLY (a trimmed line
    // equal to a known marker, optionally with a trailing ':' or '.'): the
    // old substring test also fired on ordinary prose ("I think so.",
    // "unresponsive") and silently ate the start of the answer.
    static const char * marks[] = { "response", "thinking", "think" };
    if (!s) return NULL;
    const char * p = s;
    while (*p) {
        const char * nl = strchr(p, '\n');
        size_t linelen = nl ? (size_t)(nl - p) : strlen(p);
        if (linelen < 32) {
            char line[32];
            memcpy(line, p, linelen);
            line[linelen] = 0;
            const char * t = line;
            while (*t == ' ' || *t == '\t' || *t == '\r') t++;
            char * e = (char *) t + strlen(t);
            while (e > t && (e[-1] == ' ' || e[-1] == '\t' || e[-1] == '\r')) e--;
            if (e > t && (e[-1] == ':' || e[-1] == '.')) e--;
            size_t tl = (size_t)(e - t);
            if (tl > 0 && tl <= 14) {
                char word[15];
                memcpy(word, t, tl);
                word[tl] = 0;
                for (size_t k = 0; k < sizeof(marks) / sizeof(marks[0]); k++) {
                    size_t ml = strlen(marks[k]);
                    if (tl == ml && chat_ci_cmp(word, marks[k], ml) == 0) {
                        const char * q = nl ? nl : p + strlen(p);
                        while (*q == '\n' || *q == '\r' || *q == ' ' || *q == '\t') q++;
                        return q;
                    }
                }
            }
        }
        if (!nl) break;
        p = nl + 1;
    }
    return NULL;
}

// reasoning-capable models (qwen3 / lfm2) prefix a reply with a thinking block:
// lfm2 emits "think <reasoning> think <answer>", qwen3 " thinking<reasoning>
// response<answer>". Drop everything up to and including the LAST reasoning
// separator (an opening/closing marker word, or a closing "< /think>" block), so
// only the final answer reaches the caller. A plain answer that merely contains
// such a word is left untouched: stripping only runs when the reply opens with a
// marker (the " thinking"/"response" special tokens may be glued to the text) or
// contains a reasoning closing marker.
static int chat_word_char(unsigned char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9');
}

static int chat_ci_cmp(const char * a, const char * b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        unsigned char x = (unsigned char) a[i], y = (unsigned char) b[i];
        if (x >= 'A' && x <= 'Z') x = (unsigned char) (x + 32);
        if (y >= 'A' && y <= 'Z') y = (unsigned char) (y + 32);
        if (x != y) return 1;
    }
    return 0;
}

static const char * const chat_thinking_markers[] = { "think", "thinking", "response" };

// reasoning closing markers (" response", " /think", " /response", ...); longer
// variants must be checked before their prefixes
static const char * const chat_thinking_closings[] = {
    "</thinking>", "</response>", " response", "/thinking", "/response", "/think"
};

static void chat_strip_thinking(char * s) {
    if (!s || !*s) return;
    const char * start = s;
    while (*start == ' ' || *start == '\t' || *start == '\r' || *start == '\n') start++;
    int n_markers = (int) (sizeof(chat_thinking_markers) / sizeof(chat_thinking_markers[0]));
    int n_closings = (int) (sizeof(chat_thinking_closings) / sizeof(chat_thinking_closings[0]));
    // the reply looks like a reasoning block when it opens with a marker word
    // (a marker followed by a capital letter or a non-word char; a marker that
    // merely prefixes a longer lowercase word like "thinkable" is not one) ...
    int open = 0;
    for (int i = 0; i < n_markers && !open; i++) {
        size_t wl = strlen(chat_thinking_markers[i]);
        if (chat_ci_cmp(start, chat_thinking_markers[i], wl) == 0) {
            unsigned char c = (unsigned char) start[wl];
            open = (c >= 'A' && c <= 'Z') || !chat_word_char(c);
        }
    }
    // ... or when it contains a reasoning closing marker
    int has_close = 0;
    for (int i = 0; i < n_closings && !has_close; i++) {
        if (strstr(start, chat_thinking_closings[i])) has_close = 1;
    }
    if (!open && !has_close) return;
    // find the last separator (standalone marker word or closing marker) and cut
    // the reply right after it, keeping only the final answer
    char * last = NULL;
    for (char * p = s; *p; p++) {
        int prev_wc = (p > s) && chat_word_char((unsigned char) p[-1]);
        if (!prev_wc) {
            for (int i = 0; i < n_markers; i++) {
                size_t wl = strlen(chat_thinking_markers[i]);
                if (chat_ci_cmp(p, chat_thinking_markers[i], wl) == 0 &&
                    !chat_word_char((unsigned char) p[wl])) {
                    last = p + wl;
                    break;
                }
            }
        }
        for (int i = 0; i < n_closings; i++) {
            size_t wl = strlen(chat_thinking_closings[i]);
            if (chat_ci_cmp(p, chat_thinking_closings[i], wl) == 0) {
                last = p + wl;
                break;
            }
        }
    }
    if (!last || !*last) return;
    while (*last == ' ' || *last == '\t' || *last == '\r' || *last == '\n') last++;
    if (!*last) return;
    memmove(s, last, strlen(last) + 1);
    size_t rl = strlen(s);
    while (rl && (s[rl - 1] == ' ' || s[rl - 1] == '\t' ||
                  s[rl - 1] == '\r' || s[rl - 1] == '\n')) s[--rl] = 0;
}

typedef struct {
    llm_t * l;
    int  * toks;
    int    n, cap;
    char * prev;       // reply text, built incrementally (one realloc chain)
    size_t prev_len;   // strlen(prev) without NUL
    size_t prev_cap;   // allocated size including NUL
} chat_stream_t;

static int chat_stream_cb(int id, void * ud) {
    chat_stream_t * s = (chat_stream_t *) ud;
    if (s->n == s->cap) {
        int ncap = s->cap ? s->cap * 2 : 256;
        int * nt = (int *) realloc(s->toks, sizeof(int) * (size_t) ncap);
        if (!nt) return 1; // OOM: stop generation, keep what was collected
        s->toks = nt;
        s->cap = ncap;
    }
    s->toks[s->n++] = id;
    // incremental decode: only the new token (was: llm_detokenize of the
    // whole prefix on every step -> O(N^2) with malloc/free per token)
    char * piece = llm_detokenize(s->l, &id, 1);
    if (piece) {
        size_t pl = strlen(piece);
        if (s->prev_len + pl + 1 > s->prev_cap) {
            size_t nc = (s->prev_len + pl + 1) * 2;
            if (nc < 256) nc = 256;
            char * np = (char *) realloc(s->prev, nc);
            if (!np) { free(piece); return 1; } // OOM: stop, keep collected text
            s->prev = np;
            s->prev_cap = nc;
        }
        if (s->prev) {
            memcpy(s->prev + s->prev_len, piece, pl);
            s->prev_len += pl;
            s->prev[s->prev_len] = 0;
        }
        free(piece);
    }
    if (!s->prev) return 0;
    // stop-marker check on the accumulated text (handles markers split
    // across token boundaries); the sampler already stops on single-token
    // eos/turn_end ids in llm_generate_impl, this catches the rest
    int is_gemma = (llm_get_arch(s->l) == LLM_ARCH_GEMMA3);
    char * im_end = strstr(s->prev, "<|im_end|>");
    char * eot = strstr(s->prev, "<|endoftext|>");
    char * gemma_eot = strstr(s->prev, "<end_of_turn>");
    char * cut = NULL;
    if (is_gemma) {
        cut = gemma_eot;
    } else if (im_end && (!eot || im_end < eot)) cut = im_end;
    else if (eot) cut = eot;
    int stop = (cut != NULL);
    if (stop && cut) {
        *cut = 0;
        s->prev_len = (size_t)(cut - s->prev);
    }
    return stop;
}

static int chat_can_reuse(llm_t * l) {
    if (getenv("LLM_NO_CACHE")) return 0;
    return llm_get_arch(l) == LLM_ARCH_GEMMA3 || llm_get_arch(l) == LLM_ARCH_LFM2 ||
           llm_get_arch(l) == LLM_ARCH_QWEN35;
}

// add a message and track the turn count (for the incremental KV-cache tail)
void llm_chat_add(llm_t * l, const char * role, const char * content) {
    llm_add_message(l, role, content);
    l->chat_msg_count++;
}

// read an agent prompt file and install it as the system message
int llm_chat_load_agent(llm_t * l, const char * path, const char * lang) {
    size_t al = 0;
    char * agent = read_file(path, &al);
    if (!agent) return -1;
    if (lang && lang[0]) {
        size_t extra = strlen(agent) + strlen(lang) + 32;
        char * sys = (char *) malloc(extra);
        sprintf(sys, "%s\n\nRespond in the language: %s", agent, lang);
        llm_chat_add(l, "system", sys);
        free(sys);
    } else {
        llm_chat_add(l, "system", agent);
    }
    free(agent);
    return 0;
}

// build + tokenize the warmup prompt (system/tools messages only, no
// generation prompt). The token stream doubles as the cache validator: any
// change to the prompt files changes it, forcing a cache rebuild.
static int llm_chat_warmup_tokens(llm_t * l, int ** out_toks, int * out_n) {
    *out_toks = NULL;
    *out_n = 0;
    char * tpl = llm_apply_chat_template(l, 0);
    if (!tpl) return -1;
    size_t plen = strlen(tpl);
    if (llm_get_arch(l) == LLM_ARCH_GEMMA3) {
        const char * e = "<end_of_turn>\n";
        size_t el = strlen(e);
        if (plen >= el && strcmp(tpl + plen - el, e) == 0) tpl[plen - el] = 0;
    } else if (llm_get_arch(l) == LLM_ARCH_QWEN35 || llm_get_arch(l) == LLM_ARCH_LFM2) {
        const char * e = "<|im_end|>\n";
        size_t el = strlen(e);
        if (plen >= el && strcmp(tpl + plen - el, e) == 0) tpl[plen - el] = 0;
    }
    int cap = (int) strlen(tpl) * 2 + 64;
    int * toks = (int *) malloc(sizeof(int) * (size_t) cap);
    if (!toks) { free(tpl); return -1; }
    int n = llm_tokenize(l, tpl, 0, toks, cap);
    free(tpl);
    if (n <= 0) { free(toks); return -1; }
    *out_toks = toks;
    *out_n = n;
    return 0;
}

// pre-process the constant system prompt into the KV cache right after init
void llm_chat_warmup(llm_t * l) {
    if (!chat_can_reuse(l)) return;
    if (l->chat_msg_count == 0) return;
    int * toks = NULL; int n = 0;
    if (llm_chat_warmup_tokens(l, &toks, &n) != 0) return;
    llm_generate(l, toks, n, 0, NULL, NULL);
    free(l->chat_toks);
    l->chat_toks = toks;
    l->chat_n = n;
    l->chat_from = l->chat_msg_count;
    llm_chat_capture(l);
}

// generate one assistant reply from the current history; returns malloc'd text
// verify that separately tokenized tail matches the suffix of the freshly
// rendered full template past old_n. BPE is context-sensitive across message
// boundaries, so a mismatch means the cached prefix cannot be continued and
// the caller must fall back to a fresh full pass. Returns 1 on match.
static int chat_tail_matches(llm_t * l, const int * dt, int dn, int old_n) {
    if (dn < 0 || old_n < 0) return 0;
    char * full = llm_apply_chat_template(l, 1);
    if (!full) return 0;
    int ok = 0;
    int fcap = (int) strlen(full) * 2 + 64;
    int * ft = (int *) malloc(sizeof(int) * (size_t) fcap);
    if (ft) {
        int fn = llm_tokenize(l, full, 0, ft, fcap);
        ok = (fn == old_n + dn && dn > 0 &&
              memcmp(ft + old_n, dt, sizeof(int) * (size_t) dn) == 0);
        free(ft);
    }
    free(full);
    return ok;
}

char * llm_chat_generate(llm_t * l, int n_predict) {
    chat_stream_t st;
    memset(&st, 0, sizeof st);
    st.l = l;
    if (l->chat_toks && l->chat_from > 0 && chat_can_reuse(l)) {
        char * tail = llm_apply_chat_template_tail(l, l->chat_from, 1);
        if (tail) {
            int cap = (int) strlen(tail) * 2 + 64;
            int * dt = (int *) malloc(sizeof(int) * (size_t) cap);
            int dn = dt ? llm_tokenize(l, tail, 0, dt, cap) : -1;
            free(tail);
            int old_n = l->chat_n;
            if (dn >= 0 && chat_tail_matches(l, dt, dn, old_n)) {
                int * nt = (int *) realloc(l->chat_toks, sizeof(int) * ((size_t) old_n + (size_t) dn));
                if (nt) {
                    l->chat_toks = nt;
                    memcpy(l->chat_toks + old_n, dt, sizeof(int) * (size_t) dn);
                    l->chat_n = old_n + dn;
                    free(dt);
                    llm_generate_from(l, l->chat_toks, l->chat_n, old_n, n_predict, chat_stream_cb, &st);
                } else {
                    // OOM extending the prompt buffer: keep the old prefix and
                    // redo the turn from scratch instead of corrupting history
                    free(dt);
                    LOG_ERROR("llm_chat_generate: OOM, full recompute\n");
                    char * tpl = llm_apply_chat_template(l, 1);
                    if (tpl) {
                        int fcap = (int) strlen(tpl) * 2 + 64;
                        int * toks = (int *) malloc(sizeof(int) * (size_t) fcap);
                        if (toks) {
                            int n = llm_tokenize(l, tpl, 0, toks, fcap);
                            free(tpl);
                            llm_generate(l, toks, n, n_predict, chat_stream_cb, &st);
                            free(l->chat_toks);
                            l->chat_toks = toks;
                            l->chat_n = n;
                        } else {
                            free(tpl);
                        }
                    }
                }
            } else {
                // tail does not round-trip to the full template suffix (or OOM):
                // fresh full template (also repairs a diverged token history)
                free(dt);
                LOG_WARN("llm_chat_generate: tail mismatch, full recompute\n");
                char * tpl = llm_apply_chat_template(l, 1);
                if (tpl) {
                    int fcap = (int) strlen(tpl) * 2 + 64;
                    int * toks = (int *) malloc(sizeof(int) * (size_t) fcap);
                    if (toks) {
                        int n = llm_tokenize(l, tpl, 0, toks, fcap);
                        free(tpl);
                        llm_generate(l, toks, n, n_predict, chat_stream_cb, &st);
                        free(l->chat_toks);
                        l->chat_toks = toks;
                        l->chat_n = n;
                    } else {
                        free(tpl);
                    }
                }
            }
        } else {
            char * tpl = llm_apply_chat_template(l, 1);
            int cap = tpl ? (int) strlen(tpl) * 2 + 64 : 64;
            int * toks = (int *) malloc(sizeof(int) * (size_t) cap);
            int n = (tpl && toks) ? llm_tokenize(l, tpl, 0, toks, cap) : -1;
            free(tpl);
            if (n < 0) { free(toks); free(st.toks); free(st.prev); return NULL; }
            llm_generate(l, toks, n, n_predict, chat_stream_cb, &st);
            free(l->chat_toks);
            l->chat_toks = toks;
            l->chat_n = n;
        }
    } else {
        char * tpl = llm_apply_chat_template(l, 1);
        if (!tpl) { free(st.toks); free(st.prev); return NULL; }
        int cap = (int) strlen(tpl) * 2 + 64;
        int * toks = (int *) malloc(sizeof(int) * (size_t) cap);
        int n = toks ? llm_tokenize(l, tpl, 0, toks, cap) : -1;
        free(tpl);
        if (n < 0) { free(toks); free(st.toks); free(st.prev); return NULL; }
        llm_generate(l, toks, n, n_predict, chat_stream_cb, &st);
        free(l->chat_toks);
        l->chat_toks = toks;
        l->chat_n = n;
    }
    if (st.n > 0) {
        int * nt = (int *) realloc(l->chat_toks, sizeof(int) * ((size_t) l->chat_n + (size_t) st.n));
        if (nt) {
            l->chat_toks = nt;
            memcpy(l->chat_toks + l->chat_n, st.toks, sizeof(int) * (size_t) st.n);
            l->chat_n += st.n;
        }
        // on OOM the reply tokens are dropped from history: the next turn
        // falls back to a validated full recompute (chat_tail_matches fails),
        // so the KV prefix can never silently diverge from chat_toks
    }
    free(st.toks);
    char * reply = st.prev ? st.prev : strdup("");
    char * im_end = strstr(reply, "<|im_end|>");
    if (im_end) *im_end = 0;
    char * eot = strstr(reply, "<|endoftext|>");
    if (eot) *eot = 0;
    char * gemma_eot = strstr(reply, "<end_of_turn>");
    if (gemma_eot) *gemma_eot = 0;
    char * hist_reply = NULL;
    if (llm_get_arch(l) == LLM_ARCH_QWEN35 || llm_get_arch(l) == LLM_ARCH_LFM2) {
        hist_reply = strdup(reply);
    }
    if (llm_get_arch(l) == LLM_ARCH_QWEN35) {
        const char * as = chat_answer_start(reply);
        if (as && as != reply) {
            memmove(reply, as, strlen(as) + 1);
        }
    }
    chat_strip_thinking(reply);
    if (llm_get_arch(l) == LLM_ARCH_QWEN35) {
        size_t rl = strlen(reply);
        while (rl && (reply[rl - 1] == '\n' || reply[rl - 1] == '\r' ||
                      reply[rl - 1] == ' '  || reply[rl - 1] == '\t'))
            reply[--rl] = 0;
        if (rl > 2 && reply[0] == '[' && reply[rl - 1] == ']') reply[0] = 0;
        if (!reply[0]) {
            char * fb = strdup("Нет ответа");
            free(reply);
            reply = fb;
        }
    }
    llm_chat_add(l, "assistant", hist_reply ? hist_reply : reply);
    if (hist_reply) free(hist_reply);
    l->chat_from = l->chat_msg_count;
    return reply;
}

// one chat turn: add the user message and generate a reply. Every request is
// reset to the post system/tools-prompt state first, so a session never
// accumulates history across requests.
char * llm_chat_turn(llm_t * l, const char * user_text, int n_predict) {
    llm_chat_reset(l);
    llm_chat_add(l, "user", user_text);
    return llm_chat_generate(l, n_predict);
}

// ---------------------------------------------------------------------------
// llm session cache: persist the LLM state right after the system / tools
// prompt has been processed into the KV cache, so a restart can skip that
// constant-prompt processing and every request can reset to the same state.
// ---------------------------------------------------------------------------
static int file_stamp(const char * path, uint64_t * size, uint64_t * mtime) {
    if (!path || !path[0]) return -1;
    struct stat st;
    if (stat(path, &st) != 0) return -1;
    *size  = (uint64_t) st.st_size;
    *mtime = (uint64_t) st.st_mtime;
    return 0;
}

// rows allocated for a block's KV cache (full n_ctx or a sliding window)
static int block_win(const llm_t * l, int b) {
    int win = l->blocks[b].win;
    return win > 0 ? win : l->n_ctx;
}

static int cf_wr(FILE * f, const void * p, size_t n) { return fwrite(p, 1, n, f) == n ? 0 : -1; }
static int cf_wr_u32(FILE * f, uint32_t v) { return cf_wr(f, &v, 4); }
static int cf_wr_i32(FILE * f, int32_t v) { return cf_wr(f, &v, 4); }
static int cf_wr_u64(FILE * f, uint64_t v) { return cf_wr(f, &v, 8); }
static int cf_rd(FILE * f, void * p, size_t n) { return fread(p, 1, n, f) == n ? 0 : -1; }
static int cf_rd_u32(FILE * f, uint32_t * v) { return cf_rd(f, v, 4); }
static int cf_rd_i32(FILE * f, int32_t * v) { return cf_rd(f, v, 4); }
static int cf_rd_u64(FILE * f, uint64_t * v) { return cf_rd(f, v, 8); }

static void llm_session_free(llm_session_t * s) {
    if (!s) return;
    if (s->k_cache) {
        for (int b = 0; b < s->n_layer; b++) free(s->k_cache[b]);
        free(s->k_cache);
    }
    if (s->v_cache) {
        for (int b = 0; b < s->n_layer; b++) free(s->v_cache[b]);
        free(s->v_cache);
    }
    free(s->kv_rows);
    if (s->gdn_state) {
        for (int b = 0; b < s->n_layer; b++) free(s->gdn_state[b]);
        free(s->gdn_state);
    }
    if (s->gdn_conv) {
        for (int b = 0; b < s->n_layer; b++) free(s->gdn_conv[b]);
        free(s->gdn_conv);
    }
    if (s->lfm2_state) {
        for (int b = 0; b < s->n_layer; b++) free(s->lfm2_state[b]);
        free(s->lfm2_state);
    }
    free(s->chat_toks);
    for (int i = 0; i < s->n_msg; i++) {
        free(s->msg_role[i]);
        free(s->msg_content[i]);
    }
    free(s);
}

void llm_chat_capture(llm_t * l) {
    if (!l || !chat_can_reuse(l)) return;
    llm_session_free(l->session);
    l->session = NULL;
    llm_session_t * s = (llm_session_t *) calloc(1, sizeof(llm_session_t));
    if (!s) return;
    s->n_layer = l->n_layer;
    s->n_past = l->n_past;
    s->chat_n = l->chat_n;
    s->chat_from = l->chat_from;
    s->chat_msg_count = l->chat_msg_count;
    s->n_msg = l->n_msg;
    if (l->chat_n > 0 && l->chat_toks) {
        s->chat_toks = (int *) malloc(sizeof(int) * (size_t) l->chat_n);
        if (!s->chat_toks) { llm_session_free(s); return; }
        memcpy(s->chat_toks, l->chat_toks, sizeof(int) * (size_t) l->chat_n);
    }
    for (int i = 0; i < l->n_msg; i++) {
        s->msg_role[i] = strdup(l->msg_role[i] ? l->msg_role[i] : "");
        s->msg_content[i] = strdup(l->msg_content[i] ? l->msg_content[i] : "");
        if (!s->msg_role[i] || !s->msg_content[i]) { llm_session_free(s); return; }
    }
    const int kv_dim = l->n_head_kv * l->head_dim;
    const int n_past = l->n_past < 0 ? 0 : l->n_past;
    s->k_cache = (float **) calloc(l->n_layer, sizeof(float *));
    s->v_cache = (float **) calloc(l->n_layer, sizeof(float *));
    s->kv_rows = (int *) calloc(l->n_layer, sizeof(int));
    s->gdn_state = (float **) calloc(l->n_layer, sizeof(float *));
    s->gdn_conv = (float **) calloc(l->n_layer, sizeof(float *));
    s->lfm2_state = (float **) calloc(l->n_layer, sizeof(float *));
    if (!s->k_cache || !s->v_cache || !s->kv_rows || !s->gdn_state || !s->gdn_conv || !s->lfm2_state) {
        llm_session_free(s);
        return;
    }
    for (int b = 0; b < l->n_layer; b++) {
        if (l->blocks[b].is_full_attn) {
            // non-SWA layers index the cache by absolute position; SWA layers
            // wrap at win. When win >= n_past the two coincide, otherwise all
            // win slots are used, so saving rows [0, min(win, n_past)) is exact.
            int win = block_win(l, b);
            int rows = (win >= n_past) ? n_past : win;
            if (rows < 0) rows = 0;
            size_t nf = (size_t) rows * kv_dim;
            s->kv_rows[b] = rows;
            if (nf > 0) {
                s->k_cache[b] = (float *) malloc(nf * sizeof(float));
                s->v_cache[b] = (float *) malloc(nf * sizeof(float));
                if (!s->k_cache[b] || !s->v_cache[b]) { llm_session_free(s); return; }
                memcpy(s->k_cache[b], l->k_cache[b], nf * sizeof(float));
                memcpy(s->v_cache[b], l->v_cache[b], nf * sizeof(float));
            }
        } else if (l->arch == LLM_ARCH_LFM2) {
            size_t nf = (size_t) (l->shortconv_l_cache - 1) * l->n_embd;
            if (nf > 0) {
                s->lfm2_state[b] = (float *) malloc(nf * sizeof(float));
                if (!s->lfm2_state[b]) { llm_session_free(s); return; }
                memcpy(s->lfm2_state[b], l->lfm2_state[b], nf * sizeof(float));
            }
        } else {
            size_t nsf = (size_t) l->ssm_num_v_heads * l->ssm_head_dim * l->ssm_head_dim;
            int conv_dim = 2 * l->ssm_num_k_heads * l->ssm_head_dim + l->ssm_num_v_heads * l->ssm_head_dim;
            size_t ncf = (size_t) (l->ssm_conv_kernel - 1) * conv_dim;
            if (nsf > 0) {
                s->gdn_state[b] = (float *) malloc(nsf * sizeof(float));
                if (!s->gdn_state[b]) { llm_session_free(s); return; }
                memcpy(s->gdn_state[b], l->gdn_state[b], nsf * sizeof(float));
            }
            if (ncf > 0) {
                s->gdn_conv[b] = (float *) malloc(ncf * sizeof(float));
                if (!s->gdn_conv[b]) { llm_session_free(s); return; }
                memcpy(s->gdn_conv[b], l->gdn_conv[b], ncf * sizeof(float));
            }
        }
    }
    l->session = s;
}

// restore the captured post-warmup state into the live model
void llm_chat_reset(llm_t * l) {
    llm_session_t * s = l ? l->session : NULL;
    if (!s) return;
    const int kv_dim = l->n_head_kv * l->head_dim;
    for (int b = 0; b < l->n_layer; b++) {
        if (l->blocks[b].is_full_attn) {
            size_t nf = (size_t) s->kv_rows[b] * kv_dim;
            if (nf > 0) {
                if (s->k_cache[b]) memcpy(l->k_cache[b], s->k_cache[b], nf * sizeof(float));
                if (s->v_cache[b]) memcpy(l->v_cache[b], s->v_cache[b], nf * sizeof(float));
            }
        } else if (l->arch == LLM_ARCH_LFM2) {
            size_t nf = (size_t) (l->shortconv_l_cache - 1) * l->n_embd;
            if (nf > 0 && s->lfm2_state[b]) memcpy(l->lfm2_state[b], s->lfm2_state[b], nf * sizeof(float));
        } else {
            size_t nsf = (size_t) l->ssm_num_v_heads * l->ssm_head_dim * l->ssm_head_dim;
            if (nsf > 0 && s->gdn_state[b]) memcpy(l->gdn_state[b], s->gdn_state[b], nsf * sizeof(float));
            int conv_dim = 2 * l->ssm_num_k_heads * l->ssm_head_dim + l->ssm_num_v_heads * l->ssm_head_dim;
            size_t ncf = (size_t) (l->ssm_conv_kernel - 1) * conv_dim;
            if (ncf > 0 && s->gdn_conv[b]) memcpy(l->gdn_conv[b], s->gdn_conv[b], ncf * sizeof(float));
        }
    }
    l->n_past = s->n_past;
    free(l->chat_toks);
    l->chat_toks = NULL;
    if (s->chat_n > 0 && s->chat_toks) {
        l->chat_toks = (int *) malloc(sizeof(int) * (size_t) s->chat_n);
        if (l->chat_toks) memcpy(l->chat_toks, s->chat_toks, sizeof(int) * (size_t) s->chat_n);
    }
    // rebuild the cache mirror: the restored KV state holds exactly the first
    // n_past tokens of the session token stream (capture invariant: warmup)
    llm_cache_discard(l);
    if (s->n_past > 0 && s->chat_n >= s->n_past && s->chat_toks) {
        int * ct = (int *) malloc(sizeof(int) * (size_t) s->n_past);
        if (ct) {
            memcpy(ct, s->chat_toks, sizeof(int) * (size_t) s->n_past);
            l->cache_toks = ct;
            l->cache_n = s->n_past;
            l->cache_cap = s->n_past;
            l->cache_ok = 1;
        }
    }
    l->chat_n = s->chat_n;
    l->chat_from = s->chat_from;
    l->chat_msg_count = s->chat_msg_count;
    for (int i = 0; i < l->n_msg; i++) {
        free(l->msg_role[i]);
        free(l->msg_content[i]);
        l->msg_role[i] = NULL;
        l->msg_content[i] = NULL;
    }
    l->n_msg = s->n_msg;
    for (int i = 0; i < s->n_msg; i++) {
        l->msg_role[i] = strdup(s->msg_role[i] ? s->msg_role[i] : "user");
        l->msg_content[i] = strdup(s->msg_content[i] ? s->msg_content[i] : "");
    }
    l->n_prompt_tokens = 0;
    l->n_gen_tokens = 0;
    l->n_gen_hist = 0;
}

void llm_session_cache_path(const llm_t * l, char * buf, size_t buf_len) {
    if (!buf || buf_len < 2) return;
    const char * base = (l && l->model_path[0]) ? l->model_path : "model";
    const char * name = base;
    const char * slash = strrchr(base, '/');
    const char * bslash = strrchr(base, '\\');
    if (slash && slash >= name) name = slash + 1;
    if (bslash && bslash >= name) name = bslash + 1;
    char stem[256];
    snprintf(stem, sizeof(stem), "%s", name);
    char * dot = strrchr(stem, '.');
    if (dot && dot != stem) *dot = 0;
    snprintf(buf, buf_len, "llm_session_%s.cache", stem);
}

// persist the current (post-warmup) session state. Header carries the model
// identity (size + mtime) and the rendered warmup token stream so a later
// load can reject caches built from a different model or prompt files.
static int llm_session_save(llm_t * l, const char * path) {
    FILE * f = fopen(path, "wb");
    if (!f) return -1;
    const int kv_dim = l->n_head_kv * l->head_dim;
    int ok = 0;
    int rc = cf_wr(f, "LLMSESS", 8)
        || cf_wr_u32(f, 1)
        || cf_wr_u32(f, (uint32_t) l->arch)
        || cf_wr_u32(f, (uint32_t) l->n_ctx)
        || cf_wr_u32(f, (uint32_t) l->n_layer)
        || cf_wr_u64(f, l->model_size)
        || cf_wr_u64(f, l->model_mtime)
        || cf_wr_i32(f, l->n_past)
        || cf_wr_i32(f, l->chat_n)
        || cf_wr_i32(f, l->chat_from)
        || cf_wr_i32(f, l->chat_msg_count)
        || cf_wr_i32(f, l->n_msg);
    for (int i = 0; !rc && i < l->n_msg; i++) {
        const char * role = l->msg_role[i] ? l->msg_role[i] : "user";
        const char * cont = l->msg_content[i] ? l->msg_content[i] : "";
        size_t rl = strlen(role), cl = strlen(cont);
        rc = cf_wr_u32(f, (uint32_t) rl) || cf_wr(f, role, rl)
          || cf_wr_u32(f, (uint32_t) cl) || cf_wr(f, cont, cl);
    }
    if (!rc) rc = cf_wr_i32(f, l->chat_n);
    if (!rc && l->chat_n > 0 && l->chat_toks)
        rc = cf_wr(f, l->chat_toks, sizeof(int) * (size_t) l->chat_n);
    for (int b = 0; !rc && b < l->n_layer; b++) {
        if (l->blocks[b].is_full_attn) {
            int win = block_win(l, b);
            int rows = (win >= l->n_past) ? l->n_past : win;
            if (rows < 0) rows = 0;
            size_t nf = (size_t) rows * kv_dim;
            rc = cf_wr_i32(f, rows);
            if (!rc && nf > 0)
                rc = cf_wr(f, l->k_cache[b], nf * sizeof(float))
                  || cf_wr(f, l->v_cache[b], nf * sizeof(float));
        } else if (l->arch == LLM_ARCH_LFM2) {
            size_t nf = (size_t) (l->shortconv_l_cache - 1) * l->n_embd;
            if (nf > 0) rc = cf_wr(f, l->lfm2_state[b], nf * sizeof(float));
        } else {
            size_t nsf = (size_t) l->ssm_num_v_heads * l->ssm_head_dim * l->ssm_head_dim;
            int conv_dim = 2 * l->ssm_num_k_heads * l->ssm_head_dim + l->ssm_num_v_heads * l->ssm_head_dim;
            size_t ncf = (size_t) (l->ssm_conv_kernel - 1) * conv_dim;
            if (nsf > 0) rc = cf_wr(f, l->gdn_state[b], nsf * sizeof(float));
            if (!rc && ncf > 0) rc = cf_wr(f, l->gdn_conv[b], ncf * sizeof(float));
        }
    }
    if (!rc) ok = 1;
    fclose(f);
    if (!ok) remove(path);
    return ok ? 0 : -1;
}

// restore a saved session. Accepts the cache only when it belongs to the
// current model (arch / n_ctx / n_layer / size / mtime) and its token stream
// matches the freshly rendered warmup prompt (prompt files unchanged).
// Returns 0 and applies the state on success, -1 otherwise.
static int llm_session_load(llm_t * l, const char * path, const int * want_toks, int want_n) {
    FILE * f = fopen(path, "rb");
    if (!f) return -1;
    char magic[8];
    uint32_t v, arch, nctx, nlayer;
    uint64_t msize, mmtime;
    int32_t n_past, chat_n, chat_from, chat_msg_count, n_msg;
    int ok = 0;
    int * toks = NULL;
    int fail = cf_rd(f, magic, 8) || memcmp(magic, "LLMSESS", 8) != 0;
    if (fail) goto done;
    fail = cf_rd_u32(f, &v) || v != 1
        || cf_rd_u32(f, &arch) || (int) arch != (int) l->arch
        || cf_rd_u32(f, &nctx) || (int) nctx != l->n_ctx
        || cf_rd_u32(f, &nlayer) || (int) nlayer != l->n_layer
        || cf_rd_u64(f, &msize) || msize != l->model_size
        || cf_rd_u64(f, &mmtime) || mmtime != l->model_mtime;
    if (fail) goto done;
    if (cf_rd_i32(f, &n_past) || cf_rd_i32(f, &chat_n) || cf_rd_i32(f, &chat_from) ||
        cf_rd_i32(f, &chat_msg_count) || cf_rd_i32(f, &n_msg)) goto done;
    // baseline invariants: n_past == warmup token count, all messages are
    // system messages and chat_from points right after them
    if (n_past < 0 || n_past > l->n_ctx) goto done;
    if (chat_n != n_past) goto done;
    if (n_msg < 0 || n_msg > LLM_MAX_MSG || n_msg != l->n_msg) goto done;
    if (chat_msg_count != n_msg || chat_from != chat_msg_count) goto done;
    // skip the stored messages (the token stream validates the prompt files)
    for (int i = 0; i < n_msg; i++) {
        uint32_t rl, cl;
        if (cf_rd_u32(f, &rl) || rl > 4096 || fseek(f, rl, SEEK_CUR) != 0) goto done;
        if (cf_rd_u32(f, &cl) || cl > (1u << 30) || fseek(f, cl, SEEK_CUR) != 0) goto done;
    }
    // the warmup prompt must be identical to what the cache was built from
    int32_t stored_n;
    if (cf_rd_i32(f, &stored_n) || stored_n != chat_n || stored_n != want_n) goto done;
    if (stored_n > 0) {
        toks = (int *) malloc(sizeof(int) * (size_t) stored_n);
        if (!toks || cf_rd(f, toks, sizeof(int) * (size_t) stored_n)) goto done;
        if (want_toks && memcmp(toks, want_toks, sizeof(int) * (size_t) stored_n) != 0) goto done;
    }
    // restore the layer caches (non-full-attn LFM2/GDN sizes derive from the model)
    const int kv_dim = l->n_head_kv * l->head_dim;
    for (int b = 0; b < l->n_layer; b++) {
        if (l->blocks[b].is_full_attn) {
            int32_t rows;
            int win = block_win(l, b);
            if (cf_rd_i32(f, &rows) || rows < 0 || rows > win) goto done;
            size_t nf = (size_t) rows * kv_dim;
            if (nf > 0 && (cf_rd(f, l->k_cache[b], nf * sizeof(float)) ||
                           cf_rd(f, l->v_cache[b], nf * sizeof(float)))) goto done;
        } else if (l->arch == LLM_ARCH_LFM2) {
            size_t nf = (size_t) (l->shortconv_l_cache - 1) * l->n_embd;
            if (nf > 0 && cf_rd(f, l->lfm2_state[b], nf * sizeof(float))) goto done;
        } else {
            size_t nsf = (size_t) l->ssm_num_v_heads * l->ssm_head_dim * l->ssm_head_dim;
            int conv_dim = 2 * l->ssm_num_k_heads * l->ssm_head_dim + l->ssm_num_v_heads * l->ssm_head_dim;
            size_t ncf = (size_t) (l->ssm_conv_kernel - 1) * conv_dim;
            if (nsf > 0 && cf_rd(f, l->gdn_state[b], nsf * sizeof(float))) goto done;
            if (ncf > 0 && cf_rd(f, l->gdn_conv[b], ncf * sizeof(float))) goto done;
        }
    }
    // apply the restored chat state; messages stay the currently loaded ones
    free(l->chat_toks);
    l->chat_toks = toks;
    toks = NULL;
    l->chat_n = chat_n;
    l->chat_from = chat_from;
    l->chat_msg_count = chat_msg_count;
    l->n_past = n_past;
    // the on-disk validator guarantees chat_n == n_past here, so the restored
    // token stream doubles as the cache mirror for future continuations
    llm_cache_discard(l);
    if (n_past > 0 && l->chat_toks) {
        int * ct = (int *) malloc(sizeof(int) * (size_t) n_past);
        if (ct) {
            memcpy(ct, l->chat_toks, sizeof(int) * (size_t) n_past);
            l->cache_toks = ct;
            l->cache_n = n_past;
            l->cache_cap = n_past;
            l->cache_ok = 1;
        }
    }
    ok = 1;
done:
    free(toks);
    fclose(f);
    return ok ? 0 : -1;
}

// load the session cache when valid, otherwise warm up and persist the state.
// Either way the warmup state is captured as the per-request reset point.
int llm_chat_warmup_cache(llm_t * l, const char * path) {
    if (!chat_can_reuse(l)) return 0;
    if (l->chat_msg_count == 0) return 0;
    int * toks = NULL; int n = 0;
    if (llm_chat_warmup_tokens(l, &toks, &n) != 0) return 0;
    if (llm_session_load(l, path, toks, n) == 0) {
        free(toks);
        LOG_INFO("llm session cache loaded: %s (%d tokens)", path, n);
        llm_chat_capture(l);
        return 0;
    }
    free(toks);
    llm_chat_warmup(l);
    if (llm_session_save(l, path) == 0)
        LOG_INFO("llm session cache saved: %s (%d tokens)", path, l->chat_n);
    else
        LOG_WARN("cannot write llm session cache '%s'", path);
    return 0;
}

// ---------------------------------------------------------------------------
// init / free
// ---------------------------------------------------------------------------
llm_t * llm_init(const char * model_path, const llm_params_t * params) {
    double t0 = now_ms();
    // initialize the ggml SIMD dispatch tables (f16->f32 lookup etc.) before
    // the direct quantized GEMV path uses them
    ggml_cpu_init();
    llm_t * l = (llm_t *) calloc(1, sizeof(llm_t));
    if (!l) return NULL;
#ifdef GGML_USE_VULKAN
    // GPU backend only when explicitly requested via --device gpu; the device
    // number selects the Vulkan<N> backend (0, 1, ...). Falls back to CPU.
    if (!getenv("LLM_CPU_ONLY") && params && params->use_gpu) {
        char name[32];
        snprintf(name, sizeof(name), "Vulkan%d", params->gpu_index);
        l->vk = ggml_backend_init_by_name(name, NULL);
        if (!l->vk) {
            // requested device not found: fall back to the first GPU, else CPU
            l->vk = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, NULL);
            if (!l->vk) l->vk = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU, NULL);
        }
        fprintf(stderr, "[llm] vulkan backend: %s\n", l->vk ? "OK" : "unavailable");
        if (!l->vk || strcmp(ggml_backend_name(l->vk), name) != 0)
            fprintf(stderr, "[llm] note: running on '%s' (requested '%s')\n",
                    l->vk ? ggml_backend_name(l->vk) : "CPU", name);
    }
#endif
    llm_params_default(&l->params);
    if (params) l->params = *params;
    if (model_path) snprintf(l->model_path, sizeof(l->model_path), "%s", model_path);
    file_stamp(model_path, &l->model_size, &l->model_mtime);
    l->n_ctx = l->params.n_ctx > 0 ? l->params.n_ctx : 4096;
    l->n_threads = l->params.n_threads > 0 ? l->params.n_threads : 4;
    if (gguf_load(&l->file, model_path) != 0) {
        LOG_ERROR("failed to load model: %s\n", model_path);
        llm_free(l);
        return NULL;
    }
    gguf_read_meta(&l->file, l);
    tok_build_hashes(&l->tok);
    llm_act_build(); // precompute the sigmoid / gelu-tanh lookup tables
    if (llm_load_model(l) != 0) {
        LOG_ERROR("failed to load model weights\n");
        llm_free(l);
        return NULL;
    }
    l->t_load_ms = now_ms() - t0;
    LOG_INFO("model loaded: %d main layers, embd %d, ctx %d (%.1f s)\n",
             l->n_layer_main, l->n_embd, l->n_ctx, l->t_load_ms / 1000.0);
    return l;
}

void llm_free(llm_t * l) {
    if (!l) return;
    if (l->tok.tokens) {
        for (int i = 0; i < l->tok.n_vocab; i++) free(l->tok.tokens[i]);
        free(l->tok.tokens);
    }
    free(l->tok.token_type);
    free(l->tok.char2id);
    if (l->tok.skey) {
        for (int i = 0; i < l->tok.scap; i++) free(l->tok.skey[i]);
        free(l->tok.skey);
    }
    free(l->tok.sval);
    if (l->tok.special_cps) {
        for (int i = 0; i < l->tok.n_special; i++) free(l->tok.special_cps[i]);
        free(l->tok.special_cps);
    }
    free(l->tok.special_len);
    free(l->tok.special_id);
    free(l->tok.mkey);
    free(l->tok.mval);
    free(l->tok.scores);
    if (l->tok.merges_raw) {
        for (int i = 0; i < l->tok.n_merges_raw; i++) free(l->tok.merges_raw[i]);
        free(l->tok.merges_raw);
    }
    free(l->tok.chat_template);
    free(l->blocks);
    if (l->k_cache) {
        for (int b = 0; b < l->n_layer; b++) free(l->k_cache[b]);
        free(l->k_cache);
    }
    if (l->v_cache) {
        for (int b = 0; b < l->n_layer; b++) free(l->v_cache[b]);
        free(l->v_cache);
    }
    if (l->gdn_state) {
        for (int b = 0; b < l->n_layer; b++) free(l->gdn_state[b]);
        free(l->gdn_state);
    }
    if (l->gdn_conv) {
        for (int b = 0; b < l->n_layer; b++) free(l->gdn_conv[b]);
        free(l->gdn_conv);
    }
    if (l->lfm2_state) {
        for (int b = 0; b < l->n_layer; b++) free(l->lfm2_state[b]);
        free(l->lfm2_state);
    }
    free(l->cache_toks);
    if (l->ctx_w) { ggml_free(l->ctx_w); free(l->w_buf); free(l->f32_w_buf); }
    if (l->ctx_compute) { ggml_free(l->ctx_compute); free(l->compute_buf); }
    if (l->ctx_graph) { ggml_free(l->ctx_graph); free(l->graph_buf); }
    free(l->rope_cos);
    free(l->rope_sin);
    free(l->rope_cos_swa);
    free(l->rope_sin_swa);
    free(l->rope_freqs);
    free(l->logits);
    free(l->mm_xbuf);
    free(l->mm_qbuf);
    free(l->scratch);
    free(l->bx); free(l->br); free(l->bout); free(l->bo);
    free(l->bq); free(l->bk); free(l->bv); free(l->bfg); free(l->bfu); free(l->bscores);
    free(l->bq8);
    free(l->bvin);
    free(l->scores_mt);
#ifdef GGML_USE_VULKAN
    free(l->embd_raw);
    if (l->vk_cbuf)   { ggml_backend_buffer_free(l->vk_cbuf);   l->vk_cbuf   = NULL; }
    if (l->vk_wbuf)   { ggml_backend_buffer_free(l->vk_wbuf);   l->vk_wbuf   = NULL; }
    if (l->vk_ctx)    { ggml_free(l->vk_ctx); l->vk_ctx = NULL; }
    if (l->vk_gctx)   { ggml_free(l->vk_gctx); l->vk_gctx = NULL; }
    if (l->vk)        { ggml_backend_free(l->vk); l->vk = NULL; }
#endif
    free(l->work);
    free(l->cand);
    free(l->gen_hist);
    for (int i = 0; i < l->n_msg; i++) {
        free(l->msg_role[i]);
        free(l->msg_content[i]);
    }
    if (l->tools) {
        for (int i = 0; i < l->n_tools; i++) {
            free((void *) l->tools[i].name);
            free((void *) l->tools[i].description);
            free((void *) l->tools[i].parameters);
        }
        free(l->tools);
    }
    llm_session_free(l->session);
    free(l->chat_toks);
    if (l->file.buf) munmap_file(l->file.buf, l->file.buf_size);
    free(l->file.tensors);
    free(l);
}