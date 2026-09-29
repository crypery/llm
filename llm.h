/*
 * Git: https://github.com/crypery
 * Author: https://crypery.com
 * License: GNU AGPL v3 (Affero GPL)
 */

// llm.h - CPU LLM inference module (ggml-backed) for Qwen3-Next style hybrid
// models (Gated DeltaNet linear-attention + full-attention blocks). All heavy
// math is done with plain ggml_mul_mat only.
#ifndef LLM_H
#define LLM_H

#include "common.h"

// supported model architectures
typedef enum {
    LLM_ARCH_QWEN35 = 0,  // gated-delta-net hybrid (default path)
    LLM_ARCH_GEMMA3  = 1, // gemma3 dense (all full-attention, q/k norm, SWA)
    LLM_ARCH_LFM2    = 2  // lfm2 dense (attention + shortconv hybrid, q/k norm)
} llm_arch_t;

typedef struct llm_s llm_t;

typedef struct {
    float temp;
    int   top_k;
    float top_p;
} llm_sampler_t;

typedef struct {
    int   n_ctx;        // context window
    int   n_threads;    // cpu threads
    int   n_predict;    // max tokens per reply
    llm_sampler_t sampler;
    const char * lang;  // response language hint (e.g. "ru")
#ifdef GGML_USE_VULKAN
    int   use_gpu;      // 1 = run matmuls on a Vulkan GPU, 0 = CPU only (default)
    int   gpu_index;    // Vulkan device number (0, 1, ...) -> backend "Vulkan<n>"
#endif
} llm_params_t;

void llm_params_default(llm_params_t * p);

// init / free
llm_t * llm_init(const char * model_path, const llm_params_t * params);
void    llm_free(llm_t * l);

// architecture
llm_arch_t llm_get_arch(const llm_t * l);

// tokenization
int   llm_vocab_size(const llm_t * l);
int   llm_tokenize(const llm_t * l, const char * text, int add_bos, int * tokens, int n_max);
char * llm_detokenize(const llm_t * l, const int * tokens, int n);

// chat / template
void  llm_add_message(llm_t * l, const char * role, const char * content);
void  llm_clear_messages(llm_t * l);
char * llm_apply_chat_template(llm_t * l, int add_generation_prompt);
// render only the tail of the conversation: close the turn of message[from-1]
// and append messages[from .. n_msg-1] plus the generation prompt. Used to
// extend a token stream incrementally (from == 0 means the whole conversation).
char * llm_apply_chat_template_tail(llm_t * l, int from, int add_generation_prompt);

// ---------------------------------------------------------------------------
// high-level chat helpers (dialog history + KV-cache reuse across turns)
// ---------------------------------------------------------------------------
// add a message and track the conversation turn count (for the incremental KV tail)
void  llm_chat_add(llm_t * l, const char * role, const char * content);
// read an agent prompt file and install it as the (single) system message
int   llm_chat_load_agent(llm_t * l, const char * path, const char * lang);
// pre-process the constant system prompt into the KV cache right after init
void  llm_chat_warmup(llm_t * l);
// generate one assistant reply from the current history; returns malloc'd text
char * llm_chat_generate(llm_t * l, int n_predict);
// one chat turn: add a user message and generate the reply
char * llm_chat_turn(llm_t * l, const char * user_text, int n_predict);

// ---------------------------------------------------------------------------
// llm session cache (system / tools prompt pre-processed into the KV cache)
// ---------------------------------------------------------------------------
// Derive the default on-disk session-cache file name from the model file name,
// e.g. "lfm2-4B.gguf" -> "llm_session_lfm2-4B.cache".
void llm_session_cache_path(const llm_t * l, char * buf, size_t buf_len);
// Load the session cache (validated against the current model and the rendered
// system/tools prompt) or, when missing/stale, compute the warmup state and
// persist it. Captures the resulting state as the per-request reset point.
int  llm_chat_warmup_cache(llm_t * l, const char * path);
// Remember the current state as the per-request reset point (done automatically
// by llm_chat_warmup_cache; exposed for manual control).
void llm_chat_capture(llm_t * l);
// Restore the reset point (the state right after the system/tools prompt), so
// the next request starts from a clean session. Called by llm_chat_turn.
void llm_chat_reset(llm_t * l);

// ---------------------------------------------------------------------------
// tool / function calling
// ---------------------------------------------------------------------------
// A tool the model may call. parameters is a JSON Schema string describing the
// argument object (e.g. "{\"type\":\"object\",\"properties\":{...}}").
typedef struct {
    const char * name;         // function name
    const char * description;  // short human-readable description
    const char * parameters;   // JSON Schema for the arguments (JSON string)
} llm_tool_t;

// A tool call parsed from the model output.
typedef struct {
    char   name[128];  // function name the model requested
    char * arguments;  // raw JSON object of the arguments (malloc'd; llm_tool_call_free)
} llm_tool_call_t;

// install the tool list used by llm_apply_chat_template (copies the array)
void llm_set_tools(llm_t * l, const llm_tool_t * tools, int n_tools);

// parse the first tool call from model text. Accepts the "<tool_call> ... "
// </tool_call>" block (qwen3/gemma3) or a plain {"name":...,"arguments":{...}}
// JSON object (llama). Returns 0 and fills *out on success, -1 if none.
int  llm_parse_tool_call(const char * text, llm_tool_call_t * out);
void llm_tool_call_free(llm_tool_call_t * c);

// generation: streams each generated token through cb(id, user_data);
// cb returns 0 to continue, non-zero to stop early. Returns token count.
typedef int (*llm_token_cb)(int token_id, void * user_data);
int llm_generate(llm_t * l, const int * prompt_tokens, int n_prompt,
                 int n_max_tokens, llm_token_cb cb, void * user_data);

// continue generation reusing the KV cache: prompt_tokens[0..n_past-1] must be
// identical to what was processed before; only prompt_tokens[n_past..] are new.
int llm_generate_from(llm_t * l, const int * prompt_tokens, int n_prompt, int n_past,
                      int n_max_tokens, llm_token_cb cb, void * user_data);

// stats
void llm_print_stats(const llm_t * l);

#endif
