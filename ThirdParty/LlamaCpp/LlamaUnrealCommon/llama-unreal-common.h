// Copyright 2025-current Getnamo.
//
// llama-unreal-common: the C API Llama-Unreal uses for everything it needs from llama.cpp's
// llama-common library (chat templates, sampling, speculative decoding).
//
// llama-common's API is C++ and passes std::string / std::vector across the DLL boundary. Inside Unreal
// those objects would be freed by the module's allocator (mimalloc) after llama-common.dll allocated them
// on the CRT heap, which crashes. This library is built with llama.cpp (see llama-unreal-common.cmake), so
// all of that stays on one heap. Callers only pass C structs, opaque handles and their own output buffers.
//
// Conventions: functions don't throw; failures return NULL / a negative value and llu_last_error() says
// why. Param structs carry struct_size, set by their *_default_params() function, so a stale library
// and header pair is detected instead of misread.

#pragma once

#include "llama.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef _WIN32
#    ifdef LLU_BUILD
#        define LLU_API __declspec(dllexport)
#    else
#        define LLU_API __declspec(dllimport)
#    endif
#else
#    define LLU_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Reason for the last failure on this thread, "" if none. Valid until the next call on this thread.
LLU_API const char * llu_last_error(void);

//
// Chat templates (llama.cpp's Jinja engine)
//

struct llu_chat_templates;

// Parses a chat template with llama.cpp's Jinja engine: tmpl_override if non-empty, otherwise the
// model's tokenizer.chat_template. BOS/EOS handling follows the model's vocab. Returns NULL on failure.
LLU_API struct llu_chat_templates * llu_chat_templates_init(const struct llama_model * model, const char * tmpl_override);

LLU_API void llu_chat_templates_free(struct llu_chat_templates * tmpls);

// Renders the conversation, same contract as llama_chat_apply_template: writes up to length bytes
// (not NUL-terminated) and returns the full length of the prompt, which can exceed length (call again
// with a bigger buffer). enable_thinking is passed to templates that read it. Returns < 0 on error.
LLU_API int32_t llu_chat_templates_apply(
        const struct llu_chat_templates * tmpls,
        const struct llama_chat_message * chat,
        size_t                            n_msg,
        bool                              add_ass,
        bool                              enable_thinking,
        char                            * buf,
        int32_t                           length);

// Whether the template switches reasoning on and off through enable_thinking. If so, the tag that
// opens a reasoning block and the first tag that closes it are written NUL-terminated (truncated
// to fit) when the buffers are non-null.
LLU_API bool llu_chat_templates_thinking_tags(
        const struct llu_chat_templates * tmpls,
        char                            * start_tag,
        int32_t                           start_len,
        char                            * end_tag,
        int32_t                           end_len);

//
// Sampling (llama.cpp's common sampler)
//

struct llu_sampler;

struct llu_sampler_params {
    int32_t  struct_size;     // sizeof(struct llu_sampler_params)
    uint32_t seed;            // LLAMA_DEFAULT_SEED for random
    float    temp;
    int32_t  top_k;
    float    top_p;
    float    min_p;
    float    typ_p;
    int32_t  penalty_last_n;
    float    penalty_repeat;
    float    penalty_freq;
    float    penalty_present;
    int32_t  mirostat;        // 0 = off, 1 = v1, 2 = v2
    float    mirostat_tau;
    float    mirostat_eta;
    const char * grammar;     // GBNF with a "root" rule, NULL or "" for none
};

// llama.cpp's defaults for its common sampler
LLU_API struct llu_sampler_params llu_sampler_default_params(void);

// Returns NULL on failure (e.g. an invalid grammar)
LLU_API struct llu_sampler * llu_sampler_init(const struct llama_model * model, const struct llu_sampler_params * params);

LLU_API void llu_sampler_free(struct llu_sampler * smpl);

// Samples from output row idx of the last decode (-1 = last). Returns LLAMA_TOKEN_NULL on error.
LLU_API llama_token llu_sampler_sample(struct llu_sampler * smpl, struct llama_context * ctx, int32_t idx);

// is_generated: the token was produced by sampling (advances the grammar and penalty history)
LLU_API void llu_sampler_accept(struct llu_sampler * smpl, llama_token token, bool is_generated);

// Verifies a draft: samples output rows 0..n_draft of the last decode (which held [last, draft...]) and
// accepts the tokens while they match draft[0..n_draft), followed by the first sampled token that
// doesn't (or the token after a fully accepted draft). Writes them to out, which holds n_draft + 1,
// and returns how many (>= 1), < 0 on error.
LLU_API int32_t llu_sampler_sample_and_accept_n(
        struct llu_sampler   * smpl,
        struct llama_context * ctx,
        const llama_token    * draft,
        int32_t                n_draft,
        llama_token          * out);

//
// Speculative decoding (llama.cpp's common_speculative), single sequence (seq 0)
//

struct llu_speculative;

enum llu_speculative_type {
    LLU_SPECULATIVE_NGRAM       = 1 << 0, // n-gram lookup in the context (ngram-map-k), tried first
    LLU_SPECULATIVE_DRAFT_MODEL = 1 << 1, // separate small draft model (ctx_dft)
    LLU_SPECULATIVE_MTP         = 1 << 2, // the target model's own MTP heads (ctx_dft is an MTP context)
};

struct llu_speculative_params {
    int32_t  struct_size;     // sizeof(struct llu_speculative_params)
    uint32_t types;           // llu_speculative_type flags
    int32_t  n_max;           // maximum tokens per draft
    int32_t  n_min;           // drafts shorter than this are dropped
    float    p_min;           // minimum draft-model probability to keep drafting
    struct llama_context * ctx_tgt;
    struct llama_context * ctx_dft; // draft-model or MTP context, NULL for n-gram only
};

LLU_API struct llu_speculative_params llu_speculative_default_params(void);

// Returns NULL on failure
LLU_API struct llu_speculative * llu_speculative_init(const struct llu_speculative_params * params);

LLU_API void llu_speculative_free(struct llu_speculative * spec);

// Same tokenizer as far as speculation needs (vocab type, BOS/EOS, matching token texts)
LLU_API bool llu_speculative_are_compatible(const struct llama_model * model_tgt, const struct llama_model * model_dft);

// Start of a reply: prompt holds the tokens currently in the target KV
LLU_API void llu_speculative_begin(struct llu_speculative * spec, const llama_token * prompt, int32_t n_prompt);

// Decodes tokens on ctx_tgt at positions pos0.. and feeds the same batch to the draft side.
// all_outputs: logits for every token (to verify a draft), otherwise only for the last one.
// Returns llama_process()'s result; *draft_ok (optional) is false if the draft side failed to process it.
LLU_API int32_t llu_speculative_decode(
        struct llu_speculative * spec,
        const llama_token      * tokens,
        int32_t                  n_tokens,
        llama_pos                pos0,
        bool                     all_outputs,
        bool                   * draft_ok);

// Drafts up to n_max tokens following id_last, which will be decoded at pos0; prompt holds the tokens
// in the target KV. Writes them to out (capacity out_cap) and returns how many (0 = no proposal),
// < 0 on error.
LLU_API int32_t llu_speculative_draft(
        struct llu_speculative * spec,
        const llama_token      * prompt,
        int32_t                  n_prompt,
        llama_token              id_last,
        llama_pos                pos0,
        int32_t                  n_max,
        llama_token            * out,
        int32_t                  out_cap);

// After verification: how many drafted tokens were kept
LLU_API void llu_speculative_accept(struct llu_speculative * spec, int32_t n_accepted);

//
// Misc
//

// embd_norm: -1 none, 0 max absolute, 1 taxicab, 2 euclidean, > 2 p-norm
LLU_API void llu_embd_normalize(const float * inp, float * out, int32_t n, int32_t embd_norm);

#ifdef __cplusplus
}
#endif
