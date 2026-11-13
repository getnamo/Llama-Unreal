// Copyright 2025-current Getnamo.
//
// llama-unreal-chat: a small C API over llama.cpp's Jinja chat templating (common/chat.h).
//
// llama-common's chat API is C++ and returns std::string / std::vector by value. Inside Unreal those
// objects would be freed by the module's allocator (mimalloc) after llama-common.dll allocated them on
// the CRT heap, which crashes. This library is built with llama.cpp (see llama-unreal-chat.cmake), so all
// of that stays on one heap, and callers only pass C structs and their own output buffers.

#pragma once

#include "llama.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef _WIN32
#    ifdef LLU_CHAT_BUILD
#        define LLU_CHAT_API __declspec(dllexport)
#    else
#        define LLU_CHAT_API __declspec(dllimport)
#    endif
#else
#    define LLU_CHAT_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

struct llu_chat_templates;

// Parses a chat template with llama.cpp's Jinja engine: tmpl_override if non-empty, otherwise the
// model's tokenizer.chat_template. BOS/EOS handling follows the model's vocab.
// Returns nullptr on failure, see llu_chat_last_error().
LLU_CHAT_API struct llu_chat_templates * llu_chat_templates_init(const struct llama_model * model, const char * tmpl_override);

LLU_CHAT_API void llu_chat_templates_free(struct llu_chat_templates * tmpls);

// Renders the conversation, same contract as llama_chat_apply_template: writes up to length bytes
// (not NUL-terminated) and returns the full length of the prompt, which can exceed length (call again
// with a bigger buffer). enable_thinking is passed to templates that read it. Returns < 0 on error.
LLU_CHAT_API int32_t llu_chat_templates_apply(
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
LLU_CHAT_API bool llu_chat_templates_thinking_tags(
        const struct llu_chat_templates * tmpls,
        char                            * start_tag,
        int32_t                           start_len,
        char                            * end_tag,
        int32_t                           end_len);

// Reason for the last failure on this thread, "" if none. Valid until the next call on this thread.
LLU_CHAT_API const char * llu_chat_last_error(void);

#ifdef __cplusplus
}
#endif
