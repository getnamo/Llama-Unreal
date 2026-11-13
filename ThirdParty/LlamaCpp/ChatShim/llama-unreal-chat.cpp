// Copyright 2025-current Getnamo.

#include "llama-unreal-chat.h"

#include "chat.h"

#include <cstring>
#include <exception>
#include <string>

struct llu_chat_templates {
    common_chat_templates_ptr tmpls;
};

static thread_local std::string llu_last_error;

static void llu_copy_out(const std::string & src, char * dst, int32_t length, bool terminate) {
    if (dst == nullptr || length <= 0) {
        return;
    }
    const size_t n = terminate ? std::min(src.size(), (size_t) length - 1) : std::min(src.size(), (size_t) length);
    std::memcpy(dst, src.data(), n);
    if (terminate) {
        dst[n] = '\0';
    }
}

extern "C" {

llu_chat_templates * llu_chat_templates_init(const llama_model * model, const char * tmpl_override) {
    llu_last_error.clear();
    try {
        common_chat_templates_ptr tmpls = common_chat_templates_init(model, tmpl_override ? tmpl_override : "");
        if (!tmpls) {
            llu_last_error = "common_chat_templates_init returned null";
            return nullptr;
        }
        return new llu_chat_templates{ std::move(tmpls) };
    } catch (const std::exception & e) {
        llu_last_error = e.what();
    } catch (...) {
        llu_last_error = "unknown exception parsing the chat template";
    }
    return nullptr;
}

void llu_chat_templates_free(llu_chat_templates * tmpls) {
    delete tmpls;
}

int32_t llu_chat_templates_apply(
        const llu_chat_templates * tmpls,
        const llama_chat_message * chat,
        size_t                     n_msg,
        bool                       add_ass,
        bool                       enable_thinking,
        char                     * buf,
        int32_t                    length) {
    llu_last_error.clear();
    if (tmpls == nullptr || (chat == nullptr && n_msg > 0)) {
        llu_last_error = "invalid arguments";
        return -1;
    }
    try {
        common_chat_templates_inputs inputs;
        inputs.use_jinja             = true;
        inputs.add_generation_prompt = add_ass;
        inputs.enable_thinking       = enable_thinking;
        inputs.messages.reserve(n_msg);
        for (size_t i = 0; i < n_msg; i++) {
            common_chat_msg msg;
            msg.role    = chat[i].role ? chat[i].role : "";
            msg.content = chat[i].content ? chat[i].content : "";
            inputs.messages.push_back(std::move(msg));
        }

        const common_chat_params params = common_chat_templates_apply(tmpls->tmpls.get(), inputs);
        const std::string prompt = params.prompt.str();
        if (prompt.size() > (size_t) INT32_MAX) {
            llu_last_error = "rendered prompt too large";
            return -1;
        }
        llu_copy_out(prompt, buf, length, /*terminate*/ false);
        return (int32_t) prompt.size();
    } catch (const std::exception & e) {
        llu_last_error = e.what();
    } catch (...) {
        llu_last_error = "unknown exception applying the chat template";
    }
    return -1;
}

bool llu_chat_templates_thinking_tags(
        const llu_chat_templates * tmpls,
        char                     * start_tag,
        int32_t                    start_len,
        char                     * end_tag,
        int32_t                    end_len) {
    llu_last_error.clear();
    llu_copy_out("", start_tag, start_len, true);
    llu_copy_out("", end_tag, end_len, true);
    if (tmpls == nullptr) {
        return false;
    }
    try {
        // Same probe as common_chat_templates_support_enable_thinking, keeping the tags it reports
        common_chat_templates_inputs inputs;
        common_chat_msg msg;
        msg.role    = "user";
        msg.content = "test";
        inputs.messages              = { msg };
        inputs.enable_thinking       = true;
        inputs.add_generation_prompt = true;
        inputs.reasoning_format      = COMMON_REASONING_FORMAT_DEEPSEEK;

        const common_chat_params params = common_chat_templates_apply(tmpls->tmpls.get(), inputs);
        if (!params.supports_thinking) {
            return false;
        }
        llu_copy_out(params.thinking_start_tag, start_tag, start_len, true);
        llu_copy_out(params.thinking_end_tags.empty() ? std::string() : params.thinking_end_tags.front(), end_tag, end_len, true);
        return true;
    } catch (const std::exception & e) {
        llu_last_error = e.what();
    } catch (...) {
        llu_last_error = "unknown exception probing the chat template";
    }
    return false;
}

const char * llu_chat_last_error(void) {
    return llu_last_error.c_str();
}

} // extern "C"
