// Copyright 2025-current Getnamo.

#include "llu-internal.h"

#include "common.h"
#include "speculative.h"

#include <memory>

struct llu_speculative {
    common_params_speculative params; // kept for the lifetime of spec
    common_speculative      * spec = nullptr;
    common_batch              batch;  // target-context batch, also fed to the draft side
    llama_tokens              prompt; // reused copies of the caller's token buffers
    llama_tokens              draft;

    ~llu_speculative() {
        if (spec) {
            common_speculative_free(spec);
        }
    }
};

extern "C" {

llu_speculative_params llu_speculative_default_params(void) {
    const common_params_speculative_draft d;

    llu_speculative_params p;
    p.struct_size = (int32_t) sizeof(llu_speculative_params);
    p.types       = 0;
    p.n_max       = d.n_max;
    p.n_min       = d.n_min;
    p.p_min       = d.p_min;
    p.ctx_tgt     = nullptr;
    p.ctx_dft     = nullptr;
    return p;
}

llu_speculative * llu_speculative_init(const llu_speculative_params * params) {
    return llu::guarded<llu_speculative *>(nullptr, "initializing speculative decoding", [&]() -> llu_speculative * {
        if (!llu::check_struct_size(params, "llu_speculative_params")) {
            return nullptr;
        }
        if (params->ctx_tgt == nullptr || params->types == 0) {
            llu::set_error("needs a target context and at least one speculative type");
            return nullptr;
        }
        if ((params->types & (LLU_SPECULATIVE_DRAFT_MODEL | LLU_SPECULATIVE_MTP)) && params->ctx_dft == nullptr) {
            llu::set_error("draft-model and MTP speculation need a draft context");
            return nullptr;
        }

        std::unique_ptr<llu_speculative> result(new llu_speculative());
        common_params_speculative & sp = result->params;
        sp.types.clear();
        if (params->types & LLU_SPECULATIVE_NGRAM) {
            sp.types.push_back(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K);
        }
        if (params->types & LLU_SPECULATIVE_DRAFT_MODEL) {
            sp.types.push_back(COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE);
        }
        if (params->types & LLU_SPECULATIVE_MTP) {
            sp.types.push_back(COMMON_SPECULATIVE_TYPE_DRAFT_MTP);
        }
        sp.draft.n_max   = params->n_max;
        sp.draft.n_min   = params->n_min;
        sp.draft.p_min   = params->p_min;
        sp.draft.ctx_tgt = params->ctx_tgt;
        sp.draft.ctx_dft = params->ctx_dft;

        result->spec = common_speculative_init(sp, 1);
        if (result->spec == nullptr) {
            llu::set_error("common_speculative_init failed");
            return nullptr;
        }
        result->batch = common_batch(params->ctx_tgt);
        result->draft.reserve(params->n_max > 0 ? params->n_max : 16);
        return result.release();
    });
}

void llu_speculative_free(llu_speculative * spec) {
    delete spec;
}

bool llu_speculative_are_compatible(const llama_model * model_tgt, const llama_model * model_dft) {
    return llu::guarded<bool>(false, "checking draft compatibility", [&]() -> bool {
        return model_tgt && model_dft && common_speculative_are_compatible(model_tgt, model_dft);
    });
}

void llu_speculative_begin(llu_speculative * spec, const llama_token * prompt, int32_t n_prompt) {
    llu::guarded_void("starting speculation", [&]() {
        if (spec == nullptr) {
            return;
        }
        spec->prompt.assign(prompt, prompt + (prompt ? std::max(0, n_prompt) : 0));
        common_speculative_begin(spec->spec, 0, spec->prompt);
    });
}

int32_t llu_speculative_decode(
        llu_speculative   * spec,
        const llama_token * tokens,
        int32_t             n_tokens,
        llama_pos           pos0,
        bool                all_outputs,
        bool              * draft_ok) {
    if (draft_ok) {
        *draft_ok = false;
    }
    return llu::guarded<int32_t>(-1, "decoding", [&]() -> int32_t {
        if (spec == nullptr || tokens == nullptr || n_tokens <= 0) {
            llu::set_error("invalid arguments");
            return -1;
        }
        common_batch & batch = spec->batch;
        batch.clear();
        for (int32_t i = 0; i < n_tokens; i++) {
            batch.add(tokens[i], pos0 + i, 0, all_outputs || i == n_tokens - 1);
        }

        const int32_t ret = llama_process(spec->params.draft.ctx_tgt, LLAMA_PROCESS_TYPE_DECODE, batch.get());
        if (ret != 0) {
            return ret;
        }
        const bool ok = common_speculative_process(spec->spec, batch);
        if (draft_ok) {
            *draft_ok = ok;
        }
        return 0;
    });
}

int32_t llu_speculative_draft(
        llu_speculative   * spec,
        const llama_token * prompt,
        int32_t             n_prompt,
        llama_token         id_last,
        llama_pos           pos0,
        int32_t             n_max,
        llama_token       * out,
        int32_t             out_cap) {
    return llu::guarded<int32_t>(-1, "drafting", [&]() -> int32_t {
        if (spec == nullptr || out == nullptr || out_cap < 0 || (prompt == nullptr && n_prompt > 0)) {
            llu::set_error("invalid arguments");
            return -1;
        }
        if (n_max <= 0 || out_cap == 0) {
            return 0;
        }
        spec->prompt.assign(prompt, prompt + std::max(0, n_prompt));
        spec->draft.clear();

        common_speculative_draft_params & dp = common_speculative_get_draft_params(spec->spec, 0);
        dp.drafting = true;
        dp.n_max    = n_max;
        dp.pos0     = pos0;
        dp.id_last  = id_last;
        dp.prompt   = &spec->prompt;
        dp.result   = &spec->draft;
        common_speculative_draft(spec->spec);

        const int32_t n = (int32_t) std::min({ spec->draft.size(), (size_t) n_max, (size_t) out_cap });
        std::copy(spec->draft.begin(), spec->draft.begin() + n, out);
        return n;
    });
}

void llu_speculative_accept(llu_speculative * spec, int32_t n_accepted) {
    llu::guarded_void("accepting the draft", [&]() {
        if (spec) {
            common_speculative_accept(spec->spec, 0, (uint16_t) std::max(0, n_accepted));
        }
    });
}

} // extern "C"
