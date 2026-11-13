// Copyright 2025-current Getnamo.

#include "llu-internal.h"

#include "common.h"
#include "sampling.h"

struct llu_sampler {
    common_sampler * smpl = nullptr;
    llama_tokens     draft; // reused by sample_and_accept_n

    ~llu_sampler() {
        if (smpl) {
            common_sampler_free(smpl);
        }
    }
};

extern "C" {

llu_sampler_params llu_sampler_default_params(void) {
    const common_params_sampling d;

    llu_sampler_params p;
    p.struct_size     = (int32_t) sizeof(llu_sampler_params);
    p.seed            = d.seed;
    p.temp            = d.temp;
    p.top_k           = d.top_k;
    p.top_p           = d.top_p;
    p.min_p           = d.min_p;
    p.typ_p           = d.typ_p;
    p.penalty_last_n  = d.penalty_last_n;
    p.penalty_repeat  = d.penalty_repeat;
    p.penalty_freq    = d.penalty_freq;
    p.penalty_present = d.penalty_present;
    p.mirostat        = d.mirostat;
    p.mirostat_tau    = d.mirostat_tau;
    p.mirostat_eta    = d.mirostat_eta;
    p.grammar         = nullptr;
    return p;
}

llu_sampler * llu_sampler_init(const llama_model * model, const llu_sampler_params * params) {
    return llu::guarded<llu_sampler *>(nullptr, "creating the sampler", [&]() -> llu_sampler * {
        if (!llu::check_struct_size(params, "llu_sampler_params")) {
            return nullptr;
        }

        common_params_sampling sp;
        sp.seed            = params->seed;
        sp.temp            = params->temp;
        sp.top_k           = params->top_k;
        sp.top_p           = params->top_p;
        sp.min_p           = params->min_p;
        sp.typ_p           = params->typ_p;
        sp.penalty_last_n  = params->penalty_last_n;
        sp.penalty_repeat  = params->penalty_repeat;
        sp.penalty_freq    = params->penalty_freq;
        sp.penalty_present = params->penalty_present;
        sp.mirostat        = params->mirostat;
        sp.mirostat_tau    = params->mirostat_tau;
        sp.mirostat_eta    = params->mirostat_eta;
        if (params->grammar && params->grammar[0] != '\0') {
            sp.grammar = common_grammar(COMMON_GRAMMAR_TYPE_USER, params->grammar);
        }

        common_sampler * smpl = common_sampler_init(model, sp);
        if (!smpl) {
            llu::set_error("common_sampler_init failed (invalid grammar?)");
            return nullptr;
        }
        llu_sampler * result = new llu_sampler();
        result->smpl = smpl;
        return result;
    });
}

void llu_sampler_free(llu_sampler * smpl) {
    delete smpl;
}

llama_token llu_sampler_sample(llu_sampler * smpl, llama_context * ctx, int32_t idx) {
    return llu::guarded<llama_token>(LLAMA_TOKEN_NULL, "sampling", [&]() -> llama_token {
        if (smpl == nullptr || ctx == nullptr) {
            llu::set_error("invalid arguments");
            return LLAMA_TOKEN_NULL;
        }
        return common_sampler_sample(smpl->smpl, ctx, idx);
    });
}

void llu_sampler_accept(llu_sampler * smpl, llama_token token, bool is_generated) {
    llu::guarded_void("accepting a token", [&]() {
        if (smpl) {
            common_sampler_accept(smpl->smpl, token, is_generated);
        }
    });
}

int32_t llu_sampler_sample_and_accept_n(
        llu_sampler       * smpl,
        llama_context     * ctx,
        const llama_token * draft,
        int32_t             n_draft,
        llama_token       * out) {
    return llu::guarded<int32_t>(-1, "verifying the draft", [&]() -> int32_t {
        if (smpl == nullptr || ctx == nullptr || out == nullptr || n_draft < 0 || (draft == nullptr && n_draft > 0)) {
            llu::set_error("invalid arguments");
            return -1;
        }
        smpl->draft.assign(draft, draft + n_draft);
        const std::vector<llama_token> ids = common_sampler_sample_and_accept_n(smpl->smpl, ctx, smpl->draft);
        const int32_t n = (int32_t) std::min(ids.size(), (size_t) n_draft + 1);
        std::copy(ids.begin(), ids.begin() + n, out);
        return n;
    });
}

} // extern "C"
