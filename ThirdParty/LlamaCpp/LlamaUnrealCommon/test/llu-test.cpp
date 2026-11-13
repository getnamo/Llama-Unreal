// Copyright 2025-current Getnamo.
//
// CPU-only check of llama-unreal-common. Uses only the C APIs (llama.h + llama-unreal-common.h), the same
// way Llama-Unreal does, and mirrors its speculative loop (FLlamaInternal::GenerateSpeculative).
//
// Build: configure llama.cpp with this library's cmake include and -DLLU_BUILD_TEST=ON, then build
//        the llama-unreal-common-test target.
// Usage: llama-unreal-common-test <small model.gguf> [MTP model.gguf] [Jinja-only template model.gguf]
//        e.g. Qwen3-0.6B-Q8_0.gguf  Qwen3.5-4B-MTP-Q4_K_M.gguf  gemma-4-E2B-it-Q6_K.gguf
//
// Greedy output with speculation must match greedy output without it (first 80 characters; batched
// verification can flip near-ties later on), and drafts must actually be accepted.

#include "llama-unreal-common.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static int g_failures = 0;

static void check(bool ok, const char * what) {
    std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) {
        g_failures++;
    }
}

// No offload devices: everything runs on the CPU
static ggml_backend_dev_t g_no_devices[] = { nullptr };

static llama_model * load_model(const char * path, bool load_mtp, bool vocab_only = false) {
    llama_model_params p = llama_model_default_params();
    p.n_gpu_layers = 0;
    p.devices      = g_no_devices;
    p.load_mtp     = load_mtp;
    p.vocab_only   = vocab_only;
    return llama_model_load_from_file(path, p);
}

static llama_context * make_context(llama_model * model, uint32_t n_rs_seq, llama_context_type type = LLAMA_CONTEXT_TYPE_DEFAULT, llama_context * other = nullptr) {
    llama_context_params p = llama_context_default_params();
    p.n_ctx                 = 2048;
    p.n_batch               = 512;
    p.n_threads             = 8;
    p.n_threads_batch       = 8;
    p.offload_kqv           = false;
    p.op_offload            = false;
    p.n_outputs_max_per_seq = 0;
    p.n_rs_seq              = n_rs_seq;
    p.ctx_type              = type;
    p.ctx_other             = other;
    return llama_init_from_model(model, p);
}

static std::vector<llama_token> tokenize(const llama_model * model, const std::string & text) {
    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> out(text.size() + 16);
    int n = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), out.data(), (int32_t) out.size(), true, true);
    out.resize(n > 0 ? n : 0);
    return out;
}

static std::string detokenize(const llama_model * model, const std::vector<llama_token> & tokens) {
    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::string out;
    char buf[256];
    for (llama_token t : tokens) {
        const int n = llama_token_to_piece(vocab, t, buf, sizeof(buf), 0, true);
        if (n > 0) {
            out.append(buf, n);
        }
    }
    return out;
}

// User turn with the model's built-in-templater template (the small test models' templates are recognized)
static std::string chat_prompt(const llama_model * model, const char * user) {
    const char * tmpl = llama_model_chat_template(model, nullptr);
    llama_chat_message msg = { "user", user };
    std::vector<char> buf(4096);
    int n = llama_chat_apply_template(tmpl, &msg, 1, true, buf.data(), (int32_t) buf.size());
    if (n < 0) {
        return user;
    }
    return std::string(buf.data(), std::min<size_t>(n, buf.size()));
}

static llu_sampler * greedy_sampler(const llama_model * model, const char * grammar = nullptr) {
    llu_sampler_params sp = llu_sampler_default_params();
    sp.temp    = 0.0f;
    sp.seed    = 1234;
    sp.grammar = grammar;
    return llu_sampler_init(model, &sp);
}

static std::vector<llama_token> generate_normal(llama_model * model, const std::vector<llama_token> & prompt, int n_gen, uint32_t n_rs_seq = 0) {
    std::vector<llama_token> out;
    llama_context * ctx = make_context(model, n_rs_seq);
    llu_sampler * smpl = greedy_sampler(model);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    std::vector<llama_token> p = prompt;
    if (llama_decode(ctx, llama_batch_get_one(p.data(), (int32_t) p.size())) == 0) {
        for (int i = 0; i < n_gen; i++) {
            llama_token t = llu_sampler_sample(smpl, ctx, -1);
            llu_sampler_accept(smpl, t, true);
            if (llama_vocab_is_eog(vocab, t)) {
                break;
            }
            out.push_back(t);
            if (llama_decode(ctx, llama_batch_get_one(&t, 1)) != 0) {
                break;
            }
        }
    }
    llu_sampler_free(smpl);
    llama_free(ctx);
    return out;
}

struct spec_stats {
    int drafted  = 0;
    int accepted = 0;
    int steps    = 0;
};

// Same loop as FLlamaInternal::GenerateSpeculative
static std::vector<llama_token> generate_speculative(llama_model * model, llama_context * ctx, llama_context * ctx_dft,
        uint32_t types, int n_max, const std::vector<llama_token> & prompt, int n_gen, spec_stats & stats) {
    std::vector<llama_token> out;
    const llama_vocab * vocab = llama_model_get_vocab(model);

    llu_speculative_params params = llu_speculative_default_params();
    params.types   = types;
    params.n_max   = n_max;
    params.ctx_tgt = ctx;
    params.ctx_dft = ctx_dft;
    llu_speculative * spec = llu_speculative_init(&params);
    if (!spec) {
        std::printf("  llu_speculative_init failed: %s\n", llu_last_error());
        return out;
    }
    llu_sampler * smpl = greedy_sampler(model);

    bool draft_ok = false;
    if (llu_speculative_decode(spec, prompt.data(), (int32_t) prompt.size(), 0, false, &draft_ok) != 0 || !draft_ok) {
        std::printf("  prompt decode failed (draft_ok=%d): %s\n", (int) draft_ok, llu_last_error());
    }
    std::vector<llama_token> context = prompt;
    llama_pos n_past = (llama_pos) prompt.size();

    std::vector<llama_token> draft(n_max), verify(n_max + 1), ids(n_max + 1);
    llama_token id_last = llu_sampler_sample(smpl, ctx, -1);
    llu_sampler_accept(smpl, id_last, true);
    llu_speculative_begin(spec, context.data(), (int32_t) context.size());

    while (!llama_vocab_is_eog(vocab, id_last) && (int) out.size() < n_gen) {
        out.push_back(id_last);

        const int n_draft = std::max(0, llu_speculative_draft(spec, context.data(), (int32_t) context.size(), id_last, n_past, n_max, draft.data(), n_max));
        if (ctx_dft) {
            llama_memory_seq_rm(llama_get_memory(ctx_dft), 0, n_past, -1);
        }

        verify[0] = id_last;
        std::copy(draft.begin(), draft.begin() + n_draft, verify.begin() + 1);
        if (llu_speculative_decode(spec, verify.data(), 1 + n_draft, n_past, true, &draft_ok) != 0) {
            std::printf("  verify decode failed: %s\n", llu_last_error());
            break;
        }
        const int n_ids = llu_sampler_sample_and_accept_n(smpl, ctx, draft.data(), n_draft, ids.data());
        if (n_ids < 1) {
            std::printf("  verification failed: %s\n", llu_last_error());
            break;
        }
        const int n_accepted = n_ids - 1;
        int n_keep = n_accepted;
        for (int i = 0; i < n_accepted; i++) {
            if (llama_vocab_is_eog(vocab, draft[i])) {
                n_keep = i;
                break;
            }
        }

        context.push_back(id_last);
        context.insert(context.end(), draft.begin(), draft.begin() + n_keep);
        n_past += 1 + n_keep;
        llama_memory_seq_rm(llama_get_memory(ctx), 0, n_past, -1);
        if (ctx_dft) {
            llama_memory_seq_rm(llama_get_memory(ctx_dft), 0, n_past, -1);
        }
        llu_speculative_accept(spec, n_keep);

        stats.steps++;
        stats.drafted  += n_draft;
        stats.accepted += n_keep;

        out.insert(out.end(), draft.begin(), draft.begin() + n_keep);
        if (n_keep < n_accepted) {
            break;
        }
        id_last = ids[n_ids - 1];
    }

    llu_sampler_free(smpl);
    llu_speculative_free(spec);
    return out;
}

static void compare(const char * label, const llama_model * model, const std::vector<llama_token> & normal,
        const std::vector<llama_token> & spec, const spec_stats & stats) {
    const std::string a = detokenize(model, normal);
    const std::string b = detokenize(model, spec);
    const size_t prefix = std::min<size_t>(80, a.size());
    std::printf("  [%s] %d steps, accepted %d/%d drafted, %.2f tokens/step\n", label, stats.steps, stats.accepted,
        stats.drafted, stats.steps ? (double) (stats.steps + stats.accepted) / stats.steps : 0.0);
    std::printf("    output: %.160s\n", b.c_str());
    char what[160];
    std::snprintf(what, sizeof(what), "%s: greedy output matches normal generation", label);
    check(!a.empty() && b.compare(0, prefix, a, 0, prefix) == 0, what);
    if (b.compare(0, prefix, a, 0, prefix) != 0) {
        std::printf("    normal: %.120s\n    spec:   %.120s\n", a.c_str(), b.c_str());
    }
    std::snprintf(what, sizeof(what), "%s: drafts accepted", label);
    check(stats.accepted > 0, what);
}

// The n-gram drafter (ngram-map-k) only proposes a match that is followed by at least 48 more tokens, so
// the passage needs to be well over that
static const char * PASSAGE =
    "The lighthouse keeper climbed the spiral stairs every evening at dusk. He trimmed the wick, "
    "polished the great lens until it gleamed, and wound the clockwork that turned the light. "
    "Ships passing the rocky point counted on that steady beam to find their way home through the fog. "
    "In winter the storms rattled the windows, but the keeper never missed a single night.";

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <small model.gguf> [MTP model.gguf] [Jinja-only template model.gguf]\n", argv[0]);
        return 2;
    }
    llama_log_set([](ggml_log_level level, const char * text, void *) {
        if (level == GGML_LOG_LEVEL_ERROR) {
            std::fputs(text, stderr);
        }
    }, nullptr);
    llama_backend_init();

    // Misc
    {
        const float in[2] = { 3.0f, 4.0f };
        float out[2] = {};
        llu_embd_normalize(in, out, 2, 2);
        check(std::fabs(out[0] - 0.6f) < 1e-5f && std::fabs(out[1] - 0.8f) < 1e-5f, "embd_normalize (euclidean)");

        llu_sampler_params stale = llu_sampler_default_params();
        stale.struct_size -= 4;
        check(llu_sampler_init(nullptr, &stale) == nullptr && std::strlen(llu_last_error()) > 0, "mismatched struct_size is rejected");
    }

    llama_model * model = load_model(argv[1], false);
    if (!model) {
        std::fprintf(stderr, "failed to load %s\n", argv[1]);
        return 1;
    }

    // Sampling with a grammar, and an invalid grammar
    {
        llama_context * ctx = make_context(model, 0);
        llu_sampler * smpl = greedy_sampler(model, "root ::= \"yes\" | \"no\"");
        check(smpl != nullptr, "sampler with a grammar is created");
        std::vector<llama_token> prompt = tokenize(model, chat_prompt(model, "Is the sky blue on a clear day? /no_think"));
        std::vector<llama_token> answer;
        if (smpl && llama_decode(ctx, llama_batch_get_one(prompt.data(), (int32_t) prompt.size())) == 0) {
            for (int i = 0; i < 8; i++) {
                llama_token t = llu_sampler_sample(smpl, ctx, -1);
                llu_sampler_accept(smpl, t, true);
                if (llama_vocab_is_eog(llama_model_get_vocab(model), t)) {
                    break;
                }
                answer.push_back(t);
                if (llama_decode(ctx, llama_batch_get_one(&t, 1)) != 0) {
                    break;
                }
            }
        }
        const std::string text = detokenize(model, answer);
        std::printf("  grammar answer: '%s'\n", text.c_str());
        check(text == "yes" || text == "no", "grammar constrains the output");
        llu_sampler_free(smpl);
        llama_free(ctx);

        llu_sampler * bad = greedy_sampler(model, "root ::= (");
        check(bad == nullptr && std::strlen(llu_last_error()) > 0, "invalid grammar fails with an error");
        std::printf("  error: %s\n", llu_last_error());
        llu_sampler_free(bad);
    }

    const int n_gen = 160;
    const std::string echo = std::string("Repeat the following paragraph back word for word, twice, with nothing else:\n\n") + PASSAGE;
    const std::vector<llama_token> echo_prompt = tokenize(model, chat_prompt(model, (echo + " /no_think").c_str()));
    const std::vector<llama_token> normal = generate_normal(model, echo_prompt, n_gen);

    // N-gram speculation
    {
        llama_context * ctx = make_context(model, 0);
        spec_stats stats;
        const std::vector<llama_token> spec = generate_speculative(model, ctx, nullptr, LLU_SPECULATIVE_NGRAM, 16, echo_prompt, n_gen, stats);
        compare("ngram", model, normal, spec, stats);
        llama_free(ctx);
    }

    // Draft model (the same small model, so nearly every draft is accepted) + n-gram
    {
        llama_model * draft_model = load_model(argv[1], false);
        check(llu_speculative_are_compatible(model, draft_model), "draft model vocab is compatible");
        llama_context * ctx = make_context(model, 0);
        llama_context * ctx_dft = make_context(draft_model, 0);
        spec_stats stats;
        const std::vector<llama_token> spec = generate_speculative(model, ctx, ctx_dft,
            LLU_SPECULATIVE_DRAFT_MODEL | LLU_SPECULATIVE_NGRAM, 8, echo_prompt, n_gen, stats);
        compare("draft model + ngram", model, normal, spec, stats);
        llama_free(ctx_dft);
        llama_free(ctx);
        llama_model_free(draft_model);
    }

    // MTP (hybrid models also need rollback snapshots on the target)
    if (argc >= 3 && argv[2][0]) {
        llama_model * mtp_model = load_model(argv[2], true);
        if (!mtp_model) {
            check(false, "MTP model loads");
        } else {
            const int n_max = 3;
            const bool hybrid = llama_model_is_hybrid(mtp_model) || llama_model_is_recurrent(mtp_model);
            const std::string prose = "In one paragraph of about 80 words, explain how a lighthouse helps ships at night. /no_think";
            const std::vector<llama_token> prompt = tokenize(mtp_model, chat_prompt(mtp_model, prose.c_str()));
            const std::vector<llama_token> mtp_normal = generate_normal(mtp_model, prompt, 64, hybrid ? n_max : 0);

            llama_context * ctx = make_context(mtp_model, hybrid ? n_max : 0);
            llama_context * ctx_mtp = make_context(mtp_model, 0, LLAMA_CONTEXT_TYPE_MTP, ctx);
            spec_stats stats;
            const std::vector<llama_token> spec = generate_speculative(mtp_model, ctx, ctx_mtp, LLU_SPECULATIVE_MTP, n_max, prompt, 64, stats);
            compare("mtp", mtp_model, mtp_normal, spec, stats);
            llama_free(ctx_mtp);
            llama_free(ctx);
            llama_model_free(mtp_model);
        }
    }

    // Jinja templates (vocab only: no weights needed)
    if (argc >= 4 && argv[3][0]) {
        llama_model * tmpl_model = load_model(argv[3], false, /*vocab_only*/ true);
        llu_chat_templates * tmpls = tmpl_model ? llu_chat_templates_init(tmpl_model, nullptr) : nullptr;
        check(tmpls != nullptr, "Jinja template parses");
        if (tmpls) {
            llama_chat_message msgs[3] = { { "system", "Be terse." }, { "user", "Hello there" }, { "assistant", "Hi." } };
            const int32_t needed = llu_chat_templates_apply(tmpls, msgs, 3, true, false, nullptr, 0);
            std::vector<char> buf(needed > 0 ? needed : 1);
            const int32_t n = llu_chat_templates_apply(tmpls, msgs, 3, true, false, buf.data(), (int32_t) buf.size());
            const std::string rendered(buf.data(), n > 0 ? n : 0);
            check(n == needed && rendered.find("Hello there") != std::string::npos && rendered.find("Be terse.") != std::string::npos,
                "Jinja template renders the conversation");
            char open_tag[64], close_tag[64];
            const bool thinking = llu_chat_templates_thinking_tags(tmpls, open_tag, sizeof(open_tag), close_tag, sizeof(close_tag));
            std::printf("  thinking tags: %s '%s' ... '%s'\n", thinking ? "yes" : "no", open_tag, close_tag);
            llu_chat_templates_free(tmpls);
        }
        llama_model_free(tmpl_model);
    }

    llama_model_free(model);
    llama_backend_free();

    std::printf("\n%s (%d failure%s)\n", g_failures ? "FAILED" : "ALL PASSED", g_failures, g_failures == 1 ? "" : "s");
    return g_failures ? 1 : 0;
}
