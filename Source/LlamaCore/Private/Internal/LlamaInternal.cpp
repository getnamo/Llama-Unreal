// Copyright 2025-current Getnamo.

#include "Internal/LlamaInternal.h"
#include "common/common.h"
#include "common/sampling.h"
#include "common/speculative.h"
#include "mtmd/mtmd.h"
#include "mtmd/mtmd-helper.h"
#include "LlamaDataTypes.h"
#include "LlamaUtility.h"
#include "HardwareInfo.h"

// Cross-platform strdup. MSVC ships `_strdup` and warns about plain `strdup`;
// POSIX (glibc/clang on Linux) ships `strdup` and never had `_strdup`.
#if PLATFORM_WINDOWS
    #define LLAMA_STRDUP _strdup
#else
    #define LLAMA_STRDUP strdup
#endif

// ---------------------------------------------------------------------------
// Allocator-safe wrappers around common_* helpers.
//
// Helpers in `llama-common*.lib` that return STL containers by value
// (std::vector, std::string) allocate the container's backing storage with
// the LIB's allocator. When that container destructs inside our module —
// where operator new/delete is overridden to mimalloc via PerModuleInline.inl
// — the free path tries to release CRT-allocated memory through mimalloc and
// crashes inside _mi_free_block_mt with EXCEPTION_ACCESS_VIOLATION reading
// 0xfff...f. This was latent before the b9090 split of common.lib into
// llama-common.lib + llama-common-base.lib (which changed STL/allocator
// linkage relative to our module).
//
// Fix pattern: bypass the std-returning helpers and call the raw C-ABI
// `llama_*` entry points against buffers we own, so allocation and free both
// go through the same allocator (mimalloc).
// ---------------------------------------------------------------------------
namespace
{
    // Replacement for `common_tokenize(ctx, text, add_special, parse_special)`.
    // Output vector is owned by the caller's module.
    static std::vector<llama_token> SafeTokenize(
        const llama_vocab* Vocab,
        const std::string& Text,
        bool bAddSpecial,
        bool bParseSpecial)
    {
        std::vector<llama_token> Out;
        if (!Vocab) return Out;

        const int32_t Needed = -llama_tokenize(
            Vocab, Text.data(), (int32_t)Text.size(),
            /*tokens*/ nullptr, /*n_tokens_max*/ 0,
            bAddSpecial, bParseSpecial);
        if (Needed <= 0) return Out;

        Out.resize((size_t)Needed);
        const int32_t Wrote = llama_tokenize(
            Vocab, Text.data(), (int32_t)Text.size(),
            Out.data(), (int32_t)Out.size(),
            bAddSpecial, bParseSpecial);
        if (Wrote < 0 || Wrote > Needed)
        {
            Out.clear();
            return Out;
        }
        Out.resize((size_t)Wrote);
        return Out;
    }

    // Replacement for `common_token_to_piece(vocab, token, special)`.
    // Output string is owned by the caller's module.
    static std::string SafeTokenToPiece(
        const llama_vocab* Vocab,
        llama_token Token,
        bool bSpecial)
    {
        if (!Vocab) return std::string();

        // Common case: pieces are short. Single attempt with a small buffer
        // covers virtually all tokens; fall back to the size-probe path on
        // the rare overflow.
        char StackBuf[128];
        const int32_t Wrote = llama_token_to_piece(
            Vocab, Token, StackBuf, (int32_t)sizeof(StackBuf),
            /*lstrip*/ 0, bSpecial);

        if (Wrote >= 0)
        {
            return std::string(StackBuf, (size_t)Wrote);
        }

        // Negative return = required size (negated). Allocate and retry.
        const int32_t Needed = -Wrote;
        std::string Out;
        Out.resize((size_t)Needed);
        const int32_t Wrote2 = llama_token_to_piece(
            Vocab, Token, Out.data(), (int32_t)Out.size(),
            /*lstrip*/ 0, bSpecial);
        if (Wrote2 < 0 || Wrote2 > Needed)
        {
            return std::string();
        }
        Out.resize((size_t)Wrote2);
        return Out;
    }
}

bool FLlamaInternal::LoadModelFromParams(const FLLMModelParams& InModelParams)
{
    FString RHI = FHardwareInfo::GetHardwareDetailsString();
    FString GPU = FPlatformMisc::GetPrimaryGPUBrand();

    UE_LOG(LogTemp, Log, TEXT("Device Found: %s %s"), *GPU, *RHI);

    LastLoadedParams = InModelParams;

    // only print errors
    llama_log_set([](enum ggml_log_level level, const char* text, void* /* user_data */)
    {
        if (level == GGML_LOG_LEVEL_ERROR) { // >= would also match GGML_LOG_LEVEL_CONT (progress dots)
            // Route to UE log so it appears in editor Output Log, not just stderr
            UE_LOG(LlamaLog, Warning, TEXT("[llama] %hs"), text);
        }
    }, nullptr);

    // load dynamic backends
    ggml_backend_load_all();

    std::string ModelPath = TCHAR_TO_UTF8(*FLlamaPaths::ParsePathIntoFullPath(InModelParams.PathToModel));


    //Regular init
    llama_model_params LlamaModelParams = llama_model_default_params();
    LlamaModelParams.n_gpu_layers = InModelParams.GPULayers;

    //MTP heads are skipped at load unless requested
    const ELLMSpeculativeMode SpecMode = InModelParams.Advanced.Speculative.Mode;
    LlamaModelParams.load_mtp = !InModelParams.Advanced.bEmbeddingMode &&
        (SpecMode == ELLMSpeculativeMode::MTP || SpecMode == ELLMSpeculativeMode::MTPAndNGram);

    LlamaModel = llama_model_load_from_file(ModelPath.c_str(), LlamaModelParams);
    if (!LlamaModel)
    {
        FString ErrorMessage = FString::Printf(TEXT("Unable to load model at <%hs>"), ModelPath.c_str());
        EmitErrorMessage(ErrorMessage, 10, __func__);
        return false;
    }

    llama_context_params ContextParams = llama_context_default_params();
    ContextParams.n_ctx = InModelParams.MaxContextLength;
    ContextParams.n_batch = InModelParams.MaxBatchLength;
    ContextParams.n_threads = InModelParams.Threads;
    ContextParams.n_threads_batch = InModelParams.Threads;

    if (InModelParams.Advanced.bEmbeddingMode)
    {
        ContextParams.embeddings = InModelParams.Advanced.bEmbeddingMode;
    }

    // Let the model decide flash attention — AUTO picks the best supported mode.
    // Required for vision models (Qwen2.5-Omni etc.) where the mmproj encoder needs it.
    if (!InModelParams.MmprojPath.IsEmpty())
    {
        ContextParams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
    }

    //Speculative verification needs logits for every token of the [last, draft...] batch
    if (InModelParams.Advanced.Speculative.Mode != ELLMSpeculativeMode::None && !InModelParams.Advanced.bEmbeddingMode)
    {
        ContextParams.n_outputs_max_per_seq = 0; //0 = up to n_outputs_max (n_batch)

        //Recurrent/hybrid state can't simply drop a rejected draft tail: keep enough per-token
        //snapshots to roll back a full draft
        if (llama_model_is_recurrent(LlamaModel) || llama_model_is_hybrid(LlamaModel))
        {
            ContextParams.n_rs_seq = (uint32_t)FMath::Clamp(InModelParams.Advanced.Speculative.DraftMaxTokens, 1, 64);
        }
    }

    SavedFlashAttnType = ContextParams.flash_attn_type;
    Context = llama_init_from_model(LlamaModel, ContextParams);
    
    if (!Context)
    {
        FString ErrorMessage = FString::Printf(TEXT("Unable to initialize model with given context params."));
        EmitErrorMessage(ErrorMessage, 11, __func__);
        return false;
    }

    //Only standard mode uses sampling
    if (!InModelParams.Advanced.bEmbeddingMode)
    {
        BuildSamplers(InModelParams.Advanced.Sampling, InModelParams.Seed);

        //NB: this is just a starting heuristic, 
        ContextHistory.reserve(1024);

        //Optional; failures are reported and generation falls back to the normal path
        InitSpeculative(InModelParams);
    }//End non-embedding mode

    //empty by default
    Template = std::string();
    TemplateSource = FLlamaString::ToStd(InModelParams.CustomChatTemplate.TemplateSource);

    //Prioritize: custom jinja, then name, then default
    if (!InModelParams.CustomChatTemplate.Jinja.IsEmpty())
    {
        Template = FLlamaString::ToStd(InModelParams.CustomChatTemplate.Jinja);
        if (InModelParams.CustomChatTemplate.TemplateSource.IsEmpty())
        {
            TemplateSource = std::string("Custom Jinja");
        }
    }
    else if (   !InModelParams.CustomChatTemplate.TemplateSource.IsEmpty() &&
                InModelParams.CustomChatTemplate.TemplateSource != TEXT("tokenizer.chat_template"))
    {
        //apply template source name, this may fail
        std::string TemplateName = FLlamaString::ToStd(InModelParams.CustomChatTemplate.TemplateSource);
        const char* TemplatePtr = llama_model_chat_template(LlamaModel, TemplateName.c_str());

        if (TemplatePtr != nullptr)
        {
            Template = std::string(TemplatePtr);
        }
    }

    if (InModelParams.Advanced.bEmbeddingMode)
    {
        Template = std::string("");
        TemplateSource = std::string("embedding mode, templates not used");
    }
    else
    {

        if (Template.empty())
        {
            const char* TemplatePtr = llama_model_chat_template(LlamaModel, nullptr);

            if (TemplatePtr != nullptr)
            {
                Template = std::string(TemplatePtr);
                TemplateSource = std::string("tokenizer.chat_template");
            }
        }
    }
    
    FilledContextCharLength = 0;

    //Detect thinking mode support from template
    bThinkingEnabled = InModelParams.Advanced.Thinking.bEnableThinking;
    bStripThinkingFromResponse = InModelParams.Advanced.Thinking.bStripThinkingFromResponse;

    //Auto-detect thinking tags from template source (Qwen3, DeepSeek, etc.)
    if (Template.find("<think>") != std::string::npos || Template.find("enable_thinking") != std::string::npos)
    {
        ThinkingOpenTag = "<think>";
        ThinkingCloseTag = "</think>";
        bModelSupportsThinking = true;
        UE_LOG(LlamaLog, Log, TEXT("Thinking mode detected from template. Thinking %s."),
            bThinkingEnabled ? TEXT("enabled") : TEXT("disabled (empty think block will be injected)"));
    }
    else
    {
        ThinkingOpenTag.clear();
        ThinkingCloseTag.clear();
        bModelSupportsThinking = false;
    }

    bIsModelLoaded = true;

    //Initialize multimodal if mmproj path is provided
    if (!InModelParams.MmprojPath.IsEmpty())
    {
        InitMultimodal(InModelParams.MmprojPath);
    }

    return true;
}

void FLlamaInternal::BuildSamplers(const FLLMSamplingParams& Sampling, int32 Seed)
{
    if (Sampler)
    {
        llama_sampler_free(Sampler);
        Sampler = nullptr;
    }
    if (CommonSampler)
    {
        common_sampler_free(CommonSampler);
        CommonSampler = nullptr;
    }

    const llama_vocab* Vocab = llama_model_get_vocab(LlamaModel);

    //Validate the grammar through the C API first: it returns null on a parse error, whereas
    //common_sampler_init throws (and this module doesn't catch C++ exceptions)
    ActiveGrammar.clear();
    llama_sampler* GrammarSampler = nullptr;
    if (!Sampling.Grammar.IsEmpty())
    {
        const std::string GrammarStd = FLlamaString::ToStd(Sampling.Grammar);
        GrammarSampler = llama_sampler_init_grammar(Vocab, GrammarStd.c_str(), "root");
        if (GrammarSampler)
        {
            ActiveGrammar = GrammarStd;
        }
        else
        {
            EmitErrorMessage(TEXT("Invalid GBNF grammar (see [llama] log for the parse error). Generating without a grammar."), 12, __func__);
        }
    }

    //common sampler strategy
    if (Sampling.bUseCommonSampler)
    {
        common_params_sampling SamplingParams;

        //Plugin defaults for these match llama.cpp's (temp 0.8, penalties off)
        SamplingParams.temp = Sampling.Temp;
        SamplingParams.penalty_last_n = Sampling.PenaltyLastN;
        SamplingParams.penalty_repeat = Sampling.PenaltyRepeat;
        SamplingParams.penalty_freq = Sampling.PenaltyFrequency;
        SamplingParams.penalty_present = Sampling.PenaltyPresence;

        if (Sampling.MinP != -1.f)
        {
            SamplingParams.min_p = Sampling.MinP;
        }
        if (Sampling.TopK != -1.f)
        {
            SamplingParams.top_k = Sampling.TopK;
        }
        if (Sampling.TopP != -1.f)
        {
            SamplingParams.top_p = Sampling.TopP;
        }
        if (Sampling.TypicalP != -1.f)
        {
            SamplingParams.typ_p = Sampling.TypicalP;
        }
        if (Sampling.Mirostat != -1)
        {
            SamplingParams.mirostat = Sampling.Mirostat;
            SamplingParams.mirostat_eta = Sampling.MirostatEta;
            SamplingParams.mirostat_tau = Sampling.MirostatTau;
        }

        //Seed is either default or the one specifically passed in for deterministic results
        if (Seed != -1)
        {
            SamplingParams.seed = Seed;
        }

        if (!ActiveGrammar.empty())
        {
            SamplingParams.grammar = common_grammar(COMMON_GRAMMAR_TYPE_USER, ActiveGrammar);
        }

        CommonSampler = common_sampler_init(LlamaModel, SamplingParams);
    }

    Sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());

    //Grammar goes first so every later sampler only sees grammar-valid tokens (chain takes ownership)
    if (GrammarSampler)
    {
        llama_sampler_chain_add(Sampler, GrammarSampler);
    }

    //Temperature is always applied
    llama_sampler_chain_add(Sampler, llama_sampler_init_temp(Sampling.Temp));

    //If any of the repeat penalties are set, apply penalties to sampler
    if (Sampling.PenaltyLastN != 0 ||
        Sampling.PenaltyRepeat != 1.f ||
        Sampling.PenaltyFrequency != 0.f ||
        Sampling.PenaltyPresence != 0.f)
    {
        llama_sampler_chain_add(Sampler, llama_sampler_init_penalties(
            llama_vocab_n_tokens(Vocab),
            Sampling.PenaltyLastN, Sampling.PenaltyRepeat,
            Sampling.PenaltyFrequency, Sampling.PenaltyPresence));
    }

    //Optional sampling strategies - MinP should be applied by default of 0.05f
    if (Sampling.MinP != -1.f)
    {
        llama_sampler_chain_add(Sampler, llama_sampler_init_min_p(Sampling.MinP, 1));
    }
    if (Sampling.TopK != -1.f)
    {
        llama_sampler_chain_add(Sampler, llama_sampler_init_top_k(Sampling.TopK));
    }
    if (Sampling.TopP != -1.f)
    {
        llama_sampler_chain_add(Sampler, llama_sampler_init_top_p(Sampling.TopP, 1));
    }
    if (Sampling.TypicalP != -1.f)
    {
        llama_sampler_chain_add(Sampler, llama_sampler_init_typical(Sampling.TypicalP, 1));
    }
    if (Sampling.Mirostat != -1)
    {
        llama_sampler_chain_add(Sampler, llama_sampler_init_mirostat_v2(
            Sampling.Mirostat, Sampling.MirostatTau, Sampling.MirostatEta));
    }

    //Seed is either default or the one specifically passed in for deterministic results
    llama_sampler_chain_add(Sampler, llama_sampler_init_dist(Seed == -1 ? LLAMA_DEFAULT_SEED : (uint32_t)Seed));
}

bool FLlamaInternal::UpdateSamplingParams(const FLLMSamplingParams& Sampling)
{
    if (!bIsModelLoaded || LastLoadedParams.Advanced.bEmbeddingMode)
    {
        return false;
    }
    if (IsGenerating())
    {
        StopGeneration();
    }
    BuildSamplers(Sampling, LastLoadedParams.Seed);
    LastLoadedParams.Advanced.Sampling = Sampling;
    return true;
}

void FLlamaInternal::ResetGrammarForNewResponse()
{
    if (ActiveGrammar.empty())
    {
        return;
    }
    //Rebuild rather than reset: common_sampler_reset leaves the grammar in its end state, which
    //only allows end-of-generation (empty second responses)
    BuildSamplers(LastLoadedParams.Advanced.Sampling, LastLoadedParams.Seed);
}

//Port of llama.cpp's common_speculative_are_compatible (not exported): same vocab type, matching
//BOS/EOS where the model adds them, sizes within 128 and identical token text from id 5 on
static bool AreSpeculativeVocabsCompatible(const llama_vocab* Target, const llama_vocab* Draft)
{
    if (llama_vocab_type(Target) != llama_vocab_type(Draft))
    {
        return false;
    }
    if (llama_vocab_get_add_bos(Target) != llama_vocab_get_add_bos(Draft) ||
        (llama_vocab_get_add_bos(Target) && llama_vocab_bos(Target) != llama_vocab_bos(Draft)))
    {
        return false;
    }
    if (llama_vocab_get_add_eos(Target) != llama_vocab_get_add_eos(Draft) ||
        (llama_vocab_get_add_eos(Target) && llama_vocab_eos(Target) != llama_vocab_eos(Draft)))
    {
        return false;
    }
    const int32 NTarget = llama_vocab_n_tokens(Target);
    const int32 NDraft = llama_vocab_n_tokens(Draft);
    if (FMath::Abs(NTarget - NDraft) > 128)
    {
        return false;
    }
    for (int32 i = 5; i < FMath::Min(NTarget, NDraft); i++)
    {
        if (strcmp(llama_vocab_get_text(Target, i), llama_vocab_get_text(Draft, i)) != 0)
        {
            return false;
        }
    }
    return true;
}

bool FLlamaInternal::InitSpeculative(const FLLMModelParams& InModelParams)
{
    const FLLMSpeculativeParams& Spec = InModelParams.Advanced.Speculative;
    bSpeculativeInSync = true;
    ContextTokens.clear();

    if (Spec.Mode == ELLMSpeculativeMode::None)
    {
        return false;
    }

    //Recurrent/hybrid targets roll back through the n_rs_seq snapshots reserved at context creation
    if ((llama_model_is_recurrent(LlamaModel) || llama_model_is_hybrid(LlamaModel)) && llama_n_rs_seq(Context) == 0)
    {
        EmitErrorMessage(TEXT("Speculative decoding needs rollback snapshots for this recurrent/hybrid model, generating normally."), 13, __func__);
        return false;
    }

    const bool bUseDraftModel = Spec.Mode == ELLMSpeculativeMode::DraftModel || Spec.Mode == ELLMSpeculativeMode::DraftModelAndNGram;
    const bool bUseMTP = Spec.Mode == ELLMSpeculativeMode::MTP || Spec.Mode == ELLMSpeculativeMode::MTPAndNGram;
    const bool bUseNGram = Spec.Mode == ELLMSpeculativeMode::NGram || Spec.Mode == ELLMSpeculativeMode::DraftModelAndNGram ||
        Spec.Mode == ELLMSpeculativeMode::MTPAndNGram;

    if (bUseDraftModel)
    {
        const std::string DraftPath = TCHAR_TO_UTF8(*FLlamaPaths::ParsePathIntoFullPath(Spec.DraftModelPath));
        llama_model_params DraftModelParams = llama_model_default_params();
        DraftModelParams.n_gpu_layers = Spec.DraftGPULayers;

        DraftModel = Spec.DraftModelPath.IsEmpty() ? nullptr : llama_model_load_from_file(DraftPath.c_str(), DraftModelParams);
        if (!DraftModel)
        {
            EmitErrorMessage(FString::Printf(TEXT("Unable to load draft model at <%hs>, generating without speculation."), DraftPath.c_str()), 13, __func__);
            FreeSpeculative();
            return false;
        }

        if (!AreSpeculativeVocabsCompatible(llama_model_get_vocab(LlamaModel), llama_model_get_vocab(DraftModel)))
        {
            EmitErrorMessage(TEXT("Draft model tokenizer doesn't match the main model (use a model from the same family), generating without speculation."), 13, __func__);
            FreeSpeculative();
            return false;
        }

        llama_context_params DraftContextParams = llama_context_default_params();
        DraftContextParams.n_ctx = llama_n_ctx(Context);
        DraftContextParams.n_batch = llama_n_batch(Context);
        DraftContextParams.n_threads = InModelParams.Threads;
        DraftContextParams.n_threads_batch = InModelParams.Threads;
        if (llama_model_is_recurrent(DraftModel) || llama_model_is_hybrid(DraftModel))
        {
            //Drafting runs ahead by up to n_max + 1 tokens before being rewound
            DraftContextParams.n_rs_seq = (uint32_t)FMath::Clamp(Spec.DraftMaxTokens, 1, 64) + 1;
        }
        DraftContext = llama_init_from_model(DraftModel, DraftContextParams);
        if (!DraftContext)
        {
            EmitErrorMessage(TEXT("Unable to create the draft model context, generating without speculation."), 13, __func__);
            FreeSpeculative();
            return false;
        }
    }

    if (bUseMTP)
    {
        if (llama_model_n_layer_nextn(LlamaModel) <= 0)
        {
            EmitErrorMessage(TEXT("Model has no MTP heads (needs a GGUF converted with its nextn/MTP tensors), generating without speculation."), 13, __func__);
            FreeSpeculative();
            return false;
        }

        //Second context on the same model running the MTP head (mirrors common_speculative_init_result)
        llama_context_params MTPContextParams = llama_context_default_params();
        MTPContextParams.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
        MTPContextParams.n_ctx = llama_n_ctx(Context);
        MTPContextParams.n_batch = llama_n_batch(Context);
        MTPContextParams.n_rs_seq = 0;
        MTPContextParams.ctx_other = Context;
        MTPContextParams.n_threads = InModelParams.Threads;
        MTPContextParams.n_threads_batch = InModelParams.Threads;
        DraftContext = llama_init_from_model(LlamaModel, MTPContextParams);
        if (!DraftContext)
        {
            EmitErrorMessage(TEXT("Unable to create the MTP context, generating without speculation."), 13, __func__);
            FreeSpeculative();
            return false;
        }
    }

    SpeculativeParams = new common_params_speculative();
    SpeculativeParams->types.clear();
    if (bUseNGram)
    {
        SpeculativeParams->types.push_back(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K); //tried first
    }
    if (bUseDraftModel)
    {
        SpeculativeParams->types.push_back(COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE);
    }
    if (bUseMTP)
    {
        SpeculativeParams->types.push_back(COMMON_SPECULATIVE_TYPE_DRAFT_MTP);
    }
    SpeculativeParams->draft.n_max = FMath::Clamp(Spec.DraftMaxTokens, 1, 64);
    SpeculativeParams->draft.n_min = FMath::Max(0, Spec.DraftMinTokens);
    SpeculativeParams->draft.p_min = Spec.DraftMinProbability;
    SpeculativeParams->draft.ctx_tgt = Context;
    SpeculativeParams->draft.ctx_dft = DraftContext;

    try
    {
        Speculative = common_speculative_init(*SpeculativeParams, 1);
    }
    catch (const std::exception& Error)
    {
        EmitErrorMessage(FString::Printf(TEXT("Speculative decoding init failed (%hs), generating without speculation."), Error.what()), 13, __func__);
        Speculative = nullptr;
    }
    if (!Speculative)
    {
        FreeSpeculative();
        return false;
    }

    //Largest proposal any enabled drafter writes before we clamp it (ngram-mod: n_match + n_max)
    DraftTokens.clear();
    DraftTokens.reserve(512);
    SpeculativeBatch = llama_batch_init(1 + 64, 0, 1);
    TrackedBatch = llama_batch_init(llama_n_batch(Context), 0, 1);

    UE_LOG(LlamaLog, Log, TEXT("Speculative decoding enabled (%s), max %d draft tokens%s"),
        *UEnum::GetDisplayValueAsText(Spec.Mode).ToString(), SpeculativeParams->draft.n_max,
        bUseDraftModel ? *FString::Printf(TEXT(", draft model %s"), *FPaths::GetCleanFilename(Spec.DraftModelPath)) : TEXT(""));
    return true;
}

void FLlamaInternal::FreeSpeculative()
{
    if (Speculative)
    {
        common_speculative_free(Speculative);
        Speculative = nullptr;
    }
    delete SpeculativeParams;
    SpeculativeParams = nullptr;
    if (DraftContext)
    {
        llama_free(DraftContext);
        DraftContext = nullptr;
    }
    if (DraftModel)
    {
        llama_model_free(DraftModel);
        DraftModel = nullptr;
    }
    if (SpeculativeBatch.token)
    {
        llama_batch_free(SpeculativeBatch);
    }
    SpeculativeBatch = {};
    if (TrackedBatch.token)
    {
        llama_batch_free(TrackedBatch);
    }
    TrackedBatch = {};
}

bool FLlamaInternal::IsTokenMirrorConsistent() const
{
    return Context && (int64)ContextTokens.size() == (int64)llama_memory_seq_pos_max(llama_get_memory(Context), 0) + 1;
}

bool FLlamaInternal::CanSpeculate() const
{
    //Verification uses the common sampler; the draft side must have seen everything in the KV
    return Speculative && CommonSampler && bSpeculativeInSync;
}

int32 FLlamaInternal::DecodePromptChunk(const llama_token* Tokens, int32 NTokens)
{
    if (Speculative && bSpeculativeInSync)
    {
        return DecodeTracked(Tokens, NTokens, llama_memory_seq_pos_max(llama_get_memory(Context), 0) + 1);
    }

    llama_batch Batch = llama_batch_get_one(const_cast<llama_token*>(Tokens), NTokens);
    const int32 Result = llama_decode(Context, Batch);
    if (Result == 0)
    {
        ContextTokens.insert(ContextTokens.end(), Tokens, Tokens + NTokens);
    }
    return Result;
}

int32 FLlamaInternal::DecodeTracked(const llama_token* Tokens, int32 NTokens, llama_pos StartPos)
{
    TrackedBatch.n_tokens = NTokens;
    for (int32 i = 0; i < NTokens; i++)
    {
        TrackedBatch.token[i] = Tokens[i];
        TrackedBatch.pos[i] = StartPos + i;
        TrackedBatch.n_seq_id[i] = 1;
        TrackedBatch.seq_id[i][0] = 0;
        TrackedBatch.logits[i] = (i == NTokens - 1);
    }

    const int32 Result = llama_decode(Context, TrackedBatch);
    if (Result != 0)
    {
        return Result;
    }

    ContextTokens.insert(ContextTokens.end(), Tokens, Tokens + NTokens);
    if (Speculative && bSpeculativeInSync && !common_speculative_process(Speculative, TrackedBatch))
    {
        UE_LOG(LlamaLog, Warning, TEXT("Speculative: draft side failed to process a batch, disabled until the next context reset."));
        bSpeculativeInSync = false;
    }
    return 0;
}

void FLlamaInternal::TrimContextFrom(llama_pos FromPos)
{
    llama_memory_seq_rm(llama_get_memory(Context), 0, FromPos, -1);
    if (DraftContext)
    {
        llama_memory_seq_rm(llama_get_memory(DraftContext), 0, FromPos, -1);
    }
    if (ContextTokens.size() > (size_t)FromPos)
    {
        ContextTokens.resize(FromPos);
    }
    if (FromPos == 0)
    {
        //Empty context: the draft side is trivially back in sync
        bSpeculativeInSync = true;
    }
}

void FLlamaInternal::GenerateSpeculative(std::string& Response, int32& NDecoded, llama_pos& NPast, bool& bEOGExit)
{
    const llama_vocab* Vocab = llama_model_get_vocab(LlamaModel);
    const int32 NContext = llama_n_ctx(Context);
    const int32 MaxDraft = FMath::Min(SpeculativeParams->draft.n_max, (int32)llama_n_batch(Context) - 1);
    const float PacingSleep = LastLoadedParams.Advanced.Output.TokenGenerationPacingSleep;

    //Streams one token like the normal loop; false on end-of-generation
    auto Emit = [&](llama_token Token) -> bool
    {
        if (llama_vocab_is_eog(Vocab, Token))
        {
            bEOGExit = true;
            return false;
        }
        const std::string Piece = SafeTokenToPiece(Vocab, Token, true);
        Response += Piece;
        NDecoded += 1;
        if (OnTokenGenerated)
        {
            OnTokenGenerated(Piece);
        }
        return true;
    };

    auto AddToBatch = [this](llama_token Token, llama_pos Pos)
    {
        const int32 i = SpeculativeBatch.n_tokens++;
        SpeculativeBatch.token[i] = Token;
        SpeculativeBatch.pos[i] = Pos;
        SpeculativeBatch.n_seq_id[i] = 1;
        SpeculativeBatch.seq_id[i][0] = 0;
        SpeculativeBatch.logits[i] = true; //every position is verified
    };

    //First token from the prompt's logits. IdLast is always emitted but not yet decoded.
    llama_token IdLast = common_sampler_sample(CommonSampler, Context, -1);
    common_sampler_accept(CommonSampler, IdLast, true);
    if (!Emit(IdLast))
    {
        return;
    }

    common_speculative_begin(Speculative, 0, ContextTokens);

    int32 Drafted = 0;
    int32 Accepted = 0;
    int32 Steps = 0;
    bool bIdLastPending = true;
    int64 DraftUs = 0, VerifyUs = 0, SyncUs = 0, SampleUs = 0;

    while (bGenerationActive)
    {
        if (NPast + 1 >= NContext)
        {
            bGenerationActive = false;
            EmitErrorMessage(FString::Printf(TEXT("Context size %d exceeded on generation. Try increasing the context size and re-run prompt"), NContext), 31, __func__);
            break;
        }

        //Draft. The library appends into DraftTokens' reserved capacity.
        DraftTokens.clear();
        const int32 NDraftMax = FMath::Min(MaxDraft, NContext - (int32)NPast - 2);
        if (NDraftMax > 0)
        {
            common_speculative_draft_params& DraftParams = common_speculative_get_draft_params(Speculative, 0);
            DraftParams.drafting = true;
            DraftParams.n_max = NDraftMax;
            DraftParams.pos0 = NPast;
            DraftParams.id_last = IdLast;
            DraftParams.prompt = &ContextTokens;
            DraftParams.result = &DraftTokens;
            const int64 T0 = ggml_time_us();
            common_speculative_draft(Speculative);
            DraftUs += ggml_time_us() - T0;
            if ((int32)DraftTokens.size() > NDraftMax)
            {
                DraftTokens.resize(NDraftMax);
            }
        }
        //Drafting advanced the draft KV; rewind it so it re-ingests the verified batch below
        if (DraftContext)
        {
            llama_memory_seq_rm(llama_get_memory(DraftContext), 0, NPast, -1);
        }

        //Verify [IdLast, d0..dn-1] in one pass
        const int32 NDraft = (int32)DraftTokens.size();
        SpeculativeBatch.n_tokens = 0;
        AddToBatch(IdLast, NPast);
        for (int32 i = 0; i < NDraft; i++)
        {
            AddToBatch(DraftTokens[i], NPast + 1 + i);
        }
        int64 T0 = ggml_time_us();
        if (llama_decode(Context, SpeculativeBatch))
        {
            bGenerationActive = false;
            EmitErrorMessage(TEXT("Failed to decode. Could not find a KV slot for the batch (try reducing the size of the batch or increase the context)"), 32, __func__);
            return;
        }
        int64 T1 = ggml_time_us();
        VerifyUs += T1 - T0;
        const bool bProcessed = common_speculative_process(Speculative, SpeculativeBatch);
        T0 = ggml_time_us();
        SyncUs += T0 - T1;
        if (!bProcessed)
        {
            UE_LOG(LlamaLog, Warning, TEXT("Speculative: draft side failed to process a batch, disabled until the next context reset."));
            bSpeculativeInSync = false;
        }

        //Sample each position with the main sampler; stop at the first disagreement. Same as
        //common_sampler_sample_and_accept_n, which can't be used here (returns a std::vector
        //allocated on llama-common's heap).
        int32 NAccepted = 0;
        llama_token Next = 0;
        for (int32 i = 0; ; i++)
        {
            Next = common_sampler_sample(CommonSampler, Context, i);
            common_sampler_accept(CommonSampler, Next, true);
            if (i < NDraft && Next == DraftTokens[i])
            {
                NAccepted++;
                continue;
            }
            break;
        }
        SampleUs += ggml_time_us() - T0;

        //Never keep tokens after an end-of-generation in the context
        int32 NKeep = NAccepted;
        for (int32 i = 0; i < NAccepted; i++)
        {
            if (llama_vocab_is_eog(Vocab, DraftTokens[i]))
            {
                NKeep = i;
                break;
            }
        }

        //Commit IdLast + kept drafts; drop the rejected tail on both contexts
        ContextTokens.push_back(IdLast);
        ContextTokens.insert(ContextTokens.end(), DraftTokens.begin(), DraftTokens.begin() + NKeep);
        NPast += 1 + NKeep;
        llama_memory_seq_rm(llama_get_memory(Context), 0, NPast, -1);
        if (DraftContext)
        {
            llama_memory_seq_rm(llama_get_memory(DraftContext), 0, NPast, -1);
        }
        bIdLastPending = false;

        common_speculative_accept(Speculative, 0, (uint16_t)NKeep);
        Steps++;
        Drafted += NDraft;
        Accepted += NKeep;

        for (int32 i = 0; i < NKeep; i++)
        {
            Emit(DraftTokens[i]);
        }
        if (NKeep < NAccepted || !Emit(Next))
        {
            bEOGExit = true;
            break;
        }
        IdLast = Next;
        bIdLastPending = true;

        if (PacingSleep > 0.f)
        {
            FPlatformProcess::Sleep(PacingSleep);
        }
    }

    //Stopped early: decode the last emitted token so the KV matches the normal loop's end state
    if (bIdLastPending && !bEOGExit && NPast < NContext)
    {
        if (DecodeTracked(&IdLast, 1, NPast) == 0)
        {
            NPast++;
        }
    }

    LastSpeculativeStats.DraftedTokens = Drafted;
    LastSpeculativeStats.AcceptedTokens = Accepted;
    LastSpeculativeStats.VerificationSteps = Steps;
    LastSpeculativeStats.AcceptanceRate = Drafted > 0 ? (float)Accepted / Drafted : 0.f;
    LastSpeculativeStats.TokensPerStep = Steps > 0 ? (float)(Steps + Accepted) / Steps : 0.f;

    if (Steps > 0)
    {
        UE_LOG(LlamaLog, Verbose, TEXT("Speculative timing per pass: draft %.2fms, verify %.2fms, draft sync %.2fms, sampling %.2fms"),
            DraftUs / 1000.0 / Steps, VerifyUs / 1000.0 / Steps, SyncUs / 1000.0 / Steps, SampleUs / 1000.0 / Steps);
    }
}

void FLlamaInternal::UnloadModel()
{
    //Speculative state references the target context, free it first
    FreeSpeculative();

    //Free mtmd before context/model since it holds references to them
    FreeMultimodal();

    if (Sampler)
    {
        llama_sampler_free(Sampler);
        Sampler = nullptr;
    }
    if (Context)
    {
        llama_free(Context);
        Context = nullptr;
    }
    if (LlamaModel)
    {
        llama_model_free(LlamaModel);
        LlamaModel = nullptr;
    }
    if (CommonSampler)
    {
        common_sampler_free(CommonSampler);
        CommonSampler = nullptr;
    }

    //A reload must start from a clean conversation: stale Messages would be re-templated
    //and re-ingested on the next prompt insert (history duplicated per reload, #53)
    ClearMessages();
    ContextHistory.clear();
    ContextTokens.clear();
    FilledContextCharLength = 0;
    NextGenerationNPast = 0;
    ActiveGrammar.clear();

    bIsModelLoaded = false;
}

void FLlamaInternal::ClearMessages(size_t FromIndex)
{
    if (FromIndex >= Messages.size())
    {
        return;
    }
    for (size_t i = FromIndex; i < Messages.size(); i++)
    {
        //Roles are static strings, content is LLAMA_STRDUP'd
        free(const_cast<char*>(Messages[i].content));
    }
    Messages.resize(FromIndex);
}

std::string FLlamaInternal::WrapPromptForRole(const std::string& Text, EChatTemplateRole Role, const std::string& OverrideTemplate, bool bAddAssistantBoS)
{
    std::vector<llama_chat_message> MessageListWrapper;
    MessageListWrapper.push_back({ RoleForEnum(Role), LLAMA_STRDUP(Text.c_str()) });

    //pre-allocate buffer 2x the size of text
    std::vector<char> Buffer;

    int32 NewLen = 0;

    if (OverrideTemplate.empty())
    {
        NewLen = ApplyTemplateFromMessagesToBuffer(Template, MessageListWrapper, Buffer, bAddAssistantBoS);
    }
    else
    {
        NewLen = ApplyTemplateFromMessagesToBuffer(OverrideTemplate, MessageListWrapper, Buffer, bAddAssistantBoS);
    }

    free(const_cast<char*>(MessageListWrapper[0].content));

    if(NewLen > 0)
    {
        return std::string(Buffer.data(), Buffer.data() + NewLen);
    }
    else
    {
        return std::string("");
    }
}

void FLlamaInternal::StopGeneration()
{
    bGenerationActive = false;
}

bool FLlamaInternal::IsGenerating()
{
    return bGenerationActive;
}

int32 FLlamaInternal::MaxContext()
{
    if (Context)
    {
        return llama_n_ctx(Context);
    }
    else
    {
        return 0;
    }
}

int32 FLlamaInternal::UsedContext()
{
    if (Context)
    {
        return llama_memory_seq_pos_max(llama_get_memory(Context), 0);
    }
    else
    {
        return 0;
    }
}

bool FLlamaInternal::IsModelLoaded()
{
    return bIsModelLoaded;
}

void FLlamaInternal::ResetContextHistory(bool bKeepSystemsPrompt)
{
    if (!bIsModelLoaded)
    {
        return;
    }

    if (IsGenerating())
    {
        StopGeneration();
    }

    if (bKeepSystemsPrompt)
    {
        //Valid trim case
        if (Messages.size() > 1)
        {
            //Rollback all the messages except the first one
            RollbackContextHistoryByMessages(Messages.size() - 1);
            return;
        }
        else
        {
            //Only message is the system's prompt, nothing to do
            return;
        }
    }

    //Full Reset
    ContextHistory.clear();
    ClearMessages();

    llama_memory_clear(llama_get_memory(Context), false);
    TrimContextFrom(0);
    FilledContextCharLength = 0;
    NextGenerationNPast = 0;
}

void FLlamaInternal::RollbackContextHistoryByTokens(int32 NTokensToErase)
{
    // clear the last n_regen tokens from the KV cache and update n_past
    // seq_pos_max returns the max position (0-indexed), so token count = seq_pos_max + 1
    int32 TokenCount = llama_memory_seq_pos_max(llama_get_memory(Context), 0) + 1;

    TrimContextFrom(FMath::Max(0, TokenCount - NTokensToErase));

    //FilledContextCharLength -= NTokensToErase;

    //Run a decode to sync everything else
    //llama_decode(Context, llama_batch_get_one(nullptr, 0));
}

void FLlamaInternal::RollbackContextHistoryByMessages(int32 NMessagesToErase)
{
    //cannot do rollback if model isn't loaded, ignore.
    if (!bIsModelLoaded)
    {
        return;
    }

    if (IsGenerating())
    {
        StopGeneration();
    }

    if (NMessagesToErase <= Messages.size())
    {
        ClearMessages(Messages.size() - NMessagesToErase);
    }

    //Obtain full prompt before it gets deleted
    std::string FullPrompt(ContextHistory.data(), ContextHistory.data() + FilledContextCharLength);
    
    //resize the context history
    int32 NewLen = ApplyTemplateToContextHistory(false);

    //tokenize to find out how many tokens we need to remove

    //Obtain new prompt, find delta
    std::string FormattedPrompt(ContextHistory.data(), ContextHistory.data() + NewLen);

    std::string PromptToRemove(FullPrompt.substr(FormattedPrompt.length()));

    const llama_vocab* Vocab = llama_model_get_vocab(LlamaModel);
    const int NPromptTokens = -llama_tokenize(Vocab, PromptToRemove.c_str(), PromptToRemove.size(), NULL, 0, false, true);

    //now rollback KV-cache
    RollbackContextHistoryByTokens(NPromptTokens);

    //Sync resized length;
    FilledContextCharLength = NewLen;

    //Shrink to fit
    ContextHistory.resize(FilledContextCharLength);
}

std::string FLlamaInternal::InsertRawPrompt(const std::string& Prompt, bool bGenerateReply)
{
    if (!bIsModelLoaded)
    {
        UE_LOG(LlamaLog, Warning, TEXT("Model isn't loaded"));
        return 0;
    }

    int32 TokensProcessed = ProcessPrompt(Prompt);

    FLlamaString::AppendToCharVector(ContextHistory, Prompt);

    if (bGenerateReply)
    {
        ResetGrammarForNewResponse();
        std::string Response = Generate("", false);
        FLlamaString::AppendToCharVector(ContextHistory, Response);
    }
    return "";
}

std::string FLlamaInternal::InsertTemplatedPrompt(const std::string& Prompt, EChatTemplateRole Role, bool bAddAssistantBoS, bool bGenerateReply, const std::string& AssistantPrefill)
{
    if (!bIsModelLoaded)
    {
        UE_LOG(LlamaLog, Warning, TEXT("Model isn't loaded"));
        return std::string();
    }

    int32 NewLen = FilledContextCharLength;

    if (!Prompt.empty())
    {
        Messages.push_back({ RoleForEnum(Role), LLAMA_STRDUP(Prompt.c_str()) });

        NewLen = ApplyTemplateToContextHistory(bAddAssistantBoS);
    }

    //Check for invalid lengths
    if (NewLen < 0)
    {
        UE_LOG(LlamaLog, Warning, TEXT("Inserted prompt after templating has an invalid length of %d, skipping generation. Check your jinja template or model gguf. NB: some templates merge system prompts with user prompts (e.g. gemma) and it's considered normal behavior."), NewLen);
        return std::string();
    }

    //Inject empty think block when thinking is disabled on a thinking-capable model
    if (!bThinkingEnabled && bAddAssistantBoS && bModelSupportsThinking && NewLen > 0)
    {
        std::string EmptyThinkBlock = ThinkingOpenTag + "\n\n" + ThinkingCloseTag + "\n\n";
        size_t InjLen = EmptyThinkBlock.size();
        if (ContextHistory.size() < (size_t)NewLen + InjLen)
        {
            ContextHistory.resize(NewLen + InjLen);
        }
        memcpy(ContextHistory.data() + NewLen, EmptyThinkBlock.data(), InjLen);
        NewLen += (int32)InjLen;
    }

    //Inject assistant prefill (raw bytes appended after the assistant turn header so the model
    //continues from this text without an intervening EOT). Only valid when add_ass=true; ignored
    //otherwise to avoid corrupting non-assistant turns.
    if (bAddAssistantBoS && !AssistantPrefill.empty() && NewLen > 0)
    {
        size_t PrefillLen = AssistantPrefill.size();
        if (ContextHistory.size() < (size_t)NewLen + PrefillLen)
        {
            ContextHistory.resize(NewLen + PrefillLen);
        }
        memcpy(ContextHistory.data() + NewLen, AssistantPrefill.data(), PrefillLen);
        NewLen += (int32)PrefillLen;
    }
    else if (!bAddAssistantBoS && !AssistantPrefill.empty())
    {
        UE_LOG(LlamaLog, Warning, TEXT("InsertTemplatedPrompt: AssistantPrefill ignored because bAddAssistantBoS=false"));
    }

    //Only process non-zero prompts
    if (NewLen > 0)
    {
        std::string FormattedPrompt(ContextHistory.data() + FilledContextCharLength, ContextHistory.data() + NewLen);
        int32 TokensProcessed = ProcessPrompt(FormattedPrompt, Role);
    }

    FilledContextCharLength = NewLen;

    //Check for a reply if we want to generate one, otherwise return an empty reply
    std::string Response;
    if (bGenerateReply)
    {
        //Run generation. AssistantPrefill is forwarded so Generate() can seed the response
        //accumulator and emit the prefill through OnTokenGenerated before sampling resumes.
        ResetGrammarForNewResponse();
        Response = Generate("", true, bAddAssistantBoS ? AssistantPrefill : std::string());
    }

    return Response;
}

void FLlamaInternal::RebuildContextFromHistory(const TArray<FStructuredChatMessage>& InMessages)
{
    if (!bIsModelLoaded)
    {
        UE_LOG(LlamaLog, Warning, TEXT("RebuildContextFromHistory: model not loaded, skipping."));
        return;
    }

    if (IsGenerating())
    {
        StopGeneration();
    }

    //Cheap KV+state wipe (mirrors ResetContextHistory full-reset path)
    ContextHistory.clear();
    ClearMessages();
    llama_memory_clear(llama_get_memory(Context), false);
    TrimContextFrom(0);
    FilledContextCharLength = 0;
    NextGenerationNPast = 0;

    //Replay each message through the existing template+decode pipeline without generating
    for (const FStructuredChatMessage& Msg : InMessages)
    {
        const std::string Content = TCHAR_TO_UTF8(*Msg.Content);
        InsertTemplatedPrompt(Content, Msg.Role, /*bAddAssistantBoS=*/false, /*bGenerateReply=*/false);
    }
}

std::string FLlamaInternal::ResumeGeneration()
{
    //Todo: erase last assistant message to merge the two messages if the last message was the assistant one.

    //run an empty user prompt
    return Generate();
}

void FLlamaInternal::GetPromptEmbeddings(const std::string& Text, std::vector<float>& Embeddings)
{
    //apply https://github.com/ggml-org/llama.cpp/blob/master/examples/embedding/embedding.cpp wrapping logic

    if (!Context)
    {
        EmitErrorMessage(TEXT("Context invalid, did you load the model?"), 43, __func__);
        return;
    }

    //Tokenize prompt - we're crashing out here... 
    //Check if our sampling/etc params are wrong or vocab is wrong.
    //Try tokenizing using normal method?
    //CONTINUE HERE:

    // SafeTokenize replaces `common_tokenize` to keep the result vector's
    // backing storage on our side of the static-lib boundary. See helper
    // definition at the top of this file for the cross-allocator background.
    const llama_vocab* Vocab = llama_model_get_vocab(LlamaModel);
    std::vector<llama_token> Input = SafeTokenize(Vocab, Text, /*add_special*/ true, /*parse_special*/ true);
    if (Input.empty())
    {
        UE_LOG(LlamaLog, Error, TEXT("GetPromptEmbeddings: tokenize produced 0 tokens (text bytes=%d)"),
            (int32)Text.size());
        return;
    }

    //int32 NBatch = llama_n_ctx(Context);    //todo: get this from our params
    int32 NBatch = Input.size();    //todo: get this from our params

    llama_batch Batch = llama_batch_init(NBatch, 0, 1);
    //llama_batch Batch = llama_batch_get_one(Input.data(), Input.size());

    //add single batch
    BatchAddSeq(Batch, Input, 0);

    enum llama_pooling_type PoolingType = llama_pooling_type(Context);

    //Count number of embeddings
    int32 EmbeddingCount = 0;

    if (PoolingType == llama_pooling_type::LLAMA_POOLING_TYPE_NONE)
    {
        EmbeddingCount = Input.size();
    }
    else
    {
        EmbeddingCount = 1;
    }

    int32 NEmbd = llama_model_n_embd(LlamaModel);

    //allocate raw output buffer
    std::vector<float> Raw((size_t)EmbeddingCount * NEmbd, 0.f);

    //decode
    BatchDecodeEmbedding(Context, Batch, Raw.data(), 0, NEmbd, 2, EmbeddingCount);

    //Always return a single pooled vector. For NONE pooling, mean-pool per-token rows then re-normalize L2.
    if (EmbeddingCount > 1)
    {
        Embeddings.assign(NEmbd, 0.f);
        for (int32 t = 0; t < EmbeddingCount; ++t)
        {
            const float* Row = Raw.data() + t * NEmbd;
            for (int32 d = 0; d < NEmbd; ++d)
            {
                Embeddings[d] += Row[d];
            }
        }
        const float Inv = 1.f / static_cast<float>(EmbeddingCount);
        for (int32 d = 0; d < NEmbd; ++d) { Embeddings[d] *= Inv; }

        //L2 renormalize after pooling
        double SumSq = 0.0;
        for (int32 d = 0; d < NEmbd; ++d) { SumSq += static_cast<double>(Embeddings[d]) * Embeddings[d]; }
        const float Norm = SumSq > 0.0 ? static_cast<float>(1.0 / sqrt(SumSq)) : 1.f;
        for (int32 d = 0; d < NEmbd; ++d) { Embeddings[d] *= Norm; }
    }
    else
    {
        Embeddings = std::move(Raw);
    }

    llama_batch_free(Batch);

    UE_LOG(LlamaLog, Verbose, TEXT("FLlamaInternal::GetPromptEmbeddings: %d floats (pooling=%d, tokens=%d)"),
           static_cast<int32>(Embeddings.size()),
           static_cast<int32>(PoolingType),
           static_cast<int32>(Input.size()));
}

int32 FLlamaInternal::GetEmbeddingDimension() const
{
    return LlamaModel ? llama_model_n_embd(LlamaModel) : 0;
}

int32 FLlamaInternal::ProcessPrompt(const std::string& Prompt, EChatTemplateRole Role)
{
    const auto StartTime = ggml_time_us();

    //Grab vocab. seq_pos_max is -1 for an empty cache (0 means one token is already in it).
    const llama_vocab* Vocab = llama_model_get_vocab(LlamaModel);
    const llama_pos SeqPosMax = llama_memory_seq_pos_max(llama_get_memory(Context), 0);
    const bool IsFirst = SeqPosMax < 0;

    // tokenize the prompt
    const int NPromptTokens = -llama_tokenize(Vocab, Prompt.c_str(), Prompt.size(), NULL, 0, IsFirst, true);
    std::vector<llama_token> PromptTokens(NPromptTokens);
    if (llama_tokenize(Vocab, Prompt.c_str(), Prompt.size(), PromptTokens.data(), PromptTokens.size(), IsFirst, true) < 0)
    {
        EmitErrorMessage(TEXT("failed to tokenize the prompt"), 21, __func__);
        return NPromptTokens;
    }

    //check sizing before running prompt decode
    const int32 NContext = llama_n_ctx(Context);
    const int32 NContextUsed = SeqPosMax + 1;
    if (NContextUsed + NPromptTokens > NContext)
    {
        EmitErrorMessage(FString::Printf(
            TEXT("Failed to insert, tried to insert %d tokens to currently used %d tokens which is more than the max %d context size. Try increasing the context size and re-run prompt."),
            NPromptTokens, NContextUsed, NContext
        ), 22, __func__);
        return 0;
    }

    //llama_decode aborts the process if a batch exceeds n_batch (#53), so always feed the
    //prompt in n_batch-sized chunks. Pacing optionally splits further and sleeps in between.
    int32 ChunkSize = (int32)llama_n_batch(Context);
    const float PacingSleep = LastLoadedParams.Advanced.Output.PromptProcessingPacingSleep;
    if (PacingSleep > 0.f && LastLoadedParams.Advanced.Output.PromptProcessingPacingSplitN > 1)
    {
        const int32 SplitN = LastLoadedParams.Advanced.Output.PromptProcessingPacingSplitN;
        ChunkSize = FMath::Min(ChunkSize, FMath::Max(1, (NPromptTokens + SplitN - 1) / SplitN));
    }

    for (int32 StartIndex = 0; StartIndex < NPromptTokens; StartIndex += ChunkSize)
    {
        const int32 CurrentBatchSize = FMath::Min(ChunkSize, NPromptTokens - StartIndex);
        if (DecodePromptChunk(PromptTokens.data() + StartIndex, CurrentBatchSize))
        {
            EmitErrorMessage(TEXT("Failed to decode, could not find a KV slot for the batch (try reducing the size of the batch or increase the context)."), 23, __func__);
            return StartIndex;
        }

        if (PacingSleep > 0.f && StartIndex + CurrentBatchSize < NPromptTokens)
        {
            FPlatformProcess::Sleep(PacingSleep);
        }
    }

    const auto StopTime = ggml_time_us();
    const float Duration = (StopTime - StartTime) / 1000000.0f;

    if (OnPromptProcessed)
    {
        float Speed = NPromptTokens / Duration;
        OnPromptProcessed(NPromptTokens, Role, Speed);
    }

    return NPromptTokens;
}

std::string FLlamaInternal::Generate(const std::string& Prompt, bool bAppendToMessageHistory, const std::string& AssistantPrefill)
{
    const auto StartTime = ggml_time_us();

    bGenerationActive = true;

    if (!Prompt.empty())
    {
        int32 TokensProcessed = ProcessPrompt(Prompt);
    }

    //Seed response accumulator with prefill so it shows up in the final response and in the
    //assistant message history. The prefill bytes are assumed to already be in the KV cache
    //(InsertTemplatedPrompt injects them into the prompt-eval batch). We also emit through
    //OnTokenGenerated as a single piece so subscribers see one continuous stream.
    std::string Response = AssistantPrefill;
    if (!AssistantPrefill.empty() && OnTokenGenerated)
    {
        OnTokenGenerated(AssistantPrefill);
    }

    const llama_vocab* Vocab = llama_model_get_vocab(LlamaModel);

    llama_token NewTokenId;
    int32 NDecoded = 0;

    // check if we have enough space in the context to evaluate this batch - might need to be inside loop
    int NContext = llama_n_ctx(Context);
    bool bEOGExit = false;

    // For M-RoPE models (e.g. Qwen2VL), seq_pos_max reflects the max 2D spatial position of
    // image tokens and is NOT the correct next text position. Use NextGenerationNPast when it has
    // been set by ProcessMultimodalPrompt; otherwise fall back to seq_pos_max+1 (text-only path).
    llama_pos SeqPosMaxAtGenStart = llama_memory_seq_pos_max(llama_get_memory(Context), 0);
    llama_pos NPast = (NextGenerationNPast > 0)
        ? NextGenerationNPast
        : SeqPosMaxAtGenStart + 1;
    NextGenerationNPast = 0; // consumed

    UE_LOG(LlamaLog, Log, TEXT("[Generate] NPast=%d seq_pos_max=%d (from_mtmd=%s)"),
        (int32)NPast, (int32)SeqPosMaxAtGenStart,
        (NPast != SeqPosMaxAtGenStart + 1) ? TEXT("yes") : TEXT("no"));

    LastSpeculativeStats = FLLMSpeculativeStats();
    const bool bSpeculate = CanSpeculate() && NPast == SeqPosMaxAtGenStart + 1;
    if (bSpeculate)
    {
        GenerateSpeculative(Response, NDecoded, NPast, bEOGExit);
    }

    bool bFirstToken = true;
    while (!bSpeculate && bGenerationActive) //processing can be aborted by flipping the boolean
    {
        //Common sampler is a bit faster
        if (CommonSampler)
        {
            NewTokenId = common_sampler_sample(CommonSampler, Context, -1); //sample using common sampler
            common_sampler_accept(CommonSampler, NewTokenId, true);
        }
        else
        {
            NewTokenId = llama_sampler_sample(Sampler, Context, -1);
        }

        if (bFirstToken)
        {
            bFirstToken = false;
            const std::string FirstPiece = SafeTokenToPiece(Vocab, NewTokenId, true);
            UE_LOG(LlamaLog, Log, TEXT("[Generate] first token id=%d piece='%hs'"),
                (int32)NewTokenId, FirstPiece.c_str());
        }

        // is it an end of generation?
        if (llama_vocab_is_eog(Vocab, NewTokenId))
        {
            bEOGExit = true;
            break;
        }

        // convert the token to a string, print it and add it to the response.
        // SafeTokenToPiece keeps the string allocation on our side of the static-lib
        // boundary; see helper definition for the cross-allocator background.
        std::string Piece = SafeTokenToPiece(Vocab, NewTokenId, true);

        Response += Piece;
        NDecoded += 1;

        //NPast is the position this token is about to be decoded at
        if (NPast >= NContext)
        {
            bGenerationActive = false;
            FString ErrorMessage = FString::Printf(TEXT("Context size %d exceeded on generation. Try increasing the context size and re-run prompt"), NContext);

            EmitErrorMessage(ErrorMessage, 31, __func__);
            return Response;
        }

        if (OnTokenGenerated)
        {
            OnTokenGenerated(Piece);
        }

        // Use explicit n_past position (mirrors mtmd-cli reference implementation).
        // This is critical for M-RoPE models where seq_pos_max != true next text position.
        llama_batch SingleBatch = llama_batch_get_one(&NewTokenId, 1);
        SingleBatch.pos = &NPast;  // override auto-position with tracked n_past

        const bool bTracked = Speculative && bSpeculativeInSync;
        if (bTracked ? DecodeTracked(&NewTokenId, 1, NPast) != 0 : llama_decode(Context, SingleBatch) != 0)
        {
            bGenerationActive = false;
            FString ErrorMessage = TEXT("Failed to decode. Could not find a KV slot for the batch (try reducing the size of the batch or increase the context)");
            EmitErrorMessage(ErrorMessage, 32, __func__);
            //Return partial response
            return Response;
        }

        if (!bTracked)
        {
            ContextTokens.push_back(NewTokenId);
        }
        NPast++;

        //sleep pacing
        if (LastLoadedParams.Advanced.Output.TokenGenerationPacingSleep > 0.f)
        {
            FPlatformProcess::Sleep(LastLoadedParams.Advanced.Output.TokenGenerationPacingSleep);
        }
    }

    bGenerationActive = false;

    const auto StopTime = ggml_time_us();
    const float Duration = (StopTime - StartTime) / 1000000.0f;

    if (bAppendToMessageHistory)
    {
        //Add the raw response (with thinking) to our templated messages for context preservation
        Messages.push_back({ RoleForEnum(EChatTemplateRole::Assistant), LLAMA_STRDUP(Response.c_str()) });

        //Sync ContextHistory
        FilledContextCharLength = ApplyTemplateToContextHistory(false);
    }

    //Strip thinking content from emitted response if requested
    std::string EmittedResponse = Response;
    if (bStripThinkingFromResponse && bModelSupportsThinking && !ThinkingCloseTag.empty())
    {
        size_t ClosePos = EmittedResponse.find(ThinkingCloseTag);
        if (ClosePos != std::string::npos)
        {
            size_t ContentStart = ClosePos + ThinkingCloseTag.size();
            //Skip leading whitespace after close tag
            while (ContentStart < EmittedResponse.size() &&
                   (EmittedResponse[ContentStart] == '\n' || EmittedResponse[ContentStart] == '\r'))
            {
                ContentStart++;
            }
            EmittedResponse = EmittedResponse.substr(ContentStart);
        }
    }

    if (OnGenerationComplete)
    {
        OnGenerationComplete(EmittedResponse, Duration, NDecoded, NDecoded / Duration);
    }

    return EmittedResponse;
}

void FLlamaInternal::EmitErrorMessage(const FString& ErrorMessage, int32 ErrorCode, const FString& FunctionName)
{
    UE_LOG(LlamaLog, Error, TEXT("[%s error %d]: %s"), *FunctionName, ErrorCode,  *ErrorMessage);
    if (OnError)
    {
        OnError(ErrorMessage, ErrorCode);
    }
}

//NB: this function will apply out of range errors in log, this is normal behavior due to how templates are applied
int32 FLlamaInternal::ApplyTemplateToContextHistory(bool bAddAssistantBOS)
{
    return ApplyTemplateFromMessagesToBuffer(Template, Messages, ContextHistory, bAddAssistantBOS);
}

int32 FLlamaInternal::ApplyTemplateFromMessagesToBuffer(const std::string& InTemplate, std::vector<llama_chat_message>& FromMessages, std::vector<char>& ToBuffer, bool bAddAssistantBoS)
{
    //Handle empty template case
    char* templatePtr = (char*)InTemplate.c_str();
    if (InTemplate.length() == 0)
    {
        templatePtr = nullptr;
    }

    int32 NewLen = llama_chat_apply_template(templatePtr, FromMessages.data(), FromMessages.size(),
            bAddAssistantBoS, ToBuffer.data(), ToBuffer.size());

    //Resize if ToBuffer can't hold it
    if (NewLen > ToBuffer.size())
    {
        ToBuffer.resize(NewLen);
        NewLen = llama_chat_apply_template(InTemplate.c_str(), FromMessages.data(), FromMessages.size(),
            bAddAssistantBoS, ToBuffer.data(), ToBuffer.size());
    }
    else 
    {
        if (NewLen < 0)
        {
            EmitErrorMessage(TEXT("Failed to apply the chat template ApplyTemplateFromMessagesToBuffer, negative length"), 101, __func__);
        }
        else if (NewLen == 0)
        {
            //This isn't an error but needs to be handled by downstream
            
            //EmitErrorMessage(TEXT("Failed to apply the chat template ApplyTemplateFromMessagesToBuffer, length is 0."), 102, __func__);
        }
    }
    
    return NewLen;
}

const char* FLlamaInternal::RoleForEnum(EChatTemplateRole Role)
{
    if (Role == EChatTemplateRole::User)
    {
        return "user";
    }
    else if (Role == EChatTemplateRole::Assistant)
    {
        return "assistant";
    }
    else if (Role == EChatTemplateRole::System)
    {
        return "system";
    }
    else {
        return "unknown";
    }
}

//from https://github.com/ggml-org/llama.cpp/blob/master/examples/embedding/embedding.cpp
void FLlamaInternal::BatchDecodeEmbedding(llama_context* InContext, llama_batch& Batch, float* Output, int NSeq, int NEmbd, int EmbdNorm, int MaxRows)
{
    const enum llama_pooling_type pooling_type = llama_pooling_type(InContext);
    const struct llama_model* model = llama_get_model(InContext);

    // clear previous kv_cache values (irrelevant for embeddings)
    llama_memory_clear(llama_get_memory(InContext), false);

    // run model
    if (llama_model_has_encoder(model) && !llama_model_has_decoder(model))
    {
        // encoder-only model
        if (llama_encode(InContext, Batch) < 0)
        {
            UE_LOG(LlamaLog, Error, TEXT("%hs : failed to encode"), __func__);
            return;
        }
    }
    else if (!llama_model_has_encoder(model) && llama_model_has_decoder(model))
    {
        // decoder-only model
        if (llama_decode(InContext, Batch) < 0)
        {
            UE_LOG(LlamaLog, Log, TEXT("%hs : failed to decode"), __func__);
            return;
        }
    }

    for (int i = 0; i < Batch.n_tokens; i++)
    {
        if (Batch.logits && !Batch.logits[i])
        {
            continue;
        }

        const float* Embd = nullptr;
        int EmbdPos = 0;

        if (pooling_type == LLAMA_POOLING_TYPE_NONE)
        {
            // try to get token embeddings
            Embd = llama_get_embeddings_ith(InContext, i);
            EmbdPos = i;
        }
        else if (Batch.seq_id)
        {
            // try to get sequence embeddings - supported only when pooling_type is not NONE
            const llama_seq_id SeqId = Batch.seq_id[i] ? Batch.seq_id[i][0] : 0;
            Embd = llama_get_embeddings_seq(InContext, SeqId);
            EmbdPos = SeqId;
        }
        else
        {
            //NB: this generally won't work, we should crash here.
            Embd = llama_get_embeddings(InContext);
        }

        if (Embd == nullptr)
        {
            UE_LOG(LlamaLog, Error, TEXT("[BatchDecodeEmbedding] null embd at i=%d EmbdPos=%d pool=%d — skipping write"),
                i, EmbdPos, (int32)pooling_type);
            continue;
        }

        if (MaxRows > 0 && (EmbdPos < 0 || EmbdPos >= MaxRows))
        {
            UE_LOG(LlamaLog, Error, TEXT("[BatchDecodeEmbedding] OOB EmbdPos=%d (MaxRows=%d) at i=%d — skipping write"),
                EmbdPos, MaxRows, i);
            continue;
        }

        float* Out = Output + (size_t)EmbdPos * NEmbd;
        common_embd_normalize(Embd, Out, NEmbd, EmbdNorm);
    }
}

void FLlamaInternal::BatchAddSeq(llama_batch& batch, const std::vector<int32_t>& tokens, llama_seq_id seq_id)
{
    size_t n_tokens = tokens.size();
    for (size_t i = 0; i < n_tokens; i++) 
    {
        common_batch_add(batch, tokens[i], i, { seq_id }, true);
    }
}

bool FLlamaInternal::InitMultimodal(const FString& MmprojPath)
{
    if (MmprojPath.IsEmpty())
    {
        return false;
    }
    if (!LlamaModel)
    {
        EmitErrorMessage(TEXT("Cannot init multimodal: model not loaded"), 50, __func__);
        return false;
    }

    std::string Path = TCHAR_TO_UTF8(*FLlamaPaths::ParsePathIntoFullPath(MmprojPath));

    mtmd_context_params Params = mtmd_context_params_default();
    Params.use_gpu = true;
    Params.n_threads = LastLoadedParams.Threads;
    Params.flash_attn_type = SavedFlashAttnType;

    MtmdContext = mtmd_init_from_file(Path.c_str(), LlamaModel, Params);
    if (!MtmdContext)
    {
        EmitErrorMessage(FString::Printf(TEXT("Failed to load multimodal projector from <%hs>"), Path.c_str()), 50, __func__);
        bMtmdLoaded = false;
        return false;
    }

    // Route mtmd-helper logs (image/audio batch errors) through UE log
    mtmd_helper_log_set([](enum ggml_log_level level, const char* text, void* /*user_data*/)
    {
        if (level == GGML_LOG_LEVEL_ERROR) { // >= would also match GGML_LOG_LEVEL_CONT (progress dots)
            UE_LOG(LlamaLog, Warning, TEXT("[mtmd] %hs"), text);
        }
    }, nullptr);

    bMtmdLoaded = true;
    UE_LOG(LlamaLog, Log, TEXT("Multimodal projector loaded. Vision: %s, Audio: %s"),
        mtmd_support_vision(MtmdContext) ? TEXT("yes") : TEXT("no"),
        mtmd_support_audio(MtmdContext) ? TEXT("yes") : TEXT("no"));
    return true;
}

void FLlamaInternal::FreeMultimodal()
{
    if (MtmdContext)
    {
        mtmd_free(MtmdContext);
        MtmdContext = nullptr;
    }
    bMtmdLoaded = false;
}

bool FLlamaInternal::IsMultimodalLoaded()
{
    return bMtmdLoaded;
}

bool FLlamaInternal::SupportsVision()
{
    return MtmdContext && mtmd_support_vision(MtmdContext);
}

bool FLlamaInternal::SupportsAudio()
{
    return MtmdContext && mtmd_support_audio(MtmdContext);
}

int32 FLlamaInternal::GetAudioSampleRate()
{
    if (MtmdContext)
    {
        return mtmd_get_audio_sample_rate(MtmdContext);
    }
    return 0;
}

int32 FLlamaInternal::ProcessMultimodalPrompt(const std::string& FormattedPrompt, const TArray<FLlamaMediaEntry>& MediaEntries, EChatTemplateRole Role, bool bLogitsLast)
{
    //Image/audio chunks can't be mirrored to the draft side: speculate again after the next reset
    bSpeculativeInSync = false;

    const auto StartTime = ggml_time_us();

    // 1. Build bitmaps from media entries
    TArray<mtmd_bitmap*> Bitmaps;
    TArray<const mtmd_bitmap*> BitmapPtrs;
    // Video contexts returned by the file helper must outlive tokenize/eval; freed on scope exit
    std::vector<mtmd_helper::video_ptr> VideoContexts;

    for (const FLlamaMediaEntry& Entry : MediaEntries)
    {
        mtmd_bitmap* Bmp = nullptr;

        if (!Entry.FilePath.IsEmpty())
        {
            std::string FilePath = TCHAR_TO_UTF8(*FLlamaPaths::ParsePathIntoFullPath(Entry.FilePath));
            mtmd_helper_bitmap_wrapper Result = mtmd_helper_bitmap_init_from_file(
                MtmdContext, FilePath.c_str(), false, mtmd_helper_init_opt_default());
            Bmp = Result.bitmap;
            if (Result.video_ctx)
            {
                VideoContexts.emplace_back(Result.video_ctx);
            }
        }
        else if (Entry.MediaType == ELlamaMediaType::Image)
        {
            if (Entry.ImageRGBData.Num() > 0 && Entry.ImageWidth > 0 && Entry.ImageHeight > 0)
            {
                Bmp = mtmd_bitmap_init(Entry.ImageWidth, Entry.ImageHeight, Entry.ImageRGBData.GetData());
            }
        }
        else // Audio
        {
            if (Entry.AudioPCMData.Num() > 0)
            {
                Bmp = mtmd_bitmap_init_from_audio(Entry.AudioPCMData.Num(), Entry.AudioPCMData.GetData());
            }
        }

        if (!Bmp)
        {
            // Free any previously created bitmaps
            for (mtmd_bitmap* B : Bitmaps)
            {
                mtmd_bitmap_free(B);
            }
            EmitErrorMessage(TEXT("Failed to create mtmd bitmap from media entry"), 52, __func__);
            return 0;
        }

        Bitmaps.Add(Bmp);
        BitmapPtrs.Add(Bmp);
    }

    // 2. Tokenize with mtmd
    mtmd_input_chunks* Chunks = mtmd_input_chunks_init();
    // seq_pos_max returns -1 when the KV cache is empty; only add BOS on the very first prompt
    const llama_pos SeqPosMax = llama_memory_seq_pos_max(llama_get_memory(Context), 0);
    const bool IsFirst = (SeqPosMax < 0);

    mtmd_input_text InputText;
    InputText.text = FormattedPrompt.c_str();
    InputText.add_special = IsFirst;
    InputText.parse_special = true;

    int32_t TokenizeResult = mtmd_tokenize(MtmdContext, Chunks, &InputText, BitmapPtrs.GetData(), BitmapPtrs.Num());

    if (TokenizeResult != 0)
    {
        FString Msg = (TokenizeResult == 1)
            ? TEXT("Number of <__media__> markers does not match number of media entries")
            : TEXT("Image/audio preprocessing error during mtmd_tokenize");
        EmitErrorMessage(Msg, (TokenizeResult == 1) ? 51 : 53, __func__);

        mtmd_input_chunks_free(Chunks);
        for (mtmd_bitmap* B : Bitmaps) { mtmd_bitmap_free(B); }
        return 0;
    }

    // 3. Eval all chunks into KV cache
    // seq_pos_max is the last *occupied* position; next token must go at seq_pos_max + 1.
    // For an empty cache (seq_pos_max == -1), -1 + 1 = 0 which is correct.
    llama_pos NPast = SeqPosMax + 1;
    llama_pos NewNPast = NPast;
    const int32 NCtx = llama_n_ctx(Context);
    const size_t NChunks = mtmd_input_chunks_size(Chunks);
    const size_t NTokensInChunks = mtmd_helper_get_n_tokens(Chunks);

    // Log first 150 chars of the formatted prompt so we can verify the delta content
    {
        std::string Preview = FormattedPrompt.substr(0, std::min((size_t)150, FormattedPrompt.size()));
        UE_LOG(LlamaLog, Log, TEXT("[ProcessMultimodalPrompt] prompt_preview='%hs'"), Preview.c_str());
    }

    const int32 ModelRopeType = (int32)llama_model_rope_type(LlamaModel);
    UE_LOG(LlamaLog, Log, TEXT("[ProcessMultimodalPrompt] n_past=%d n_ctx=%d chunks=%zu tokens_in_chunks=%zu n_batch=%d uses_mrope=%d model_rope_type=%d"),
        (int32)NPast, NCtx, NChunks, NTokensInChunks,
        LastLoadedParams.MaxBatchLength,
        (int32)mtmd_decode_use_mrope(MtmdContext),
        ModelRopeType);

    int32_t EvalResult = mtmd_helper_eval_chunks(
        MtmdContext, Context, Chunks,
        NPast, 0,
        LastLoadedParams.MaxBatchLength,
        bLogitsLast, &NewNPast);

    int32 TokensProcessed = (int32)NTokensInChunks;

    // Cleanup
    mtmd_input_chunks_free(Chunks);
    for (mtmd_bitmap* B : Bitmaps) { mtmd_bitmap_free(B); }

    if (EvalResult != 0)
    {
        EmitErrorMessage(FString::Printf(TEXT("mtmd_helper_eval_chunks failed (ret=%d, n_past=%d, n_ctx=%d, chunks=%zu, tokens=%zu)"),
            EvalResult, (int32)NPast, NCtx, NChunks, NTokensInChunks), 54, __func__);
        return 0;
    }

    // Record the correct next KV position from mtmd (NOT seq_pos_max, which is wrong for M-RoPE
    // because 2D spatial positions from image tokens inflate seq_pos_max beyond the true text position).
    NextGenerationNPast = NewNPast;
    UE_LOG(LlamaLog, Log, TEXT("[ProcessMultimodalPrompt] eval complete: NewNPast=%d seq_pos_max=%d"),
        (int32)NewNPast,
        (int32)llama_memory_seq_pos_max(llama_get_memory(Context), 0));

    const auto StopTime = ggml_time_us();
    const float Duration = (StopTime - StartTime) / 1000000.0f;

    if (OnPromptProcessed)
    {
        float Speed = (Duration > 0.f) ? TokensProcessed / Duration : 0.f;
        OnPromptProcessed(TokensProcessed, Role, Speed);
    }

    return TokensProcessed;
}

std::string FLlamaInternal::InsertMultimodalPrompt(const std::string& TextWithMarkers, const TArray<FLlamaMediaEntry>& MediaEntries, EChatTemplateRole Role, bool bAddAssistantBoS, bool bGenerateReply)
{
    if (!bIsModelLoaded)
    {
        UE_LOG(LlamaLog, Warning, TEXT("Model isn't loaded"));
        return std::string();
    }

    if (!bMtmdLoaded)
    {
        EmitErrorMessage(TEXT("Multimodal projector not loaded. Set MmprojPath in ModelParams before calling LoadModel."), 50, __func__);
        return std::string();
    }

    int32 NewLen = FilledContextCharLength;

    if (!TextWithMarkers.empty())
    {
        Messages.push_back({ RoleForEnum(Role), LLAMA_STRDUP(TextWithMarkers.c_str()) });
        // When generating a reply, always add the assistant BOS so the model responds immediately
        // rather than generating the <|im_start|>assistant token itself (which causes loops on vision models)
        const bool bActualAddAssistantBoS = bAddAssistantBoS || bGenerateReply;
        NewLen = ApplyTemplateToContextHistory(bActualAddAssistantBoS);
    }

    if (NewLen < 0)
    {
        UE_LOG(LlamaLog, Warning, TEXT("Multimodal prompt after templating has an invalid length of %d"), NewLen);
        return std::string();
    }

    //Inject empty think block when thinking is disabled on a thinking-capable model
    const bool bActualThinkingInject = !bThinkingEnabled && (bAddAssistantBoS || bGenerateReply) && bModelSupportsThinking && NewLen > 0;
    if (bActualThinkingInject)
    {
        std::string EmptyThinkBlock = ThinkingOpenTag + "\n\n" + ThinkingCloseTag + "\n\n";
        size_t InjLen = EmptyThinkBlock.size();
        if (ContextHistory.size() < (size_t)NewLen + InjLen)
        {
            ContextHistory.resize(NewLen + InjLen);
        }
        memcpy(ContextHistory.data() + NewLen, EmptyThinkBlock.data(), InjLen);
        NewLen += (int32)InjLen;
    }

    if (NewLen > 0)
    {
        std::string FormattedPrompt(ContextHistory.data() + FilledContextCharLength, ContextHistory.data() + NewLen);
        int32 TokensProcessed = ProcessMultimodalPrompt(FormattedPrompt, MediaEntries, Role, bGenerateReply);
    }

    FilledContextCharLength = NewLen;

    std::string Response;
    if (bGenerateReply)
    {
        ResetGrammarForNewResponse();
        Response = Generate();
    }

    return Response;
}

FLlamaInternal::FLlamaInternal()
{

}

FLlamaInternal::~FLlamaInternal()
{
    OnTokenGenerated = nullptr;
    UnloadModel();
    llama_backend_free();
}
