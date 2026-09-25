// Copyright 2025-current Getnamo.

#pragma once

#include "LlamaDataTypes.h"
#include "HAL/ThreadSafeBool.h"
#include "HAL/ThreadSafeCounter.h"

#include <string>
#include <vector>
#include "llama.h"

struct mtmd_context;
struct common_speculative;
struct common_params_speculative;

/** 
* Uses mostly Llama.cpp native API, meant to be embedded in LlamaNative that wraps 
* unreal threading and data types.
*/
class FLlamaInternal
{
public:
    //Core State
    llama_model* LlamaModel = nullptr;
    llama_context* Context = nullptr;
    llama_sampler* Sampler = nullptr;
    struct common_sampler* CommonSampler = nullptr;

    //Speculative decoding state (see FLLMSpeculativeParams)
    llama_model* DraftModel = nullptr;
    llama_context* DraftContext = nullptr;
    common_speculative* Speculative = nullptr;
    common_params_speculative* SpeculativeParams = nullptr;
    FLLMSpeculativeStats LastSpeculativeStats;

    //Multimodal state
    mtmd_context* MtmdContext = nullptr;
    FThreadSafeBool bMtmdLoaded = false;

    //main streaming callback
    TFunction<void(const std::string& TokenPiece)>OnTokenGenerated = nullptr;
    TFunction<void(int32 TokensProcessed, EChatTemplateRole ForRole, float Speed)>OnPromptProcessed = nullptr;   //useful for waiting for system prompt ready
    TFunction<void(const std::string& Response, float Time, int32 Tokens, float Speed)>OnGenerationComplete = nullptr;

    //NB basic error codes: 1x == Load Error, 2x == Process Prompt error, 3x == Generate error. 1xx == Misc errors
    TFunction<void(const FString& ErrorMessage, int32 ErrorCode)> OnError = nullptr;     //doesn't use std::string due to expected consumer

    //Messaging state (message content is strdup'd; remove entries via ClearMessages)
    std::vector<llama_chat_message> Messages;
    std::vector<char> ContextHistory;

    //Frees and removes Messages[FromIndex..end)
    void ClearMessages(size_t FromIndex = 0);

    //Swap sampling params (incl. grammar) on a loaded model without reloading or touching history
    bool UpdateSamplingParams(const FLLMSamplingParams& Sampling);
    bool HasActiveGrammar() const { return !ActiveGrammar.empty(); }

    //Speculative decoding state checks (the token mirror must track the text KV exactly)
    bool IsSpeculativeActive() const { return Speculative != nullptr; }
    bool IsTokenMirrorConsistent() const;

    //Loaded state
    std::string Template;
    std::string TemplateSource;

    //Thinking mode state (auto-detected from template)
    std::string ThinkingOpenTag;
    std::string ThinkingCloseTag;
    bool bThinkingEnabled = true;
    bool bStripThinkingFromResponse = false;
    bool bModelSupportsThinking = false;

    //Cached params, should be accessed on BT
    FLLMModelParams LastLoadedParams;

    //Model loading
    bool LoadModelFromParams(const FLLMModelParams& InModelParams);
    void UnloadModel();
    bool IsModelLoaded();

    //Generation
    void ResetContextHistory(bool bKeepSystemsPrompt = false);
    void RollbackContextHistoryByTokens(int32 NTokensToErase);
    void RollbackContextHistoryByMessages(int32 NMessagesToErase);

    //raw prompt insert doesn't not update messages, just context history
    std::string InsertRawPrompt(const std::string& Prompt, bool bGenerateReply = true);

    //main function for structure insert and generation
    //AssistantPrefill: optional text prepended into the assistant turn before sampling resumes.
    //  Requires bAddAssistantBoS=true (silently ignored otherwise). The prefill is fed through the
    //  prompt-eval batch (so it lands in KV) and is treated as if the model emitted it: streamed via
    //  OnTokenGenerated, included in the returned response, and stored in the assistant message
    //  history. Useful for steering first-token behavior (e.g. forcing "Answer: " or pre-closing a
    //  thinking block).
    std::string InsertTemplatedPrompt(const std::string& Prompt, EChatTemplateRole Role = EChatTemplateRole::User, bool bAddAssistantBoS = true, bool bGenerateReply = true, const std::string& AssistantPrefill = "");

    //Wipe KV + message state and re-ingest the supplied messages so the KV cache mirrors `Messages`.
    //Each message is fed via InsertTemplatedPrompt(bGenerateReply=false) so the existing template+decode path runs.
    //No reply generation. Caller is responsible for any GT-side state sync.
    void RebuildContextFromHistory(const TArray<FStructuredChatMessage>& Messages);

    //continue generating from last stop
    std::string ResumeGeneration();

    //Feature todo: delete the last message and try again
    //std::string RerollLastGeneration();

    std::string WrapPromptForRole(const std::string& Text, EChatTemplateRole Role, const std::string& OverrideTemplate, bool bAddAssistantBoS = false);


    //flips bGenerationActive which will stop generation on next token. Threadsafe call.
    void StopGeneration();
    bool IsGenerating();

    int32 MaxContext();
    int32 UsedContext();

    FLlamaInternal();
    ~FLlamaInternal();


    //Multimodal
    bool InitMultimodal(const FString& MmprojPath);
    void FreeMultimodal();
    bool IsMultimodalLoaded();
    bool SupportsVision();
    bool SupportsAudio();
    int32 GetAudioSampleRate();

    //Main multimodal prompt entry point
    std::string InsertMultimodalPrompt(const std::string& TextWithMarkers, const TArray<FLlamaMediaEntry>& MediaEntries, EChatTemplateRole Role, bool bAddAssistantBoS, bool bGenerateReply);

    //for embedding models

    //take a prompt and return an array of floats signifying the embeddings.
    //Always returns a single pooled vector of length GetEmbeddingDimension() — for models
    //with LLAMA_POOLING_TYPE_NONE, per-token embeddings are mean-pooled and re-L2-normalized.
    void GetPromptEmbeddings(const std::string& Text, std::vector<float>& Embeddings);

    //Per-vector embedding dimension of the loaded embedding model. 0 if not loaded.
    int32 GetEmbeddingDimension() const;

protected:
    //Wrapper for user<->assistant templated conversation
    int32 ProcessPrompt(const std::string& Prompt, EChatTemplateRole Role = EChatTemplateRole::Unknown);
    int32 ProcessMultimodalPrompt(const std::string& FormattedPrompt, const TArray<FLlamaMediaEntry>& MediaEntries, EChatTemplateRole Role, bool bLogitsLast = true);
    //AssistantPrefill: if non-empty, seeds the response accumulator with this text and emits it
    //  through OnTokenGenerated as if the model produced it. Caller is responsible for having
    //  already fed the prefill through the prompt-eval batch (InsertTemplatedPrompt does this).
    std::string Generate(const std::string& Prompt = "", bool bAppendToMessageHistory = true, const std::string& AssistantPrefill = "");

    void EmitErrorMessage(const FString& ErrorMessage, int32 ErrorCode = -1, const FString& FunctionName = TEXT("unknown"));

    int32 ApplyTemplateToContextHistory(bool bAddAssistantBOS = false);
    int32 ApplyTemplateFromMessagesToBuffer(const std::string& Template, std::vector<llama_chat_message>& FromMessages, std::vector<char>& ToBuffer, bool bAddAssistantBoS = false);

    const char* RoleForEnum(EChatTemplateRole Role);

    //(Re)builds Sampler + CommonSampler from the given params; frees any existing ones first
    void BuildSamplers(const FLLMSamplingParams& Sampling, int32 Seed);

    //Grammar samplers are stateful across a response: restart them at the start of each new reply.
    //(common_sampler_reset doesn't reset the grammar, so the common sampler is rebuilt.)
    void ResetGrammarForNewResponse();

    //Grammar actually applied (validated) by the current samplers; empty if none/invalid
    std::string ActiveGrammar;

    //Speculative decoding
    bool InitSpeculative(const FLLMModelParams& InModelParams);
    void FreeSpeculative();
    bool CanSpeculate() const;
    //Speculative token loop; same contract as Generate's normal loop (emits tokens, advances NPast)
    void GenerateSpeculative(std::string& Response, int32& NDecoded, llama_pos& NPast, bool& bEOGExit);

    //Decodes a text prompt chunk, mirrors it into ContextTokens and feeds the draft side
    int32 DecodePromptChunk(const llama_token* Tokens, int32 NTokens);

    //Decodes tokens at explicit positions through TrackedBatch (logits for the last token only), then
    //mirrors + feeds the draft side. Speculative implementations (e.g. MTP) read the batch's positions
    //and sequence ids, which llama_batch_get_one leaves null.
    int32 DecodeTracked(const llama_token* Tokens, int32 NTokens, llama_pos StartPos);

    //Removes KV / token mirror entries from position FromPos onward on the target and draft contexts
    void TrimContextFrom(llama_pos FromPos);

    //Tokens currently in the text KV cache (seq 0), in order. N-gram drafting looks up proposals in
    //it; kept in step with every decode / rollback / reset
    std::vector<llama_token> ContextTokens;

    //False once the KV holds content the draft side didn't see (image/audio chunks) until the next reset
    bool bSpeculativeInSync = true;

    //Draft output buffer. Reserved up front: llama-common is a separate DLL/heap, so the library must
    //only ever write into existing capacity (it push_backs / resizes within it), never reallocate ours
    std::vector<llama_token> DraftTokens;
    llama_batch SpeculativeBatch = {};
    llama_batch TrackedBatch = {};

    FThreadSafeBool bIsModelLoaded = false;
    int32 FilledContextCharLength = 0;
    FThreadSafeBool bGenerationActive = false;
    enum llama_flash_attn_type SavedFlashAttnType = LLAMA_FLASH_ATTN_TYPE_AUTO;

    // Tracks the next KV position for generation. Must be updated explicitly after
    // multimodal eval (seq_pos_max is wrong for M-RoPE due to 2D spatial positions).
    llama_pos NextGenerationNPast = 0;

    //Embedding Decoding utilities
    void BatchDecodeEmbedding(llama_context* ctx, llama_batch& batch, float* output, int n_seq, int n_embd, int embd_norm, int max_rows = 0);
    void BatchAddSeq(llama_batch& batch, const std::vector<int32_t>& tokens, llama_seq_id seq_id);
};