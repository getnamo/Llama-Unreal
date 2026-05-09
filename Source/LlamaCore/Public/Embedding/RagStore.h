// Copyright 2025-current Getnamo.

#pragma once

#include "CoreMinimal.h"
#include "UObject/Object.h"
#include "LlamaDataTypes.h"
#include "LlamaDualBackend.h"
#include "Embedding/CorpusChunker.h"
#include "Embedding/VectorDatabase.h"
#include "Embedding/BM25Index.h"
#include "RagStore.generated.h"

class ULlamaComponent;

UENUM(BlueprintType)
enum class ERagRetrievalMode : uint8
{
    /** Pure dense retrieval against the embedding index. */
    Vector,
    /** Lexical BM25 retrieval against the inverted index. */
    BM25,
    /** Reciprocal Rank Fusion of vector + BM25. Recommended default for mixed content. */
    Hybrid
};

USTRUCT(BlueprintType)
struct FRagRetrievalParams
{
    GENERATED_USTRUCT_BODY();

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "RAG")
    ERagRetrievalMode Mode = ERagRetrievalMode::Hybrid;

    /** Number of chunks to return to the caller. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "RAG")
    int32 TopK = 5;

    /** Per-side candidate pool size before fusion. Only used in Hybrid mode. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "RAG")
    int32 CandidatesPerSide = 50;

    /** RRF k-constant. Default 60 is the value from the original RRF paper. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "RAG")
    int32 RRFConstant = 60;

    /** Drop chunks whose normalized Confidence falls below this threshold.
     *  Range 0..1. Default 0 = no filter. Top-1 is always retained when results exist
     *  (its Confidence is 1.0 by construction); the filter only trims the tail. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "RAG", meta = (ClampMin = "0.0", ClampMax = "1.0"))
    float MinConfidence = 0.f;
};

DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnRagIngestCompleteSignature, int32, ChunksAdded);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnRagAskRetrievedSignature, const TArray<FLlamaChunk>&, RetrievedChunks);
DECLARE_DYNAMIC_MULTICAST_DELEGATE(FOnRagPipelineReadySignature);

/**
 * Engine-side RAG store. Owns its own embedding model (FLlamaDualBackend in embedding
 * mode) and, optionally, an answer model for the streaming Ask() pipeline. External
 * `ULlamaComponent` references can override either backend for power-user flows
 * (sharing one loaded embedder across multiple stores, routing answers through a
 * pre-existing chat actor, etc).
 *
 * Quickstart:
 *   1. Set EmbeddingModelParams.PathToModel (and optionally AnswerModelParams.PathToModel).
 *   2. Call LoadModels(). Wait for the model load to finish (poll IsEmbedderReady()
 *      or bind OnEmbedderLoaded — or just call Initialize+Ingest after a short delay
 *      from the OnEmbedderLoaded callback).
 *   3. Initialize() (or it's called for you on first Ingest if EmbeddingModelParams
 *      is configured).
 *   4. IngestText/IngestFile/IngestDirectory to populate the index.
 *   5. AskDefault(query) to retrieve+answer in one shot, OR RetrieveAsync(query, params)
 *      for direct chunk access.
 */
UCLASS(Blueprintable, BlueprintType, ClassGroup = "LLM")
class LLAMACORE_API URagStore : public UObject
{
    GENERATED_BODY()
public:
    URagStore();
    virtual ~URagStore();

    // ── Embedder configuration ──────────────────────────────────────────────

    /** Embedding model parameters. PathToModel drives the internal embedder backend.
     *  Advanced.bEmbeddingMode is force-set true at LoadModels() time regardless of the
     *  user-supplied value. Leave PathToModel empty to defer to ExternalEmbedder. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "RAG|Embedding")
    FLLMModelParams EmbeddingModelParams;

    /** Optional override: if set, embedding queries are routed through this component
     *  instead of the internal embedder. Useful when one embedding model is shared
     *  across multiple RagStores to save VRAM. The component must already be loaded
     *  with bEmbeddingMode = true. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "RAG|Embedding")
    ULlamaComponent* ExternalEmbedder = nullptr;

    // ── Answer (chat) configuration ─────────────────────────────────────────

    /** Chat model parameters for the Ask() pipeline. PathToModel drives the internal
     *  answer backend. Leave empty to defer to AnswerEngine (or to disable Ask()). */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "RAG|Answer")
    FLLMModelParams AnswerModelParams;

    /** Optional override: used when AnswerModelParams.PathToModel is empty. The Ask()
     *  pipeline routes the formatted prompt through this component's InsertTemplatedPrompt. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "RAG|Answer")
    ULlamaComponent* AnswerEngine = nullptr;

    /** Template applied to the formatted-context + query before Ask() sends it to the
     *  answer model. `{context}` is substituted with the retrieved chunks; `{query}`
     *  with the user's question. Substitution is plain string replace, not full
     *  templating — only these two tokens are honored.
     *
     *  The first-token-EOT primer (e.g. "Answer:") that some instruction-tuned models
     *  need to avoid empty completions is no longer baked into this template; it is
     *  applied via AnswerPrefill (see below) which is positioned correctly inside the
     *  assistant turn rather than smuggled at the tail of the user message. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "RAG|Answer", meta = (MultiLine = true))
    FString SummarizingPromptTemplate =
        TEXT("Use only the following context to answer the question. ")
        TEXT("If the answer isn't in the context, say so plainly.\n\n")
        TEXT("Context:\n{context}\n\nQuestion: {query}");

    /** Optional assistant-turn prefill prepended to every Ask() answer. The text is
     *  inserted into the assistant turn (after the BOS header, before sampling) so the
     *  model continues from it as if it generated the prefill itself. Streamed through
     *  OnAskTokenGenerated/OnAskPartialGenerated and included in OnAskResponseGenerated.
     *
     *  Defaults to "Answer: " — a cue that primes the model to produce the answer body
     *  and works around the first-token-EOT quirk on instruction-tuned chat models
     *  (notably Gemma3). Set to empty for "raw" generation. To hard-suppress thinking
     *  on a thinking-capable model, set to "<think></think>\n\n". */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "RAG|Answer", meta = (MultiLine = true))
    FString AnswerPrefill = TEXT("Answer: ");

    // ── Index configuration ─────────────────────────────────────────────────

    /** Vector index parameters. With `bSyncVectorDimToEmbedder = true` (default),
     *  Dimensions is auto-overwritten from the loaded embedder during Initialize() —
     *  ignore the field unless you're loading from a saved file or using an external
     *  embedder that hasn't loaded yet. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "RAG")
    FVectorDBParams VectorParams;

    /** When true (default), Initialize() overwrites VectorParams.Dimensions with the
     *  loaded embedder's actual output dim. Set false if you want full manual control
     *  (e.g. you're using a saved index whose dim must stay pinned regardless of which
     *  embedder is loaded). When off, a mismatch triggers a warning instead. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "RAG")
    bool bSyncVectorDimToEmbedder = true;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "RAG")
    FLlamaChunkerParams ChunkerParams;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "RAG")
    FRagRetrievalParams RetrievalDefaults;

    // ── Ingest delegates ────────────────────────────────────────────────────

    UPROPERTY(BlueprintAssignable)
    FOnRagIngestCompleteSignature OnIngestComplete;

    // ── Ask pipeline delegates ──────────────────────────────────────────────

    /** Fired with the retrieved chunks before the answer model sees them. Useful for
     *  showing source citations in UI as soon as retrieval completes. Only broadcasts
     *  when `bBroadcastChunksOnAsk = true` — opt in for debug / citation overlays. */
    UPROPERTY(BlueprintAssignable)
    FOnRagAskRetrievedSignature OnAskRetrievedChunks;

    /** When true, Ask()/AskDefault() broadcast OnAskRetrievedChunks with the retrieved
     *  context chunks before the answer streams. Default false so the standard answer
     *  flow is quiet — flip on to peek at what's feeding the prompt for debugging or
     *  citation UI. Direct RetrieveAsync() always returns chunks regardless. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "RAG|Answer")
    bool bBroadcastChunksOnAsk = false;

    /** Token-level streaming from the answer model. */
    UPROPERTY(BlueprintAssignable)
    FOnTokenGeneratedSignature OnAskTokenGenerated;

    /** Sentence-level partials (uses ModelParams.Advanced.Output.PartialsSeparators). */
    UPROPERTY(BlueprintAssignable)
    FOnPartialSignature OnAskPartialGenerated;

    /** Markdown-aware partials, if Advanced.Markdown.bSplitMarkdown is on. */
    UPROPERTY(BlueprintAssignable)
    FOnMarkdownPartialSignature OnAskMarkdownPartialGenerated;

    /** Final, complete answer (markdown thinking-tag stripping respected). */
    UPROPERTY(BlueprintAssignable)
    FOnResponseGeneratedSignature OnAskResponseGenerated;

    UPROPERTY(BlueprintAssignable)
    FOnEndOfStreamSignature OnAskEndOfStream;

    UPROPERTY(BlueprintAssignable)
    FOnErrorSignature OnAskError;

    // ── Lifecycle ───────────────────────────────────────────────────────────

    /** Fires once when the full RAG pipeline is ready: every configured internal model has
     *  loaded AND `Initialize()` has run. Subscribe before calling `LoadAndInitialize()`. */
    UPROPERTY(BlueprintAssignable)
    FOnRagPipelineReadySignature OnRagPipelineReady;

    /** Loads internally-owned models (embedder iff EmbeddingModelParams.PathToModel non-empty,
     *  answerer iff AnswerModelParams.PathToModel non-empty). Loads are SERIALIZED — the
     *  embedder finishes before the answerer kicks off — to avoid concurrent llama.cpp /
     *  Vulkan backend init races. Idempotent. */
    UFUNCTION(BlueprintCallable, Category = "RAG")
    void LoadModels();

    /** Convenience: `LoadModels()` + auto-`Initialize()` once the embedder is ready, with
     *  `OnRagPipelineReady` broadcast on completion. The recommended one-call setup path. */
    UFUNCTION(BlueprintCallable, Category = "RAG")
    void LoadAndInitialize();

    /** True iff the embedder (internal or external) is ready to produce embeddings. */
    UFUNCTION(BlueprintPure, Category = "RAG")
    bool IsEmbedderReady() const;

    /** True iff some answer pathway (internal or external) is available for Ask(). */
    UFUNCTION(BlueprintPure, Category = "RAG")
    bool IsAnswerEngineReady() const;

    /** Build the empty index. Must be called once before ingesting. Auto-pulls
     *  VectorParams.Dimensions from the loaded embedder if unset. */
    UFUNCTION(BlueprintCallable, Category = "RAG")
    void Initialize();

    UFUNCTION(BlueprintPure, Category = "RAG")
    bool IsInitialized() const { return bInitialized; }

    UFUNCTION(BlueprintPure, Category = "RAG")
    int32 NumChunks() const { return Chunks.Num(); }

    UFUNCTION(BlueprintCallable, Category = "RAG")
    void Reset();

    // ── Ingest ──────────────────────────────────────────────────────────────

    UFUNCTION(BlueprintCallable, Category = "RAG")
    void IngestText(const FString& Text, const FString& Source);

    UFUNCTION(BlueprintCallable, Category = "RAG")
    bool IngestFile(const FString& FilePath);

    UFUNCTION(BlueprintCallable, Category = "RAG")
    void IngestDocuments(const TArray<FString>& Texts, const TArray<FString>& Sources);

    UFUNCTION(BlueprintCallable, Category = "RAG")
    int32 IngestDirectory(const FString& FolderPath,
                          const FString& ExtensionsCsv = TEXT("txt,md"),
                          bool bRecursive = true);

    // ── Retrieve ────────────────────────────────────────────────────────────

    /** C++ retrieval entry. Caller already has a query embedding (e.g. computed externally). */
    void Retrieve(const TArray<float>& QueryEmbedding, const FString& QueryText,
                  const FRagRetrievalParams& Params, TArray<FLlamaChunk>& OutChunks);

    /** Embed the query through the configured embedder, then retrieve. Results delivered
     *  via OnDone on the game thread. */
    void RetrieveAsync(const FString& QueryText, const FRagRetrievalParams& Params,
                       TFunction<void(const TArray<FLlamaChunk>&)> OnDone);

    // ── Ask (full retrieve + answer pipeline) ───────────────────────────────

    /** End-to-end: embed `Query`, retrieve top-K, format with SummarizingPromptTemplate,
     *  send to the answer model, stream results through the OnAsk* delegates.
     *  ParamsOverride lets callers customize retrieval per-call; pass a default-constructed
     *  FRagRetrievalParams to inherit RetrievalDefaults. */
    UFUNCTION(BlueprintCallable, Category = "RAG")
    void Ask(const FString& Query, FRagRetrievalParams ParamsOverride);

    /** Convenience: Ask() using RetrievalDefaults. */
    UFUNCTION(BlueprintCallable, Category = "RAG")
    void AskDefault(const FString& Query);

    // ── Utilities ───────────────────────────────────────────────────────────

    UFUNCTION(BlueprintCallable, Category = "RAG")
    FString FormatChunksAsContext(const TArray<FLlamaChunk>& InChunks, const FString& HeaderTemplate) const;

    UFUNCTION(BlueprintCallable, Category = "RAG|Persistence")
    bool SaveToFile(const FString& FilePath);

    UFUNCTION(BlueprintCallable, Category = "RAG|Persistence")
    bool LoadFromFile(const FString& FilePath);

    const TArray<FLlamaChunk>& GetChunks() const { return Chunks; }

    /** Direct ingest path for advanced consumers that compute embeddings themselves. */
    void IngestChunksWithEmbeddings(const TArray<FLlamaChunk>& NewChunks,
                                    const TArray<TArray<float>>& Embeddings);

protected:
    virtual void BeginDestroy() override;

    // Relay handlers for the AnswerEngine (external ULlamaComponent) path. These are
    // UFUNCTION so they can subscribe to the component's dynamic multicasts; each gates
    // on bAskInFlight to avoid forwarding unrelated direct-chat output as Ask responses.
    UFUNCTION() void RelayAnswerToken(const FString& Token);
    UFUNCTION() void RelayAnswerPartial(const FString& Partial);
    UFUNCTION() void RelayAnswerMarkdownPartial(const FString& Partial, EMarkdownStreamState State);
    UFUNCTION() void RelayAnswerResponse(const FString& Response);
    UFUNCTION() void RelayAnswerEndOfStream(bool bStopSequenceTriggered, float TokensPerSecond);
    UFUNCTION() void RelayAnswerError(const FString& ErrorMessage, int32 ErrorCode);

private:
    /** Async embed via whichever embedder is configured. Falls back through:
     *  ExternalEmbedder → InternalEmbedder. Calls OnDone with the embeddings (or empty
     *  arrays on error). */
    void EmbedTextsViaActiveEmbedder(const TArray<FString>& Texts,
        TFunction<void(const TArray<TArray<float>>&, const TArray<FString>&)> OnDone);

    /** Send a fully-formatted prompt to whichever answer engine is configured.
     *  Wires the engine's streaming callbacks to OnAsk* delegates for the duration. */
    void SendFormattedPromptToActiveAnswerer(const FString& FormattedPrompt);

    /** Substitutes {context} and {query} in SummarizingPromptTemplate. */
    FString BuildSummarizingPrompt(const FString& Query, const TArray<FLlamaChunk>& InChunks) const;

    /** Build (if needed) and kick off the embedder load. Calls OnEmbedderDone when it lands. */
    void LoadEmbedderInternal();
    /** Build (if needed) and kick off the answerer load. Calls OnAnswererDone when it lands. */
    void LoadAnswererInternal();
    /** Called when the chain in LoadAndInitialize finishes both model loads. */
    void OnAllInternalLoadsComplete();

    /** True when LoadAndInitialize is mid-chain — Initialize() is deferred until both loads land. */
    bool bAutoInitPending = false;

    bool bInitialized = false;

    /** Internal owned backends. Created by LoadModels iff their PathToModel is non-empty. */
    TUniquePtr<FLlamaDualBackend> InternalEmbedder;
    TUniquePtr<FLlamaDualBackend> InternalAnswerer;
    bool bInternalEmbedderReady = false;
    bool bInternalAnswererReady = false;
    /** Active answer-side stream — true between Ask() invocation and the OnEndOfStream
     *  forwarded broadcast, used to swallow stray callbacks if the user calls Ask in
     *  rapid succession. */
    bool bAskInFlight = false;

    TUniquePtr<FVectorDatabase> Vector;
    TUniquePtr<FBM25Index>      Bm25;

    UPROPERTY()
    TArray<FLlamaChunk> Chunks;
};
