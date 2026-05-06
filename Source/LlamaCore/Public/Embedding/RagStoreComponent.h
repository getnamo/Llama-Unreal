// Copyright 2025-current Getnamo.

#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "Embedding/RagStore.h"
#include "RagStoreComponent.generated.h"

class ULlamaComponent;

DECLARE_DYNAMIC_MULTICAST_DELEGATE_TwoParams(FOnRagRetrievalCompleteSignature,
    const TArray<FLlamaChunk>&, RetrievedChunks, const FString&, Query);

/**
 * Actor-component wrapper around URagStore. Owns one URagStore subobject for its
 * lifetime so RAG state is rooted in the actor and follows the usual ActorComponent
 * lifecycle — easier to drop into a test map / BP actor than the bare UObject form.
 *
 * Quickstart for users:
 *  1. Configure EmbeddingModelParams.PathToModel (and optionally AnswerModelParams.PathToModel
 *     for the streaming Ask() pipeline).
 *  2. Place on an actor. With bAutoInitializeOnBeginPlay = true (default), BeginPlay
 *     calls LoadModels(). Once the embedder reports its dimension, it auto-Initializes.
 *  3. Call IngestText/IngestDirectory to populate.
 *  4. Bind OnAskTokenGenerated / OnAskResponseGenerated, then call AskDefault(query).
 *
 * Power-user paths:
 *  - Set ExternalEmbedder to share a pre-loaded ULlamaComponent across stores.
 *  - Set AnswerEngine to route Ask() through a chat actor that already exists.
 *  - If neither EmbeddingModelParams.PathToModel nor ExternalEmbedder is set, BeginPlay
 *    looks for a sibling ULlamaComponent on the same actor that's loaded in embedding
 *    mode and uses that as ExternalEmbedder (back-compat / discovery shortcut).
 */
UCLASS(Category = "LLM", BlueprintType, meta = (BlueprintSpawnableComponent))
class LLAMACORE_API URagStoreComponent : public UActorComponent
{
    GENERATED_BODY()
public:
    URagStoreComponent();

    // ── Embedder / Answer config (mirror of URagStore) ──────────────────────

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "RAG|Embedding")
    FLLMModelParams EmbeddingModelParams;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "RAG|Embedding")
    ULlamaComponent* ExternalEmbedder = nullptr;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "RAG|Answer")
    FLLMModelParams AnswerModelParams;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "RAG|Answer")
    ULlamaComponent* AnswerEngine = nullptr;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "RAG|Answer", meta = (MultiLine = true))
    FString SummarizingPromptTemplate =
        TEXT("Use only the following context to answer the question. ")
        TEXT("If the answer isn't in the context, say so plainly.\n\n")
        TEXT("Context:\n{context}\n\nQuestion: {query}");

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "RAG")
    FVectorDBParams VectorParams;

    /** When true (default), Initialize() overwrites VectorParams.Dimensions with the
     *  embedder's actual output dim. See URagStore::bSyncVectorDimToEmbedder. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "RAG")
    bool bSyncVectorDimToEmbedder = true;

    /** When true, Ask()/AskDefault() broadcast OnAskRetrievedChunks before streaming the
     *  answer. Default false; flip on for debugging or citation UI. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "RAG|Answer")
    bool bBroadcastChunksOnAsk = false;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "RAG")
    FLlamaChunkerParams ChunkerParams;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "RAG")
    FRagRetrievalParams RetrievalDefaults;

    /** When true, BeginPlay calls LoadModels() and (when an embedder is ready)
     *  Initialize(). Set false for fully-manual control. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "RAG")
    bool bAutoInitializeOnBeginPlay = true;

    // ── Delegates (mirror inner store) ──────────────────────────────────────

    UPROPERTY(BlueprintAssignable)
    FOnRagIngestCompleteSignature OnIngestComplete;

    UPROPERTY(BlueprintAssignable)
    FOnRagRetrievalCompleteSignature OnRetrievalComplete;

    UPROPERTY(BlueprintAssignable) FOnRagAskRetrievedSignature      OnAskRetrievedChunks;
    UPROPERTY(BlueprintAssignable) FOnTokenGeneratedSignature       OnAskTokenGenerated;
    UPROPERTY(BlueprintAssignable) FOnPartialSignature              OnAskPartialGenerated;
    UPROPERTY(BlueprintAssignable) FOnMarkdownPartialSignature      OnAskMarkdownPartialGenerated;
    UPROPERTY(BlueprintAssignable) FOnResponseGeneratedSignature    OnAskResponseGenerated;
    UPROPERTY(BlueprintAssignable) FOnEndOfStreamSignature          OnAskEndOfStream;
    UPROPERTY(BlueprintAssignable) FOnErrorSignature                OnAskError;

    /** Fires once when the full pipeline (models loaded + Initialize() complete) is ready
     *  to use. Bind before calling LoadAndInitialize() (or before BeginPlay if you're
     *  relying on bAutoInitializeOnBeginPlay). */
    UPROPERTY(BlueprintAssignable)
    FOnRagPipelineReadySignature OnRagPipelineReady;

    // ── Lifecycle ───────────────────────────────────────────────────────────

    virtual void BeginPlay() override;
    virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;
    virtual void TickComponent(float DeltaTime, ELevelTick TickType,
                               FActorComponentTickFunction* ThisTickFunction) override;

    UFUNCTION(BlueprintCallable, Category = "RAG")
    void LoadModels();

    /** One-call setup: load embedder + answerer (sequential, no concurrent backend init),
     *  auto-Initialize when models are ready, broadcast OnRagPipelineReady. */
    UFUNCTION(BlueprintCallable, Category = "RAG")
    void LoadAndInitialize();

    UFUNCTION(BlueprintCallable, Category = "RAG")
    bool Initialize();

    UFUNCTION(BlueprintPure, Category = "RAG")
    bool IsInitialized() const;

    UFUNCTION(BlueprintPure, Category = "RAG")
    int32 NumChunks() const;

    UFUNCTION(BlueprintPure, Category = "RAG")
    bool IsEmbedderReady() const;

    UFUNCTION(BlueprintPure, Category = "RAG")
    bool IsAnswerEngineReady() const;

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

    // ── Retrieval ───────────────────────────────────────────────────────────

    UFUNCTION(BlueprintCallable, Category = "RAG")
    void RetrieveAsync(const FString& Query, FRagRetrievalParams Params);

    UFUNCTION(BlueprintCallable, Category = "RAG")
    void RetrieveAsyncDefault(const FString& Query);

    // ── Ask (full retrieve + answer pipeline) ───────────────────────────────

    UFUNCTION(BlueprintCallable, Category = "RAG")
    void Ask(const FString& Query, FRagRetrievalParams Params);

    UFUNCTION(BlueprintCallable, Category = "RAG")
    void AskDefault(const FString& Query);

    // ── Utilities / persistence ─────────────────────────────────────────────

    UFUNCTION(BlueprintCallable, Category = "RAG")
    FString FormatChunksAsContext(const TArray<FLlamaChunk>& InChunks, const FString& HeaderTemplate) const;

    UFUNCTION(BlueprintCallable, Category = "RAG|Persistence")
    bool SaveToFile(const FString& FilePath);

    UFUNCTION(BlueprintCallable, Category = "RAG|Persistence")
    bool LoadFromFile(const FString& FilePath);

    UFUNCTION(BlueprintPure, Category = "RAG")
    URagStore* GetStore() const { return Store; }

protected:
    UPROPERTY(VisibleInstanceOnly, Transient, Category = "RAG")
    URagStore* Store = nullptr;

    /** Auto-discover a sibling ULlamaComponent loaded in embedding mode and use it
     *  as ExternalEmbedder. Called from BeginPlay only when neither
     *  EmbeddingModelParams.PathToModel nor ExternalEmbedder is set. */
    ULlamaComponent* AutoDiscoverEmbedder();

    void EnsureStore();
    void HookStoreDelegates();
    void SyncStoreConfig();

    UFUNCTION()
    void HandleStoreIngestComplete(int32 ChunksAdded);

    UFUNCTION()
    void HandleStoreRagPipelineReady();

    // Re-broadcast helpers for inner store's OnAsk* events.
    UFUNCTION() void HandleStoreAskRetrieved(const TArray<FLlamaChunk>& RetrievedChunks);
    UFUNCTION() void HandleStoreAskToken(const FString& Token);
    UFUNCTION() void HandleStoreAskPartial(const FString& Partial);
    UFUNCTION() void HandleStoreAskMarkdownPartial(const FString& Partial, EMarkdownStreamState State);
    UFUNCTION() void HandleStoreAskResponse(const FString& Response);
    UFUNCTION() void HandleStoreAskEndOfStream(bool bStopSequenceTriggered, float TokensPerSecond);
    UFUNCTION() void HandleStoreAskError(const FString& ErrorMessage, int32 ErrorCode);
};
