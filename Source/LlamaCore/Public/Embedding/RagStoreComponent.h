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
 * Actor-component wrapper around URagStore. Owns one URagStore subobject for its lifetime
 * so RAG state is rooted in the actor and follows the usual ActorComponent lifecycle —
 * easier to drop into a test map / BP actor than the bare UObject form.
 *
 * Default behavior:
 *  - On BeginPlay (or first call), the inner URagStore is constructed using the configured params.
 *  - If `Embedder` is left null, the component looks for a sibling `ULlamaComponent` on the same actor
 *    that's loaded in embedding mode, and uses it.
 *  - Initialize() must be called before ingest. Set bAutoInitializeOnBeginPlay = true to do it for you
 *    once the embedder reports a valid embedding dimension.
 */
UCLASS(Category = "LLM", BlueprintType, meta = (BlueprintSpawnableComponent))
class LLAMACORE_API URagStoreComponent : public UActorComponent
{
    GENERATED_BODY()
public:
    URagStoreComponent();

    /** Embedding component to drive ingest + query embedding. If null, an attached
     *  ULlamaComponent on the owning actor is used. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "RAG")
    ULlamaComponent* Embedder = nullptr;

    /** Vector index parameters. Set Dimensions to match the embedding model's output dim
     *  (or leave 0 to auto-pull from Embedder->GetEmbeddingDimension() at Initialize-time). */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "RAG")
    FVectorDBParams VectorParams;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "RAG")
    FLlamaChunkerParams ChunkerParams;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "RAG")
    FRagRetrievalParams RetrievalDefaults;

    /** When true, BeginPlay calls Initialize() once the embedder reports a non-zero embedding
     *  dimension. Falls back to manual Initialize() if false or no embedder is bound. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "RAG")
    bool bAutoInitializeOnBeginPlay = true;

    /** Fired on the game thread once an Ingest* call has fully ingested. Mirrors URagStore::OnIngestComplete. */
    UPROPERTY(BlueprintAssignable)
    FOnRagIngestCompleteSignature OnIngestComplete;

    /** Fired on the game thread once a RetrieveAsync call resolves. */
    UPROPERTY(BlueprintAssignable)
    FOnRagRetrievalCompleteSignature OnRetrievalComplete;

    // -- Lifecycle ------------------------------------------------------------

    virtual void BeginPlay() override;
    virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;

    /** Build the inner URagStore + indexes. If VectorParams.Dimensions == 0 and an embedder is set,
     *  pulls the dim from Embedder->GetEmbeddingDimension(). Returns true on success. */
    UFUNCTION(BlueprintCallable, Category = "RAG")
    bool Initialize();

    UFUNCTION(BlueprintPure, Category = "RAG")
    bool IsInitialized() const;

    UFUNCTION(BlueprintPure, Category = "RAG")
    int32 NumChunks() const;

    UFUNCTION(BlueprintCallable, Category = "RAG")
    void Reset();

    // -- Ingest ---------------------------------------------------------------

    UFUNCTION(BlueprintCallable, Category = "RAG")
    void IngestText(const FString& Text, const FString& Source);

    UFUNCTION(BlueprintCallable, Category = "RAG")
    bool IngestFile(const FString& FilePath);

    UFUNCTION(BlueprintCallable, Category = "RAG")
    void IngestDocuments(const TArray<FString>& Texts, const TArray<FString>& Sources);

    /** Walk a directory and ingest every matching file. See URagStore::IngestDirectory. */
    UFUNCTION(BlueprintCallable, Category = "RAG")
    int32 IngestDirectory(const FString& FolderPath,
                          const FString& ExtensionsCsv = TEXT("txt,md"),
                          bool bRecursive = true);

    // -- Retrieval ------------------------------------------------------------

    /** Embed Query through Embedder, retrieve, then broadcast OnRetrievalComplete on the game thread. */
    UFUNCTION(BlueprintCallable, Category = "RAG")
    void RetrieveAsync(const FString& Query, FRagRetrievalParams Params);

    /** Same as RetrieveAsync but uses RetrievalDefaults. Convenience for one-line BP wiring. */
    UFUNCTION(BlueprintCallable, Category = "RAG")
    void RetrieveAsyncDefault(const FString& Query);

    /** Format chunks as a single context string for prompt prepending. */
    UFUNCTION(BlueprintCallable, Category = "RAG")
    FString FormatChunksAsContext(const TArray<FLlamaChunk>& InChunks, const FString& HeaderTemplate) const;

    // -- Persistence ----------------------------------------------------------

    UFUNCTION(BlueprintCallable, Category = "RAG|Persistence")
    bool SaveToFile(const FString& FilePath);

    UFUNCTION(BlueprintCallable, Category = "RAG|Persistence")
    bool LoadFromFile(const FString& FilePath);

    /** Direct access to the underlying store for advanced workflows (custom retrievers, etc). */
    UFUNCTION(BlueprintPure, Category = "RAG")
    URagStore* GetStore() const { return Store; }

protected:
    UPROPERTY(VisibleInstanceOnly, Transient, Category = "RAG")
    URagStore* Store = nullptr;

    /** Locates an embedder on the owning actor if one wasn't set explicitly. */
    ULlamaComponent* ResolveEmbedder();

    void EnsureStore();
    void HookStoreDelegates();

    UFUNCTION()
    void HandleStoreIngestComplete(int32 ChunksAdded);
};
