// Copyright 2025-current Getnamo.

#pragma once

#include "CoreMinimal.h"
#include "UObject/Object.h"
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
};

DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnRagIngestCompleteSignature, int32, ChunksAdded);

/**
 * Engine-side RAG store: combines a vector database, an inverted index for BM25, and
 * the source-chunk metadata. Owns its embedding pipeline through a configured
 * ULlamaComponent (must be loaded in embedding mode).
 *
 * Construct via `NewObject<URagStore>(...)`, set `Embedder` and `VectorParams.Dimensions`
 * to match the embedding model, then call `Initialize()` followed by `IngestText`/`IngestFile`.
 */
UCLASS(Blueprintable, BlueprintType, ClassGroup = "LLM")
class LLAMACORE_API URagStore : public UObject
{
    GENERATED_BODY()
public:
    URagStore();
    virtual ~URagStore();

    /** Component that produces embeddings. Must be loaded with bEmbeddingMode = true. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "RAG")
    ULlamaComponent* Embedder = nullptr;

    /** Vector index parameters. Set Dimensions to match the embedding model before Initialize. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "RAG")
    FVectorDBParams VectorParams;

    /** Chunker parameters used by IngestText / IngestFile. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "RAG")
    FLlamaChunkerParams ChunkerParams;

    /** Default retrieval parameters. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "RAG")
    FRagRetrievalParams RetrievalDefaults;

    /** Fired on the game thread once an IngestText/IngestFile call has fully ingested. */
    UPROPERTY(BlueprintAssignable)
    FOnRagIngestCompleteSignature OnIngestComplete;

    /** Build the empty index. Must be called once before ingesting. */
    UFUNCTION(BlueprintCallable, Category = "RAG")
    void Initialize();

    UFUNCTION(BlueprintPure, Category = "RAG")
    bool IsInitialized() const { return bInitialized; }

    /** Number of chunks currently indexed. */
    UFUNCTION(BlueprintPure, Category = "RAG")
    int32 NumChunks() const { return Chunks.Num(); }

    /** Drops all indexed content. */
    UFUNCTION(BlueprintCallable, Category = "RAG")
    void Reset();

    /** Chunk the text, embed each chunk asynchronously, and add to the indexes. Fires OnIngestComplete on finish. */
    UFUNCTION(BlueprintCallable, Category = "RAG")
    void IngestText(const FString& Text, const FString& Source);

    UFUNCTION(BlueprintCallable, Category = "RAG")
    bool IngestFile(const FString& FilePath);

    /** Chunk multiple documents and embed every resulting chunk in a single batch.
     *  Texts and Sources must be the same length. Fires OnIngestComplete once. */
    UFUNCTION(BlueprintCallable, Category = "RAG")
    void IngestDocuments(const TArray<FString>& Texts, const TArray<FString>& Sources);

    /** Walk a directory and ingest every matching file in one batched embedding round.
     *  @param FolderPath        Absolute or project-relative directory path.
     *  @param ExtensionsCsv     Comma-separated extensions (no dots). Default "txt,md". Empty = all files.
     *  @param bRecursive        Recurse into subdirectories.
     *  @return Number of files queued for ingest. Actual chunk count delivered via OnIngestComplete. */
    UFUNCTION(BlueprintCallable, Category = "RAG")
    int32 IngestDirectory(const FString& FolderPath,
                          const FString& ExtensionsCsv = TEXT("txt,md"),
                          bool bRecursive = true);

    /** C++ retrieval entry. Caller already has a query embedding (e.g. from EmbedTextsAsync). */
    void Retrieve(const TArray<float>& QueryEmbedding, const FString& QueryText,
                  const FRagRetrievalParams& Params, TArray<FLlamaChunk>& OutChunks);

    /** Convenience: embed the query text via Embedder, then retrieve and pass results to OnDone. */
    void RetrieveAsync(const FString& QueryText, const FRagRetrievalParams& Params,
                       TFunction<void(const TArray<FLlamaChunk>&)> OnDone);

    /** Format retrieved chunks as a single context string for prompt prepending. */
    UFUNCTION(BlueprintCallable, Category = "RAG")
    FString FormatChunksAsContext(const TArray<FLlamaChunk>& InChunks, const FString& HeaderTemplate) const;

    /** Persist the entire RAG store (vectors + chunk metadata + BM25 index) to a single .rag file. */
    UFUNCTION(BlueprintCallable, Category = "RAG|Persistence")
    bool SaveToFile(const FString& FilePath);

    UFUNCTION(BlueprintCallable, Category = "RAG|Persistence")
    bool LoadFromFile(const FString& FilePath);

    const TArray<FLlamaChunk>& GetChunks() const { return Chunks; }

private:
    void IngestChunksWithEmbeddings(const TArray<FLlamaChunk>& NewChunks,
                                    const TArray<TArray<float>>& Embeddings);

    bool bInitialized = false;

    TUniquePtr<FVectorDatabase> Vector;
    TUniquePtr<FBM25Index>      Bm25;

    /** Indexed chunk metadata. The id used in Vector / BM25 is `chunk index + 1` (matching the
     *  text-pair id auto-increment scheme so 0 is reserved). */
    UPROPERTY()
    TArray<FLlamaChunk> Chunks;
};
