// Copyright 2025-current Getnamo.

#pragma once

#include "CoreMinimal.h"
#include "HAL/CriticalSection.h"
#include "VectorDatabase.generated.h"

USTRUCT(BlueprintType)
struct FVectorDBParams
{
    GENERATED_USTRUCT_BODY();

    // Dimension of each embedding vector. Must match `llama_model_n_embd` of the embedding model
    // (e.g. 384 for bge-small/MiniLM, 768 for nomic-embed, 1024 for Qwen3-Embedding-0.6B).
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VectorDB Params")
    int32 Dimensions = 384;

    // Maximum number of elements the index can hold; pre-allocated.
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VectorDB Params")
    int32 MaxElements = 10000;

    // HNSW graph connectivity. 16 is a common default; higher = more accurate, more memory.
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VectorDB Params")
    int32 M = 16;

    // Build-time search depth. Higher = better recall but slower build.
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VectorDB Params")
    int32 EFConstruction = 200;

    // Query-time search depth. Higher = better recall but slower queries.
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VectorDB Params")
    int32 EFQuery = 64;
};


/**
 * Native HNSW-backed vector store for k-nearest-neighbor retrieval over high-dimensional
 * float embeddings. L2 distance metric (works as cosine when input is L2-normalized — which
 * `FLlamaInternal::GetPromptEmbeddings` produces by default).
 *
 * Thread-safety: hnswlib's add/search are concurrent-safe on the same instance. The
 * accompanying TextDatabase is guarded internally by a critical section.
 *
 * Persistence: `Save()`/`Load()` write a single binary file containing both the HNSW
 * index and the text-database sidecar. Versioned with a magic header.
 */
class LLAMATOOLS_API FVectorDatabase
{
public:
    FVectorDBParams Params;

    FVectorDatabase();
    ~FVectorDatabase();

    /** Build an empty index from the current Params. Required before adding vectors. */
    void InitializeDB();

    /** Returns true if InitializeDB() (or Load()) has completed successfully. */
    bool IsInitialized() const;

    /** Number of vectors currently in the index. */
    int32 Num() const;

    /** Drops the index and the text database. Index becomes uninitialized. */
    void Reset();

    // ---- Add ----------------------------------------------------------------

    /** Add a vector with a caller-managed unique id. Embedding.Num() must equal Params.Dimensions. */
    void AddVectorEmbeddingIdPair(const TArray<float>& Embedding, int64 UniqueId);

    /**
     * Add a vector + associated text snippet. An auto-incrementing id is assigned
     * and the text is stored in the side table; queries can return either.
     * Returns the id assigned, or -1 on failure.
     */
    int64 AddVectorEmbeddingStringPair(const TArray<float>& Embedding, const FString& Text);

    // ---- Query --------------------------------------------------------------

    /** Top-1 id lookup. Returns -1 if not initialized or empty. */
    int64 FindNearestId(const TArray<float>& ForEmbedding);

    /** Top-1 string lookup. Returns empty string if no string was associated or index is empty. */
    FString FindNearestString(const TArray<float>& ForEmbedding);

    /** Top-N id lookup. Results are sorted nearest-first (index 0 = best match). */
    void FindNearestNIds(TArray<int64>& OutIds, const TArray<float>& ForEmbedding, int32 N = 1);

    /** Top-N id+distance lookup. Results sorted nearest-first; OutDistances aligned with OutIds. */
    void FindNearestNIds(TArray<int64>& OutIds, TArray<float>& OutDistances,
                         const TArray<float>& ForEmbedding, int32 N = 1);

    /** Top-N string lookup. Skips ids that have no associated string. Sorted nearest-first. */
    void FindNearestNStrings(TArray<FString>& OutStrings, const TArray<float>& ForEmbedding, int32 N = 1);

    /** Top-N string + distance lookup. */
    void FindNearestNStrings(TArray<FString>& OutStrings, TArray<float>& OutDistances,
                             const TArray<float>& ForEmbedding, int32 N = 1);

    /** Lookup the text for a given id. Returns true and sets OutText on hit. */
    bool TryGetText(int64 UniqueId, FString& OutText) const;

    // ---- Persistence --------------------------------------------------------

    /** Persist the full database (HNSW + text sidecar + Params) to a single .vdb file. */
    bool Save(const FString& FilePath) const;

    /** Restore a database previously written by Save(). Replaces current state. */
    bool Load(const FString& FilePath);

    // ---- Diagnostics --------------------------------------------------------

    /** Self-recall sanity check used during development. Logs recall, returns it. */
    float BasicsTest();

private:
    class FHNSWPrivate* Private = nullptr;

    // Maps UniqueId -> raw text snippet. -1 reserved as sentinel.
    TMap<int64, FString> TextDatabase;
    int64 TextDatabaseMaxId = 0;

    // Guards TextDatabase + initialization state. HNSW itself is internally thread-safe.
    mutable FCriticalSection TextLock;

    bool bInitialized = false;
};
