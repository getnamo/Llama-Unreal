// Copyright 2025-current Getnamo.

#pragma once

#include "CoreMinimal.h"
#include "BM25Index.generated.h"

USTRUCT(BlueprintType)
struct FBM25Params
{
    GENERATED_USTRUCT_BODY();

    /** BM25 k1: term-frequency saturation. Common default 1.5. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "BM25")
    float K1 = 1.5f;

    /** BM25 b: length normalization. 0 = ignore length, 1 = full normalization. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "BM25")
    float B = 0.75f;

    /** If true, drop tokens shorter than 2 chars and ASCII stopwords (the, and, of, ...). */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "BM25")
    bool bFilterStopwords = true;
};

/**
 * In-memory BM25 inverted index with on-the-fly statistics. Tokenizer is model-free
 * (Unicode-aware lowercase + alphanumeric word split) so this index is independent of
 * any loaded LLM. Designed to complement FVectorDatabase for hybrid retrieval.
 *
 * Workflow:
 *   1. AddDocument() per chunk
 *   2. Finalize() once (computes IDF + average doc length)
 *   3. Query() for retrieval
 *
 * Re-indexing after additions: call Finalize() again — IDF weights are recomputed.
 */
class LLAMACORE_API FBM25Index
{
public:
    FBM25Params Params;

    FBM25Index();
    ~FBM25Index();

    /** Add a document. DocId is caller-managed; duplicate ids are merged. */
    void AddDocument(int64 DocId, const FString& Text);

    /** Recompute IDF + AvgDocLen. Required before Query() will return useful scores. */
    void Finalize();

    /** Drops all documents and statistics. */
    void Reset();

    /** Top-K retrieval. Results sorted by descending BM25 score. */
    void Query(const FString& QueryText, int32 K,
               TArray<int64>& OutIds, TArray<float>& OutScores) const;

    int32 NumDocuments() const;

    bool Save(FArchive& Ar);
    bool Load(FArchive& Ar);

    /** Tokenize a string the way this index does. Exposed for tests / debugging. */
    static void Tokenize(const FString& Text, TArray<FString>& OutTokens, bool bFilterStopwords);

private:
    void RebuildStatsForDoc(int64 DocId, const TArray<FString>& Tokens);

    // term -> array of (doc id, term frequency)
    TMap<FString, TArray<TPair<int64, uint16>>> Postings;

    TMap<int64, uint32> DocLengths;     // tokens per doc
    TMap<FString, float> Idf;           // term -> IDF weight
    float AvgDocLen = 0.f;
    bool bFinalized = false;
};
