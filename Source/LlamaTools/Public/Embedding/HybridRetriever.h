// Copyright 2025-current Getnamo.

#pragma once

#include "CoreMinimal.h"

class FVectorDatabase;
class FBM25Index;

/**
 * Reciprocal Rank Fusion (RRF) of vector + BM25 retrievers. Parameter-free across
 * heterogeneous score scales — robust default for hybrid local RAG.
 *
 *   score_rrf(d) = Σ_r 1 / (k + rank_r(d))    (k=60 in the original paper)
 *
 * Pulls Candidates from each side, fuses, returns top K.
 */
class LLAMATOOLS_API FHybridRetriever
{
public:
    FVectorDatabase* Vector = nullptr;   // not owned
    FBM25Index*      Bm25   = nullptr;   // not owned

    /**
     * @param QueryEmbedding  vector for dense retrieval; pass empty TArray to disable vector side
     * @param QueryText       raw text for BM25 retrieval; pass empty FString to disable BM25 side
     * @param K               number of fused results to return
     * @param Candidates      per-side candidate pool size before fusion (e.g. 50)
     * @param RRFConstant     RRF k constant; 60 is the canonical default
     */
    void Query(const TArray<float>& QueryEmbedding, const FString& QueryText,
               int32 K, int32 Candidates, int32 RRFConstant,
               TArray<int64>& OutIds, TArray<float>& OutScores) const;
};
