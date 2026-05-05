// Copyright 2025-current Getnamo.

#include "Embedding/HybridRetriever.h"
#include "Embedding/VectorDatabase.h"
#include "Embedding/BM25Index.h"

void FHybridRetriever::Query(const TArray<float>& QueryEmbedding, const FString& QueryText,
                             int32 K, int32 Candidates, int32 RRFConstant,
                             TArray<int64>& OutIds, TArray<float>& OutScores) const
{
    OutIds.Reset();
    OutScores.Reset();
    if (K <= 0) { return; }

    Candidates = FMath::Max(Candidates, K);
    const float Kf = static_cast<float>(FMath::Max(RRFConstant, 1));

    TMap<int64, float> Fused;

    // Vector side
    if (Vector && Vector->IsInitialized() && QueryEmbedding.Num() > 0)
    {
        TArray<int64> Ids;
        TArray<float> Distances;
        Vector->FindNearestNIds(Ids, Distances, QueryEmbedding, Candidates);
        for (int32 Rank = 0; Rank < Ids.Num(); ++Rank)
        {
            float& Score = Fused.FindOrAdd(Ids[Rank], 0.f);
            Score += 1.f / (Kf + static_cast<float>(Rank + 1));
        }
    }

    // BM25 side
    if (Bm25 && !QueryText.IsEmpty())
    {
        TArray<int64> Ids;
        TArray<float> BmScores;
        Bm25->Query(QueryText, Candidates, Ids, BmScores);
        for (int32 Rank = 0; Rank < Ids.Num(); ++Rank)
        {
            float& Score = Fused.FindOrAdd(Ids[Rank], 0.f);
            Score += 1.f / (Kf + static_cast<float>(Rank + 1));
        }
    }

    if (Fused.Num() == 0) { return; }

    TArray<TPair<int64, float>> Sorted;
    Sorted.Reserve(Fused.Num());
    for (const auto& KV : Fused) { Sorted.Emplace(KV.Key, KV.Value); }
    Sorted.Sort([](const TPair<int64, float>& A, const TPair<int64, float>& B){ return A.Value > B.Value; });

    const int32 Take = FMath::Min(K, Sorted.Num());
    OutIds.Reserve(Take);
    OutScores.Reserve(Take);
    for (int32 i = 0; i < Take; ++i)
    {
        OutIds.Add(Sorted[i].Key);
        OutScores.Add(Sorted[i].Value);
    }
}
