// Copyright 2025-current Getnamo.

#include "Misc/AutomationTest.h"

#if WITH_DEV_AUTOMATION_TESTS

#include "Embedding/RagStore.h"
#include "Embedding/CorpusChunker.h"

#include <initializer_list>

namespace
{
    static FLlamaChunk MakeChunk(const TCHAR* Text, const TCHAR* Source)
    {
        FLlamaChunk C;
        C.Text   = Text;
        C.Source = Source;
        return C;
    }

    static TArray<float> Vec(int32 D, std::initializer_list<float> NonZeroPrefix)
    {
        TArray<float> V;
        V.Init(0.f, D);
        int32 i = 0;
        for (float Val : NonZeroPrefix) { if (i < D) V[i++] = Val; }
        return V;
    }
}

/**
 * End-to-end test of URagStore's score-normalization layer in vector mode.
 * Synthesizes a 4-doc corpus where the embeddings put doc 0 closest to a
 * known query, and verifies:
 *   - Top-1 has Confidence == 1.0
 *   - Confidences descend monotonically
 *   - SourceRetriever is correctly tagged
 *   - RetrievalScore (raw L2 distance) is preserved on each chunk
 *   - MinConfidence pre-filter trims the tail without dropping top-1
 */
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FRagScoreNormalizationVectorModeTest,
    "LlamaTools.RAG.ScoreNormalizationVector",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FRagScoreNormalizationVectorModeTest::RunTest(const FString& /*Parameters*/)
{
    constexpr int32 D = 8;

    URagStore* Store = NewObject<URagStore>();
    Store->VectorParams.Dimensions  = D;
    Store->VectorParams.MaxElements = 16;
    Store->Initialize();

    // 4 docs along a 1-d line (only first dim varies).
    TArray<FLlamaChunk> NewChunks = {
        MakeChunk(TEXT("doc-zero"),  TEXT("zero.md")),
        MakeChunk(TEXT("doc-one"),   TEXT("one.md")),
        MakeChunk(TEXT("doc-two"),   TEXT("two.md")),
        MakeChunk(TEXT("doc-three"), TEXT("three.md")),
    };
    TArray<TArray<float>> Embeddings = {
        Vec(D, {0.0f}),
        Vec(D, {0.5f}),
        Vec(D, {1.5f}),
        Vec(D, {3.0f}),
    };

    Store->IngestChunksWithEmbeddings(NewChunks, Embeddings);
    TestEqual(TEXT("All 4 chunks ingested"), Store->NumChunks(), 4);

    FRagRetrievalParams Params;
    Params.Mode = ERagRetrievalMode::Vector;
    Params.TopK = 4;
    Params.MinConfidence = 0.f;

    TArray<FLlamaChunk> Out;
    Store->Retrieve(/*Query*/ Vec(D, {0.0f}), /*QueryText*/ TEXT(""), Params, Out);

    TestEqual(TEXT("Got 4 results"), Out.Num(), 4);
    if (Out.Num() < 4) return false;

    TestEqual(TEXT("Top-1 source = zero.md"),    Out[0].Source, FString(TEXT("zero.md")));
    TestEqual(TEXT("Top-1 SourceRetriever = Vector"),
        (uint8)Out[0].SourceRetriever, (uint8)ERagRetrievalSource::Vector);
    TestTrue(TEXT("Top-1 Confidence == 1.0"), FMath::IsNearlyEqual(Out[0].Confidence, 1.f, 1e-4f));

    for (int32 i = 1; i < Out.Num(); ++i)
    {
        TestTrue(FString::Printf(TEXT("Confidence descending at i=%d (%f <= %f)"),
            i, Out[i].Confidence, Out[i-1].Confidence),
            Out[i].Confidence <= Out[i-1].Confidence + 1e-5f);
        TestTrue(FString::Printf(TEXT("Confidence in [0,1] at i=%d: %f"), i, Out[i].Confidence),
            Out[i].Confidence >= 0.f && Out[i].Confidence <= 1.f + 1e-5f);
    }

    // RetrievalScore = raw L2 distance, should also ascend (lower=better for Vector).
    for (int32 i = 1; i < Out.Num(); ++i)
    {
        TestTrue(FString::Printf(TEXT("RetrievalScore (L2 dist) ascending at i=%d (%f >= %f)"),
            i, Out[i].RetrievalScore, Out[i-1].RetrievalScore),
            Out[i].RetrievalScore >= Out[i-1].RetrievalScore - 1e-5f);
    }

    // -- MinConfidence pre-filter: aggressive cutoff should keep only top-1. --
    Params.MinConfidence = 0.999f;
    Store->Retrieve(Vec(D, {0.0f}), TEXT(""), Params, Out);
    TestEqual(TEXT("MinConfidence=0.999 → only top-1 retained"), Out.Num(), 1);
    TestEqual(TEXT("Surviving chunk is top-1"), Out[0].Source, FString(TEXT("zero.md")));

    // -- MinConfidence intermediate: keeps top-1 plus near matches. --
    Params.MinConfidence = 0.5f;
    Store->Retrieve(Vec(D, {0.0f}), TEXT(""), Params, Out);
    TestTrue(FString::Printf(TEXT("MinConfidence=0.5 keeps at least top-1 (%d)"), Out.Num()),
        Out.Num() >= 1);
    TestTrue(FString::Printf(TEXT("MinConfidence=0.5 trims at least one (%d < 4)"), Out.Num()),
        Out.Num() < 4);
    return true;
}

/**
 * BM25-mode score normalization. Same contract — top-1 = 1.0, descending, tagged.
 */
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FRagScoreNormalizationBM25ModeTest,
    "LlamaTools.RAG.ScoreNormalizationBM25",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FRagScoreNormalizationBM25ModeTest::RunTest(const FString& /*Parameters*/)
{
    constexpr int32 D = 4;

    URagStore* Store = NewObject<URagStore>();
    Store->VectorParams.Dimensions  = D;
    Store->VectorParams.MaxElements = 16;
    Store->Initialize();

    TArray<FLlamaChunk> NewChunks = {
        MakeChunk(TEXT("alpha beta gamma delta epsilon"),       TEXT("a.md")),
        MakeChunk(TEXT("alpha beta gamma — partial overlap"),   TEXT("b.md")),
        MakeChunk(TEXT("zeta eta theta — no overlap with query"), TEXT("c.md")),
    };
    TArray<TArray<float>> Embeddings = { Vec(D, {1.f}), Vec(D, {0.f, 1.f}), Vec(D, {0.f, 0.f, 1.f}) };
    Store->IngestChunksWithEmbeddings(NewChunks, Embeddings);

    FRagRetrievalParams Params;
    Params.Mode = ERagRetrievalMode::BM25;
    Params.TopK = 3;

    TArray<FLlamaChunk> Out;
    Store->Retrieve(/*Query*/ TArray<float>(), /*QueryText*/ TEXT("alpha beta gamma delta epsilon"), Params, Out);

    TestTrue(TEXT("BM25 returns at least one result"), Out.Num() >= 1);
    if (Out.Num() == 0) return false;

    TestEqual(TEXT("Top-1 source = a.md (full match)"), Out[0].Source, FString(TEXT("a.md")));
    TestEqual(TEXT("Top-1 SourceRetriever = BM25"),
        (uint8)Out[0].SourceRetriever, (uint8)ERagRetrievalSource::BM25);
    TestTrue(TEXT("Top-1 Confidence == 1.0"), FMath::IsNearlyEqual(Out[0].Confidence, 1.f, 1e-4f));
    TestTrue(TEXT("Top-1 RetrievalScore > 0"), Out[0].RetrievalScore > 0.f);

    for (int32 i = 1; i < Out.Num(); ++i)
    {
        TestTrue(FString::Printf(TEXT("BM25 confidence descending at i=%d"), i),
            Out[i].Confidence <= Out[i-1].Confidence + 1e-5f);
    }
    return true;
}

#endif // WITH_DEV_AUTOMATION_TESTS
