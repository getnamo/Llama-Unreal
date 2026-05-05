// Copyright 2025-current Getnamo.

#include "Misc/AutomationTest.h"

#if WITH_DEV_AUTOMATION_TESTS

#include "Embedding/BM25Index.h"
#include "Embedding/HybridRetriever.h"
#include "Embedding/VectorDatabase.h"
#include "Serialization/MemoryReader.h"
#include "Serialization/MemoryWriter.h"

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBM25TokenizeTest,
    "LlamaCore.BM25.Tokenize",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FBM25TokenizeTest::RunTest(const FString& /*Parameters*/)
{
    TArray<FString> T;
    FBM25Index::Tokenize(TEXT("Hello, World! This is a TEST_VALUE."), T, /*filter*/ true);
    TestTrue(TEXT("Has hello"), T.Contains(FString(TEXT("hello"))));
    TestTrue(TEXT("Has world"), T.Contains(FString(TEXT("world"))));
    TestTrue(TEXT("Has test_value"), T.Contains(FString(TEXT("test_value"))));
    TestFalse(TEXT("Stopword 'the' filtered when present"),
        T.Contains(FString(TEXT("the"))));
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBM25BasicQueryTest,
    "LlamaCore.BM25.BasicQuery",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FBM25BasicQueryTest::RunTest(const FString& /*Parameters*/)
{
    FBM25Index Idx;
    Idx.AddDocument(1, TEXT("How do I rotate a vector around an axis?"));
    Idx.AddDocument(2, TEXT("Pizza dough rises better with warm water."));
    Idx.AddDocument(3, TEXT("Vectors and quaternions are common in 3D math."));
    Idx.AddDocument(4, TEXT("The quick brown fox jumps over the lazy dog."));
    Idx.Finalize();

    TArray<int64> Ids;
    TArray<float> Scores;
    Idx.Query(TEXT("rotate vector"), 4, Ids, Scores);

    TestTrue(TEXT("Got results"), Ids.Num() > 0);
    TestEqual(TEXT("Best match is the vector-rotation doc"), Ids[0], int64(1));
    if (Ids.Num() >= 2)
    {
        TestEqual(TEXT("Second is the 3D math doc"), Ids[1], int64(3));
    }
    // Pizza must not be in the top results
    for (int64 Id : Ids) { TestTrue(TEXT("Pizza excluded"), Id != int64(2)); }
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBM25SaveLoadTest,
    "LlamaCore.BM25.SaveLoad",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FBM25SaveLoadTest::RunTest(const FString& /*Parameters*/)
{
    FBM25Index Original;
    Original.AddDocument(1, TEXT("alpha beta gamma"));
    Original.AddDocument(2, TEXT("beta delta epsilon"));
    Original.AddDocument(3, TEXT("gamma epsilon zeta"));
    Original.Finalize();

    TArray<uint8> Buffer;
    {
        FMemoryWriter Writer(Buffer);
        Original.Save(Writer);
    }

    FBM25Index Restored;
    {
        FMemoryReader Reader(Buffer);
        TestTrue(TEXT("Load succeeded"), Restored.Load(Reader));
    }

    TArray<int64> A, B; TArray<float> SA, SB;
    Original.Query(TEXT("beta gamma"), 3, A, SA);
    Restored.Query(TEXT("beta gamma"), 3, B, SB);

    TestEqual(TEXT("Result count matches"), A.Num(), B.Num());
    for (int32 i = 0; i < A.Num(); ++i)
    {
        TestEqual(FString::Printf(TEXT("ID[%d] matches"), i), A[i], B[i]);
        TestTrue(FString::Printf(TEXT("Score[%d] within 1e-4"), i),
            FMath::IsNearlyEqual(SA[i], SB[i], 1e-4f));
    }
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FHybridRRFFusionTest,
    "LlamaCore.RAG.HybridRRF",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FHybridRRFFusionTest::RunTest(const FString& /*Parameters*/)
{
    // Synthetic 8-d "embeddings" — three topical clusters, each with a discriminating dim.
    constexpr int32 D = 8;

    FVectorDatabase V;
    V.Params.Dimensions = D;
    V.Params.MaxElements = 16;
    V.InitializeDB();

    FBM25Index B;

    auto AddPair = [&](int64 Id, const TArray<float>& Vec, const FString& Text) {
        V.AddVectorEmbeddingIdPair(Vec, Id);
        B.AddDocument(Id, Text);
    };

    // Three docs in the "rotation/vector" cluster, three in "cooking", three in "networking".
    auto Vec = [](float A, float B_, float C){ TArray<float> R; R.Add(A); R.Add(B_); R.Add(C); R.Add(0); R.Add(0); R.Add(0); R.Add(0); R.Add(0); return R; };

    AddPair(1, Vec(1.0f, 0.f, 0.f), TEXT("Rotate the vector around the up axis."));
    AddPair(2, Vec(0.9f, 0.1f, 0.f), TEXT("A quaternion stores a 3D rotation."));
    AddPair(3, Vec(0.8f, 0.2f, 0.f), TEXT("Cross product produces a perpendicular vector."));

    AddPair(4, Vec(0.f, 1.0f, 0.f), TEXT("Pizza dough rises in a warm bowl."));
    AddPair(5, Vec(0.f, 0.9f, 0.1f), TEXT("Bread requires gluten development."));
    AddPair(6, Vec(0.f, 0.8f, 0.2f), TEXT("Yeast ferments sugar producing CO2."));

    AddPair(7, Vec(0.f, 0.f, 1.0f), TEXT("TCP guarantees ordered delivery of packets."));
    AddPair(8, Vec(0.1f, 0.f, 0.9f), TEXT("UDP is connectionless and unordered."));
    AddPair(9, Vec(0.2f, 0.f, 0.8f), TEXT("HTTP is built on top of TCP."));

    B.Finalize();

    FHybridRetriever R;
    R.Vector = &V;
    R.Bm25   = &B;

    // Query: should retrieve cluster 1 (rotate/vector). The query text uses the literal
    // "rotate" keyword (BM25 strong) plus an embedding nearest cluster 1.
    TArray<int64> Ids;
    TArray<float> Scores;
    R.Query(Vec(1.0f, 0.f, 0.f), TEXT("rotate vector"), 3, 5, 60, Ids, Scores);

    TestEqual(TEXT("Three results"), Ids.Num(), 3);
    for (int64 Id : Ids)
    {
        TestTrue(FString::Printf(TEXT("Result %lld is in cluster 1"), Id),
            Id >= 1 && Id <= 3);
    }
    // Scores monotonically non-increasing.
    for (int32 i = 1; i < Scores.Num(); ++i)
    {
        TestTrue(TEXT("Scores descending"), Scores[i] <= Scores[i-1] + 1e-5f);
    }
    return true;
}

#endif // WITH_DEV_AUTOMATION_TESTS
