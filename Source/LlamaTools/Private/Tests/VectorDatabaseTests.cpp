// Copyright 2025-current Getnamo.

#include "Misc/AutomationTest.h"

#if WITH_DEV_AUTOMATION_TESTS

#include "Embedding/VectorDatabase.h"
#include "HAL/FileManager.h"
#include "Misc/Paths.h"

#include <random>

namespace
{
    static void FillRandomVectors(TArray<float>& OutData, int32 D, int32 N, uint32 Seed)
    {
        std::mt19937 Rng(Seed);
        std::uniform_real_distribution<float> Dist;
        OutData.SetNumUninitialized(D * N);
        for (int32 i = 0; i < D * N; ++i)
        {
            OutData[i] = Dist(Rng);
        }
    }

    static TArray<float> SliceVector(const TArray<float>& Data, int32 Index, int32 D)
    {
        return TArray<float>(Data.GetData() + Index * D, D);
    }
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FVectorDatabaseSelfRecallTest,
    "LlamaTools.VectorDatabase.SelfRecall",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FVectorDatabaseSelfRecallTest::RunTest(const FString& /*Parameters*/)
{
    FVectorDatabase DB;
    DB.Params.Dimensions = 64;
    DB.Params.MaxElements = 500;
    DB.InitializeDB();

    TestTrue(TEXT("DB initialized"), DB.IsInitialized());

    TArray<float> Data;
    FillRandomVectors(Data, DB.Params.Dimensions, DB.Params.MaxElements, /*seed*/ 17u);

    for (int32 i = 0; i < DB.Params.MaxElements; ++i)
    {
        DB.AddVectorEmbeddingIdPair(SliceVector(Data, i, DB.Params.Dimensions), i);
    }
    TestEqual(TEXT("Element count after adds"), DB.Num(), DB.Params.MaxElements);

    int32 Correct = 0;
    for (int32 i = 0; i < DB.Params.MaxElements; ++i)
    {
        const int64 Hit = DB.FindNearestId(SliceVector(Data, i, DB.Params.Dimensions));
        if (Hit == i) ++Correct;
    }

    const float Recall = static_cast<float>(Correct) / static_cast<float>(DB.Params.MaxElements);
    TestTrue(FString::Printf(TEXT("Recall %.3f >= 0.99"), Recall), Recall >= 0.99f);
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FVectorDatabaseTopKOrderTest,
    "LlamaTools.VectorDatabase.TopKOrdering",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FVectorDatabaseTopKOrderTest::RunTest(const FString& /*Parameters*/)
{
    // 1-D ground truth embedded into D dims (only first dim varies). The nearest neighbours
    // of a query at position X are the points whose first dim is closest to X.
    constexpr int32 D = 16;
    constexpr int32 N = 100;

    FVectorDatabase DB;
    DB.Params.Dimensions = D;
    DB.Params.MaxElements = N;
    DB.InitializeDB();

    for (int32 i = 0; i < N; ++i)
    {
        TArray<float> V;
        V.Init(0.f, D);
        V[0] = static_cast<float>(i);
        DB.AddVectorEmbeddingIdPair(V, i);
    }

    TArray<float> Q; Q.Init(0.f, D); Q[0] = 42.5f;

    TArray<int64> Ids;
    TArray<float> Distances;
    DB.FindNearestNIds(Ids, Distances, Q, 5);

    TestTrue(TEXT("Got 5 results"), Ids.Num() == 5);

    // Expected nearest order: 42, 43, 41, 44, 40 (distances 0.5, 0.5, 1.5, 1.5, 2.5)
    // Either of the equal-distance pairs may swap; assert the set is correct and distances
    // are monotonically non-decreasing.
    TestTrue(TEXT("First two are 42/43"),
        (Ids[0] == 42 && Ids[1] == 43) || (Ids[0] == 43 && Ids[1] == 42));

    for (int32 i = 1; i < Distances.Num(); ++i)
    {
        TestTrue(FString::Printf(TEXT("Distances monotonic at %d: %f <= %f"), i, Distances[i-1], Distances[i]),
            Distances[i-1] <= Distances[i] + 1e-5f);
    }
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FVectorDatabaseStringPairTest,
    "LlamaTools.VectorDatabase.StringPair",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FVectorDatabaseStringPairTest::RunTest(const FString& /*Parameters*/)
{
    FVectorDatabase DB;
    DB.Params.Dimensions = 4;
    DB.Params.MaxElements = 8;
    DB.InitializeDB();

    auto Make = [](float a, float b, float c, float d) {
        TArray<float> V; V.Add(a); V.Add(b); V.Add(c); V.Add(d); return V;
    };

    DB.AddVectorEmbeddingStringPair(Make(1, 0, 0, 0), TEXT("alpha"));
    DB.AddVectorEmbeddingStringPair(Make(0, 1, 0, 0), TEXT("beta"));
    DB.AddVectorEmbeddingStringPair(Make(0, 0, 1, 0), TEXT("gamma"));

    const FString Best = DB.FindNearestString(Make(0.9f, 0.1f, 0.f, 0.f));
    TestEqual(TEXT("alpha is nearest to (0.9,0.1,0,0)"), Best, FString(TEXT("alpha")));

    TArray<FString> TopTwo;
    DB.FindNearestNStrings(TopTwo, Make(0.f, 0.5f, 0.5f, 0.f), 2);
    TestEqual(TEXT("Two strings returned"), TopTwo.Num(), 2);
    // Both beta and gamma are equidistant; assert the set
    TestTrue(TEXT("Top-2 are beta and gamma"),
        (TopTwo.Contains(FString(TEXT("beta"))) && TopTwo.Contains(FString(TEXT("gamma")))));
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FVectorDatabaseSaveLoadTest,
    "LlamaTools.VectorDatabase.SaveLoad",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FVectorDatabaseSaveLoadTest::RunTest(const FString& /*Parameters*/)
{
    constexpr int32 D = 32;
    constexpr int32 N = 200;

    const FString TmpPath = FPaths::ProjectIntermediateDir() / TEXT("LlamaCoreTests") / TEXT("vdb_roundtrip.vdb");

    TArray<float> Data;
    FillRandomVectors(Data, D, N, 31u);

    {
        FVectorDatabase DB;
        DB.Params.Dimensions = D;
        DB.Params.MaxElements = N;
        DB.InitializeDB();
        for (int32 i = 0; i < N; ++i)
        {
            DB.AddVectorEmbeddingStringPair(SliceVector(Data, i, D), FString::Printf(TEXT("doc-%d"), i));
        }
        TestTrue(TEXT("Save"), DB.Save(TmpPath));
    }

    {
        FVectorDatabase DB;
        TestTrue(TEXT("Load"), DB.Load(TmpPath));
        TestEqual(TEXT("Element count restored"), DB.Num(), N);

        // Recall self-search after load
        int32 Correct = 0;
        for (int32 i = 0; i < N; ++i)
        {
            const int64 Hit = DB.FindNearestId(SliceVector(Data, i, D));
            if (Hit == static_cast<int64>(i + 1)) ++Correct; // string-pair ids start at 1
        }
        const float Recall = static_cast<float>(Correct) / static_cast<float>(N);
        TestTrue(FString::Printf(TEXT("Post-load recall %.3f >= 0.99"), Recall), Recall >= 0.99f);

        // Text sidecar restored
        FString OutText;
        TestTrue(TEXT("TryGetText for id 1"), DB.TryGetText(1, OutText));
        TestEqual(TEXT("TryGetText value"), OutText, FString(TEXT("doc-0")));
    }

    IFileManager::Get().Delete(*TmpPath, false, true, true);
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FVectorDatabaseDimMismatchTest,
    "LlamaTools.VectorDatabase.DimensionMismatch",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FVectorDatabaseDimMismatchTest::RunTest(const FString& /*Parameters*/)
{
    FVectorDatabase DB;
    DB.Params.Dimensions = 8;
    DB.Params.MaxElements = 4;
    DB.InitializeDB();

    // Mismatched query should not crash and should return empty results.
    AddExpectedError(TEXT("query embedding dim"), EAutomationExpectedErrorFlags::Contains, 1);
    TArray<float> Bad; Bad.Init(0.f, 4);
    TArray<int64> Out;
    DB.FindNearestNIds(Out, Bad, 3);
    TestEqual(TEXT("Mismatched query yields no results"), Out.Num(), 0);
    return true;
}

#endif // WITH_DEV_AUTOMATION_TESTS
