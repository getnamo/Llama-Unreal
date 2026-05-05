// Copyright 2025-current Getnamo.

#include "Misc/AutomationTest.h"

#if WITH_DEV_AUTOMATION_TESTS

#include "Embedding/CorpusChunker.h"

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FCorpusChunkerShortTextTest,
    "LlamaCore.Chunker.ShortText",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FCorpusChunkerShortTextTest::RunTest(const FString& /*Parameters*/)
{
    FLlamaChunkerParams P; P.TargetChars = 1000; P.MaxChars = 2000; P.MinChars = 1;
    TArray<FLlamaChunk> Chunks;
    FLlamaCorpusChunker::ChunkText(TEXT("Hello world."), TEXT("a.txt"), P, Chunks);
    TestEqual(TEXT("One chunk"), Chunks.Num(), 1);
    TestEqual(TEXT("Source preserved"), Chunks[0].Source, FString(TEXT("a.txt")));
    TestEqual(TEXT("Text preserved"), Chunks[0].Text, FString(TEXT("Hello world.")));
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FCorpusChunkerParagraphsTest,
    "LlamaCore.Chunker.Paragraphs",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FCorpusChunkerParagraphsTest::RunTest(const FString& /*Parameters*/)
{
    FLlamaChunkerParams P; P.TargetChars = 1000; P.MaxChars = 2000; P.MinChars = 1;
    const FString Source = TEXT("First paragraph here.\n\nSecond paragraph here.\n\nThird paragraph here.");
    TArray<FLlamaChunk> Chunks;
    FLlamaCorpusChunker::ChunkText(Source, TEXT("doc"), P, Chunks);
    TestEqual(TEXT("Three chunks"), Chunks.Num(), 3);
    TestTrue(TEXT("First contains 'First'"), Chunks[0].Text.Contains(TEXT("First")));
    TestTrue(TEXT("Last contains 'Third'"), Chunks[2].Text.Contains(TEXT("Third")));
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FCorpusChunkerSlidingWindowTest,
    "LlamaCore.Chunker.SlidingWindow",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FCorpusChunkerSlidingWindowTest::RunTest(const FString& /*Parameters*/)
{
    FLlamaChunkerParams P;
    P.TargetChars = 100;
    P.MaxChars    = 120;
    P.OverlapChars = 20;
    P.MinChars = 1;

    // 500-char paragraph with sentence boundaries.
    FString Body;
    for (int32 i = 0; i < 50; ++i) { Body += TEXT("Sentence number "); Body += FString::FromInt(i); Body += TEXT(". "); }

    TArray<FLlamaChunk> Chunks;
    FLlamaCorpusChunker::ChunkText(Body, TEXT("long"), P, Chunks);

    TestTrue(TEXT("Multiple chunks produced"), Chunks.Num() >= 5);
    for (const FLlamaChunk& C : Chunks)
    {
        TestTrue(FString::Printf(TEXT("Chunk under MaxChars (%d)"), C.Text.Len()),
            C.Text.Len() <= P.MaxChars + 5);  // small slack for trim
    }
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FCorpusChunkerDeterminismTest,
    "LlamaCore.Chunker.Determinism",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FCorpusChunkerDeterminismTest::RunTest(const FString& /*Parameters*/)
{
    FLlamaChunkerParams P; P.TargetChars = 80; P.MaxChars = 100; P.OverlapChars = 15; P.MinChars = 1;
    FString Body;
    for (int32 i = 0; i < 20; ++i) { Body += TEXT("Sentence "); Body += FString::FromInt(i); Body += TEXT(". "); }

    TArray<FLlamaChunk> A, B;
    FLlamaCorpusChunker::ChunkText(Body, TEXT("d"), P, A);
    FLlamaCorpusChunker::ChunkText(Body, TEXT("d"), P, B);
    TestEqual(TEXT("Same count"), A.Num(), B.Num());
    for (int32 i = 0; i < A.Num(); ++i)
    {
        TestEqual(FString::Printf(TEXT("Chunk %d text matches"), i), A[i].Text, B[i].Text);
        TestEqual(FString::Printf(TEXT("Chunk %d start matches"), i), A[i].StartChar, B[i].StartChar);
    }
    return true;
}

#endif // WITH_DEV_AUTOMATION_TESTS
