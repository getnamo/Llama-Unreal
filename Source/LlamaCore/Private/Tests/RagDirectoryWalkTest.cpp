// Copyright 2025-current Getnamo.

#include "Misc/AutomationTest.h"

#if WITH_DEV_AUTOMATION_TESTS

#include "Embedding/RagStore.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "HAL/FileManager.h"

namespace
{
    static FString MakeTempCorpusDir()
    {
        const FString Root = FPaths::ConvertRelativePathToFull(
            FPaths::ProjectIntermediateDir() / TEXT("LlamaCoreTests") / TEXT("CorpusWalk"));
        IFileManager::Get().MakeDirectory(*Root, /*Tree*/ true);
        IFileManager::Get().MakeDirectory(*(Root / TEXT("sub")), /*Tree*/ true);

        FFileHelper::SaveStringToFile(TEXT("alpha doc one"), *(Root / TEXT("a.txt")));
        FFileHelper::SaveStringToFile(TEXT("alpha doc two"), *(Root / TEXT("b.txt")));
        FFileHelper::SaveStringToFile(TEXT("# heading\nbody"), *(Root / TEXT("c.md")));
        FFileHelper::SaveStringToFile(TEXT("not ingested"),    *(Root / TEXT("d.bin")));
        FFileHelper::SaveStringToFile(TEXT("nested doc"),      *(Root / TEXT("sub") / TEXT("e.txt")));
        return Root;
    }

    static void Cleanup(const FString& Root)
    {
        IFileManager::Get().DeleteDirectory(*Root, /*RequireExists*/ false, /*Tree*/ true);
    }
}

/** Verifies the directory walk + file filter behaviour. The store has no embedder
 *  so ingestion early-outs after enumeration — we just want the file count. */
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FRagIngestDirectoryWalkTest,
    "LlamaCore.RAG.IngestDirectoryWalk",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FRagIngestDirectoryWalkTest::RunTest(const FString& /*Parameters*/)
{
    const FString Root = MakeTempCorpusDir();

    URagStore* Store = NewObject<URagStore>();
    Store->VectorParams.Dimensions = 8;
    Store->VectorParams.MaxElements = 32;
    Store->ChunkerParams.MinChars = 1; // fixture files are tiny — let them through to the embedder check
    Store->Initialize();

    // No embedder set → IngestDocuments will early-out, but IngestDirectory still enumerates.
    AddExpectedError(TEXT("no embedder ready"), EAutomationExpectedErrorFlags::Contains, 1);

    // Default extensions: txt, md, recursive.
    int32 N = Store->IngestDirectory(Root, TEXT("txt,md"), /*recursive*/ true);
    TestEqual(TEXT("Recursive walk picks up 4 files (3 .txt + 1 .md)"), N, 4);

    // Non-recursive: should miss the nested file.
    AddExpectedError(TEXT("no embedder ready"), EAutomationExpectedErrorFlags::Contains, 1);
    int32 NShallow = Store->IngestDirectory(Root, TEXT("txt,md"), /*recursive*/ false);
    TestEqual(TEXT("Shallow walk picks up 3 files"), NShallow, 3);

    // Extension filter excludes .bin.
    AddExpectedError(TEXT("no embedder ready"), EAutomationExpectedErrorFlags::Contains, 1);
    int32 NTxtOnly = Store->IngestDirectory(Root, TEXT("txt"), /*recursive*/ true);
    TestEqual(TEXT("Only .txt walks 3 files"), NTxtOnly, 3);

    Cleanup(Root);
    return true;
}

#endif // WITH_DEV_AUTOMATION_TESTS
