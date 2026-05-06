// Copyright 2025-current Getnamo.

#include "Misc/AutomationTest.h"

#if WITH_DEV_AUTOMATION_TESTS

#include "Embedding/RagStore.h"
#include "RagAskTestSink.h"
#include "Misc/Paths.h"
#include "HAL/PlatformProcess.h"
#include "HAL/PlatformTime.h"
#include "Containers/Ticker.h"

namespace
{
    static FString FindEmbeddingModel()
    {
        const FString Root = FPaths::ProjectSavedDir() / TEXT("Models");
        const TArray<FString> Candidates = {
            TEXT("bge-small-en-v1.5-q4_k_m.gguf"),
            TEXT("nomic-embed-text-v1.5.Q4_K_M.gguf"),
            TEXT("multilingual-e5-large-instruct-q8_0.gguf"),
        };
        for (const FString& F : Candidates)
        {
            const FString Full = Root / F;
            if (FPaths::FileExists(Full)) { return FPaths::ConvertRelativePathToFull(Full); }
        }
        return FString();
    }

    static FString FindChatModel()
    {
        const FString Root = FPaths::ProjectSavedDir() / TEXT("Models");
        const TArray<FString> Candidates = {
            TEXT("google_gemma-3-4b-it-Q4_K_L.gguf"),
            TEXT("gemma-4-E2B-it-Q6_K.gguf"),
            TEXT("Qwen2.5-Omni-7B-Q4_K_M.gguf"),
            TEXT("Qwen3.5-9B-Q4_K_M.gguf"),
        };
        for (const FString& F : Candidates)
        {
            const FString Full = Root / F;
            if (FPaths::FileExists(Full)) { return FPaths::ConvertRelativePathToFull(Full); }
        }
        return FString();
    }

    static FString FindCorpusDir()
    {
        const FString Candidate = FPaths::ConvertRelativePathToFull(
            FPaths::ProjectDir() / TEXT("Notes") / TEXT("RagDocs"));
        return FPaths::DirectoryExists(Candidate) ? Candidate : FString();
    }

    static bool WaitFor(double TimeoutSec, TFunctionRef<bool()> Predicate)
    {
        const double Deadline = FPlatformTime::Seconds() + TimeoutSec;
        while (FPlatformTime::Seconds() < Deadline)
        {
            FTSTicker::GetCoreTicker().Tick(0.016f);
            if (Predicate()) { return true; }
            FPlatformProcess::Sleep(0.016f);
        }
        return Predicate();
    }
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FRagAskPipelineTest,
    "LlamaCore.RAG.AskPipeline",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FRagAskPipelineTest::RunTest(const FString& /*Parameters*/)
{
    const FString EmbedPath = FindEmbeddingModel();
    const FString ChatPath  = FindChatModel();
    const FString Corpus    = FindCorpusDir();

    if (EmbedPath.IsEmpty() || ChatPath.IsEmpty() || Corpus.IsEmpty())
    {
        AddInfo(FString::Printf(TEXT(
            "Skipping integration test — required artifacts missing.\n"
            "  EmbeddingModel: %s\n"
            "  ChatModel:      %s\n"
            "  Corpus:         %s"),
            EmbedPath.IsEmpty() ? TEXT("NOT FOUND") : *EmbedPath,
            ChatPath.IsEmpty()  ? TEXT("NOT FOUND") : *ChatPath,
            Corpus.IsEmpty()    ? TEXT("NOT FOUND") : *Corpus));
        return true;
    }
    AddInfo(FString::Printf(TEXT("Embedder: %s"), *EmbedPath));
    AddInfo(FString::Printf(TEXT("Chat:     %s"), *ChatPath));
    AddInfo(FString::Printf(TEXT("Corpus:   %s"), *Corpus));

    URagStore* Store = NewObject<URagStore>();
    URagAskTestSink* Sink = NewObject<URagAskTestSink>();

    Store->EmbeddingModelParams.PathToModel       = EmbedPath;
    Store->EmbeddingModelParams.MaxContextLength  = 2048;
    Store->EmbeddingModelParams.GPULayers         = 99;
    Store->EmbeddingModelParams.bAutoInsertSystemPromptOnLoad = false;

    Store->AnswerModelParams.PathToModel       = ChatPath;
    Store->AnswerModelParams.MaxContextLength  = 4096;
    Store->AnswerModelParams.GPULayers         = 99;
    Store->AnswerModelParams.bAutoInsertSystemPromptOnLoad = false;
    Store->AnswerModelParams.SystemPrompt      = TEXT("");
    Store->AnswerModelParams.Seed              = 47;

    // Test assertions inspect the retrieved chunks — opt in to the broadcast.
    Store->bBroadcastChunksOnAsk = true;

    Store->OnAskRetrievedChunks.AddDynamic(Sink, &URagAskTestSink::HandleRetrieved);
    Store->OnAskResponseGenerated.AddDynamic(Sink, &URagAskTestSink::HandleResponse);
    Store->OnAskEndOfStream.AddDynamic(Sink, &URagAskTestSink::HandleEnd);
    Store->OnAskError.AddDynamic(Sink, &URagAskTestSink::HandleError);
    Store->OnIngestComplete.AddDynamic(Sink, &URagAskTestSink::HandleIngest);

    Store->LoadModels();
    if (!WaitFor(180.0, [&]() { return Store->IsEmbedderReady() && Store->IsAnswerEngineReady(); }))
    {
        AddError(TEXT("Model load timed out (180s)"));
        return false;
    }
    AddInfo(FString::Printf(TEXT("Models loaded. VectorParams.Dimensions = %d"),
        Store->VectorParams.Dimensions));

    Store->Initialize();
    TestTrue(TEXT("Store initialized"), Store->IsInitialized());
    TestTrue(TEXT("VectorParams.Dimensions auto-pulled from embedder"), Store->VectorParams.Dimensions > 0);

    const int32 FilesQueued = Store->IngestDirectory(Corpus, TEXT("md"), /*recursive*/ true);
    TestTrue(FString::Printf(TEXT("Files queued (%d)"), FilesQueued), FilesQueued >= 5);

    if (!WaitFor(180.0, [&]() { return Sink->IngestAdded >= 0; }))
    {
        AddError(TEXT("Ingest timed out (180s)"));
        return false;
    }
    AddInfo(FString::Printf(TEXT("Ingested %d chunks across %d files"), Sink->IngestAdded, FilesQueued));
    TestTrue(TEXT("At least one chunk indexed"), Sink->IngestAdded > 0);

    const FString Question = TEXT("How can I tell my dough has finished bulk fermentation?");
    Store->AskDefault(Question);

    if (!WaitFor(180.0, [&]() { return Sink->bAskEnd || Sink->bAskError; }))
    {
        AddError(TEXT("Ask timed out (180s)"));
        return false;
    }
    if (Sink->bAskError)
    {
        AddError(FString::Printf(TEXT("Ask reported an error: %s"), *Sink->LastError));
        return false;
    }

    TestTrue(TEXT("OnAskRetrievedChunks fired"), Sink->bAskRetrieved);
    TestTrue(TEXT("OnAskResponseGenerated fired"), Sink->bAskResponse);
    TestTrue(TEXT("OnAskEndOfStream fired"), Sink->bAskEnd);
    TestTrue(TEXT("Retrieved at least one chunk"), Sink->RetrievedChunks.Num() > 0);

    if (Sink->RetrievedChunks.Num() > 0)
    {
        const FLlamaChunk& Top = Sink->RetrievedChunks[0];
        AddInfo(FString::Printf(
            TEXT("Top-1: source=%s | confidence=%.3f | retriever=%d | text starts: %s"),
            *Top.Source, Top.Confidence, (int32)Top.SourceRetriever,
            *Top.Text.Left(80)));
        TestTrue(TEXT("Top-1 source contains 'sourdough' (expected for this question)"),
            Top.Source.Contains(TEXT("sourdough"), ESearchCase::IgnoreCase));
        TestTrue(TEXT("Top-1 Confidence == 1.0"),
            FMath::IsNearlyEqual(Top.Confidence, 1.f, 1e-4f));
    }

    AddInfo(FString::Printf(TEXT("Answer (truncated): %s"), *Sink->FinalAnswer.Left(400)));
    TestTrue(TEXT("Answer is non-empty"), !Sink->FinalAnswer.IsEmpty());

    Store->Reset();
    return true;
}

#endif // WITH_DEV_AUTOMATION_TESTS
