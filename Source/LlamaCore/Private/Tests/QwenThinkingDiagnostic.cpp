// Copyright 2025-current Getnamo.

#include "Misc/AutomationTest.h"

#if WITH_DEV_AUTOMATION_TESTS

#include "Embedding/RagStore.h"
#include "Tests/RagAskTestSink.h"
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
            TEXT("nomic-embed-text-v1.5.Q4_K_M.gguf"),
            TEXT("bge-small-en-v1.5-q4_k_m.gguf"),
            TEXT("multilingual-e5-large-instruct-q8_0.gguf"),
        };
        for (const FString& F : Candidates)
        {
            const FString Full = Root / F;
            if (FPaths::FileExists(Full)) return FPaths::ConvertRelativePathToFull(Full);
        }
        return FString();
    }

    static FString FindCorpusDir()
    {
        const FString C = FPaths::ConvertRelativePathToFull(
            FPaths::ProjectDir() / TEXT("Notes") / TEXT("RagDocs"));
        return FPaths::DirectoryExists(C) ? C : FString();
    }

    static bool WaitFor(double TimeoutSec, TFunctionRef<bool()> Predicate)
    {
        const double Deadline = FPlatformTime::Seconds() + TimeoutSec;
        while (FPlatformTime::Seconds() < Deadline)
        {
            FTSTicker::GetCoreTicker().Tick(0.016f);
            if (Predicate()) return true;
            FPlatformProcess::Sleep(0.016f);
        }
        return Predicate();
    }

    /** Pulls the substring strictly between the FIRST <think>…</think> pair in `Text`.
     *  Returns the inner content (whitespace included). If no closing tag, returns
     *  everything after the opening tag. If no opening tag, returns empty. */
    static FString ExtractThinkBlock(const FString& Text)
    {
        const FString Open  = TEXT("<think>");
        const FString Close = TEXT("</think>");
        const int32 OpenIdx = Text.Find(Open, ESearchCase::IgnoreCase, ESearchDir::FromStart);
        if (OpenIdx == INDEX_NONE) return FString();

        const int32 ContentStart = OpenIdx + Open.Len();
        const int32 CloseIdx = Text.Find(Close, ESearchCase::IgnoreCase, ESearchDir::FromStart, ContentStart);
        if (CloseIdx == INDEX_NONE)
        {
            return Text.Mid(ContentStart);
        }
        return Text.Mid(ContentStart, CloseIdx - ContentStart);
    }
}

/** Multi-query diagnostic: loops over a battery of questions through Qwen3.5-9B with
 *  bEnableThinking=false and inspects the actual content of every emitted think block.
 *  Whitespace-only inner content = the directive worked. Non-whitespace content = the
 *  model thought anyway (i.e. the "disable" directive is being ignored). */
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FQwenThinkingDiagnosticTest,
    "LlamaCore.RAG.QwenThinkingDiagnostic",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FQwenThinkingDiagnosticTest::RunTest(const FString& /*Parameters*/)
{
    const FString Embed  = FindEmbeddingModel();
    const FString Corpus = FindCorpusDir();
    const FString QwenPath = FPaths::ConvertRelativePathToFull(
        FPaths::ProjectSavedDir() / TEXT("Models") / TEXT("Qwen3.5-9B-Q4_K_M.gguf"));

    if (Embed.IsEmpty() || Corpus.IsEmpty() || !FPaths::FileExists(QwenPath))
    {
        AddInfo(TEXT("Skipping: Qwen3.5 / embedder / corpus missing"));
        return true;
    }

    URagStore* Store = NewObject<URagStore>();
    URagAskTestSink* Sink = NewObject<URagAskTestSink>();

    Store->EmbeddingModelParams.PathToModel    = Embed;
    Store->AnswerModelParams.PathToModel       = QwenPath;
    Store->AnswerModelParams.MaxContextLength  = 4096;
    Store->AnswerModelParams.GPULayers         = 99;
    Store->AnswerModelParams.Seed              = -1; // random; loop over multiple queries.

    AddInfo(FString::Printf(TEXT("Defaults: bEnableThinking=%s bStripThinkingFromResponse=%s Temp=%.2f"),
        Store->AnswerModelParams.Advanced.Thinking.bEnableThinking ? TEXT("true") : TEXT("false"),
        Store->AnswerModelParams.Advanced.Thinking.bStripThinkingFromResponse ? TEXT("true") : TEXT("false"),
        Store->AnswerModelParams.Advanced.Sampling.Temp));

    Store->bBroadcastChunksOnAsk = false;

    Store->OnAskRetrievedChunks.AddDynamic(Sink, &URagAskTestSink::HandleRetrieved);
    Store->OnAskResponseGenerated.AddDynamic(Sink, &URagAskTestSink::HandleResponse);
    Store->OnAskEndOfStream.AddDynamic(Sink, &URagAskTestSink::HandleEnd);
    Store->OnAskError.AddDynamic(Sink, &URagAskTestSink::HandleError);
    Store->OnIngestComplete.AddDynamic(Sink, &URagAskTestSink::HandleIngest);
    Store->OnAskTokenGenerated.AddDynamic(Sink, &URagAskTestSink::HandleTokenStream);
    Store->OnAskPartialGenerated.AddDynamic(Sink, &URagAskTestSink::HandlePartialStream);

    Store->LoadModels();
    if (!WaitFor(300.0, [&]() { return Store->IsEmbedderReady() && Store->IsAnswerEngineReady(); }))
    {
        AddError(TEXT("Model load timeout"));
        return false;
    }
    Store->Initialize();

    Sink->IngestAdded = -1;
    const int32 N = Store->IngestDirectory(Corpus, TEXT("md"), true);
    if (!WaitFor(180.0, [&]() { return Sink->IngestAdded >= 0; })) { AddError(TEXT("Ingest timeout")); return false; }
    AddInfo(FString::Printf(TEXT("Ingested %d chunks from %d files"), Sink->IngestAdded, N));

    // 5 distinct queries chosen to exercise varied retrieval + reasoning:
    // some have direct answers in chunks, some require synthesis, some are open-ended
    // — exactly the kind of variety that might tempt a thinking model to reason.
    const TArray<FString> Questions = {
        TEXT("How can I tell my dough has finished bulk fermentation?"),
        TEXT("What's the difference between a Langstroth and a Warré hive?"),
        TEXT("Compare Lumen's surface cache to traditional lightmaps and explain when you'd use each."),
        TEXT("Why was Gutenberg's antimony alloy a key innovation, and what would have happened with pure lead?"),
        TEXT("Step through what would go wrong if I beekeep without managing varroa mites for two years."),
    };

    int32 NonEmptyThinkCount = 0;
    int32 EmptyThinkCount    = 0;
    int32 NoThinkTagCount    = 0;
    int32 LongestThinkChars  = 0;

    for (int32 q = 0; q < Questions.Num(); ++q)
    {
        const FString& Q = Questions[q];

        // Reset sink state between queries.
        Sink->bAskRetrieved = false;
        Sink->bAskResponse  = false;
        Sink->bAskEnd       = false;
        Sink->bAskError     = false;
        Sink->RetrievedChunks.Empty();
        Sink->FinalAnswer.Empty();
        Sink->LastError.Empty();
        Sink->StreamedTokens.Empty();
        Sink->StreamedPartials.Empty();
        Sink->StreamedTokenCount = 0;

        AddInfo(FString::Printf(TEXT("\n========== Query %d/%d =========="), q + 1, Questions.Num()));
        AddInfo(FString::Printf(TEXT("Q: %s"), *Q));

        Store->AskDefault(Q);
        if (!WaitFor(300.0, [&]() { return Sink->bAskEnd || Sink->bAskError; }))
        {
            AddWarning(FString::Printf(TEXT("Query %d timed out — skipping"), q + 1));
            continue;
        }
        if (Sink->bAskError)
        {
            AddWarning(FString::Printf(TEXT("Query %d errored: %s"), q + 1, *Sink->LastError));
            continue;
        }

        const FString& Stream = Sink->StreamedTokens;
        const FString  ThinkContent = ExtractThinkBlock(Stream);
        FString ThinkTrimmed = ThinkContent;
        ThinkTrimmed.TrimStartAndEndInline();

        const bool bHasOpen = Stream.Contains(TEXT("<think>"));
        const bool bHasContent = !ThinkTrimmed.IsEmpty();

        if (!bHasOpen)
        {
            ++NoThinkTagCount;
            AddInfo(TEXT("  THINK: <no <think> tag emitted>"));
        }
        else if (!bHasContent)
        {
            ++EmptyThinkCount;
            AddInfo(FString::Printf(TEXT("  THINK: empty (raw inner: %d chars, %d after trim)"),
                ThinkContent.Len(), ThinkTrimmed.Len()));
        }
        else
        {
            ++NonEmptyThinkCount;
            LongestThinkChars = FMath::Max(LongestThinkChars, ThinkTrimmed.Len());
            // Print the full thinking content so we can see what the model is reasoning about.
            AddWarning(FString::Printf(TEXT("  THINK: NON-EMPTY (%d chars after trim)"), ThinkTrimmed.Len()));
            AddInfo(FString::Printf(TEXT("  Think content (truncated to 800 chars):\n%s"),
                *ThinkTrimmed.Left(800)));
        }

        AddInfo(FString::Printf(TEXT("  Token count: %d  Partial count: %d  Final-answer chars: %d"),
            Sink->StreamedTokenCount, Sink->StreamedPartials.Num(), Sink->FinalAnswer.Len()));
        AddInfo(FString::Printf(TEXT("  Final answer (truncated 240): %s"),
            *Sink->FinalAnswer.Left(240)));
    }

    // ── Summary ─────────────────────────────────────────────────────────────
    AddInfo(FString::Printf(TEXT("\n========== SUMMARY (Qwen3.5-9B, bEnableThinking=false) ==========")));
    AddInfo(FString::Printf(TEXT("  No <think> tag:    %d / %d"), NoThinkTagCount,    Questions.Num()));
    AddInfo(FString::Printf(TEXT("  Empty think block: %d / %d"), EmptyThinkCount,    Questions.Num()));
    AddInfo(FString::Printf(TEXT("  NON-EMPTY think:   %d / %d   (longest %d chars)"),
        NonEmptyThinkCount, Questions.Num(), LongestThinkChars));

    if (NonEmptyThinkCount > 0)
    {
        AddWarning(FString::Printf(TEXT(
            "Confirmed: bEnableThinking=false is being IGNORED on Qwen3.5 for %d/%d queries. "
            "Model is producing real chain-of-thought between the <think> tags despite the directive."),
            NonEmptyThinkCount, Questions.Num()));
    }
    else
    {
        AddInfo(TEXT("All queries produced empty/no think blocks — directive working in this run."));
    }
    return true;
}

#endif // WITH_DEV_AUTOMATION_TESTS
