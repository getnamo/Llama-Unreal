// Copyright 2025-current Getnamo.

#include "Misc/AutomationTest.h"

#if WITH_DEV_AUTOMATION_TESTS

#include "LlamaNative.h"
#include "Internal/LlamaInternal.h"
#include "Misc/Paths.h"
#include "HAL/PlatformProcess.h"

// Regression coverage for getnamo/Llama-Unreal#53: reloading a model re-ingested the previous
// conversation (history duplicated per reload), and any prompt larger than n_batch aborted the
// process inside llama_decode. Needs a small local model; skipped when it isn't present.
namespace LlamaReloadTests
{
    static FString FindTestModel()
    {
        const FString Path = FPaths::ConvertRelativePathToFull(
            FPaths::ProjectSavedDir() / TEXT("Models") / TEXT("google_gemma-3-4b-it-Q4_K_L.gguf"));
        return FPaths::FileExists(Path) ? Path : FString();
    }

    static FString RepeatedText(const TCHAR* Sentence, int32 Count)
    {
        FString Out;
        for (int32 i = 0; i < Count; i++)
        {
            Out += Sentence;
            Out += TEXT(" ");
        }
        return Out;
    }

    static FLLMModelParams MakeParams(const FString& ModelPath)
    {
        FLLMModelParams Params;
        Params.PathToModel = ModelPath;
        Params.MaxContextLength = 8192;
        Params.MaxBatchLength = 256;    //small on purpose so the prompts below exceed n_batch
        Params.SystemPrompt = RepeatedText(TEXT("This is just a long system prompt."), 40);
        return Params;
    }

    /** Pumps the native game-thread queue until Predicate() or timeout. */
    template <typename PredicateType>
    static bool PumpUntil(FLlamaNative& Native, PredicateType Predicate, double TimeoutSec)
    {
        const double End = FPlatformTime::Seconds() + TimeoutSec;
        while (!Predicate())
        {
            if (FPlatformTime::Seconds() > End)
            {
                return false;
            }
            Native.OnGameThreadTick(0.016f);
            FPlatformProcess::Sleep(0.016f);
        }
        return true;
    }
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FLlamaInternalReloadAndBatchTest,
    "LlamaCore.Regression.ReloadHistoryAndBatchLimit",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FLlamaInternalReloadAndBatchTest::RunTest(const FString& /*Parameters*/)
{
    using namespace LlamaReloadTests;

    const FString ModelPath = FindTestModel();
    if (ModelPath.IsEmpty())
    {
        AddInfo(TEXT("Skipping: test model not found in Saved/Models"));
        return true;
    }

    const FLLMModelParams Params = MakeParams(ModelPath);
    const std::string SystemPrompt = TCHAR_TO_UTF8(*Params.SystemPrompt);
    const std::string UserTurn = TCHAR_TO_UTF8(*RepeatedText(TEXT("Tell me about the old lighthouse."), 40));

    FLlamaInternal Internal;
    int32 LastProcessed = 0;
    Internal.OnPromptProcessed = [&LastProcessed](int32 Tokens, EChatTemplateRole, float)
    {
        LastProcessed = Tokens;
    };

    // 1) The first turn must cost the same after every reload (was 1x, 2x, 3x ... then abort).
    //    Measured on a user turn: templates that merge the system prompt into the first user turn
    //    (e.g. gemma) process nothing for a system-only insert.
    int32 FirstProcessed = 0;
    for (int32 Reload = 0; Reload < 5; Reload++)
    {
        Internal.UnloadModel();
        if (!TestTrue(FString::Printf(TEXT("load #%d"), Reload), Internal.LoadModelFromParams(Params)))
        {
            return false;
        }
        Internal.InsertTemplatedPrompt(SystemPrompt, EChatTemplateRole::System, false, false);
        LastProcessed = 0;
        Internal.InsertTemplatedPrompt(UserTurn, EChatTemplateRole::User, false, false);

        if (Reload == 0)
        {
            FirstProcessed = LastProcessed;
            TestTrue(TEXT("first turn exceeds n_batch (exercises chunking)"), FirstProcessed > Params.MaxBatchLength);
        }
        else
        {
            TestEqual(FString::Printf(TEXT("tokens processed after reload #%d"), Reload), LastProcessed, FirstProcessed);
        }
        TestEqual(TEXT("two messages after reload + first turn"), (int32)Internal.Messages.size(), 2);
    }

    // 2) A single prompt several times n_batch decodes in chunks instead of aborting
    const std::string LongPrompt = TCHAR_TO_UTF8(*RepeatedText(TEXT("Describe the weather in a small mountain village."), 150));
    const int32 UsedBefore = Internal.UsedContext();
    Internal.InsertTemplatedPrompt(LongPrompt, EChatTemplateRole::User, false, false);
    TestTrue(TEXT("long prompt larger than 4x n_batch"), LastProcessed > 4 * Params.MaxBatchLength);
    TestEqual(TEXT("KV grew by exactly the processed tokens"), Internal.UsedContext() - UsedBefore, LastProcessed);

    // 3) Full reset clears the conversation too
    Internal.ResetContextHistory(false);
    TestEqual(TEXT("no messages after full reset"), (int32)Internal.Messages.size(), 0);
    TestTrue(TEXT("empty KV after full reset"), Internal.UsedContext() < 0);

    Internal.UnloadModel();
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FLlamaNativeLoadSemanticsTest,
    "LlamaCore.Regression.LoadModelReuseSemantics",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FLlamaNativeLoadSemanticsTest::RunTest(const FString& /*Parameters*/)
{
    using namespace LlamaReloadTests;

    const FString ModelPath = FindTestModel();
    if (ModelPath.IsEmpty())
    {
        AddInfo(TEXT("Skipping: test model not found in Saved/Models"));
        return true;
    }

    FLlamaNative Native;
    FLLMModelState LastState;
    int32 LastProcessed = 0;
    Native.OnModelStateChanged = [&LastState](const FLLMModelState& State) { LastState = State; };
    Native.OnPromptProcessed = [&LastProcessed](int32 Tokens, EChatTemplateRole, float) { LastProcessed = Tokens; };

    FLLMModelParams Params = MakeParams(ModelPath);
    Native.SetModelParams(Params);

    auto LoadAndWait = [&](bool bForceReload, double& OutSeconds) -> int32
    {
        int32 Status = -1;
        const double Start = FPlatformTime::Seconds();
        Native.LoadModel(bForceReload, [&Status](const FString&, int32 StatusCode) { Status = StatusCode; });
        PumpUntil(Native, [&Status] { return Status != -1; }, 120.0);
        OutSeconds = FPlatformTime::Seconds() - Start;
        return Status;
    };

    //Inserts one user turn (no generation) and returns the tokens it cost. On templates that merge
    //the system prompt into the first user turn (e.g. gemma) this includes the system prompt.
    auto InsertUserTurn = [&]() -> int32
    {
        FLlamaChatPrompt UserTurn;
        UserTurn.Prompt = TEXT("Hello there, how are you today?");
        UserTurn.bGenerateReply = false;
        LastProcessed = 0;
        Native.InsertTemplatedPrompt(UserTurn, nullptr);
        PumpUntil(Native, [&] { return LastProcessed > 0 && LastState.ChatHistory.History.Num() == 2; }, 60.0);
        return LastProcessed;
    };

    double Seconds = 0.0;
    TestEqual(TEXT("initial load"), LoadAndWait(true, Seconds), 0);
    TestEqual(TEXT("history after load = system prompt"), LastState.ChatHistory.History.Num(), 1);
    const int32 FirstTurnTokens = InsertUserTurn();
    TestTrue(TEXT("first turn processed"), FirstTurnTokens > 0);
    TestEqual(TEXT("history after user turn"), LastState.ChatHistory.History.Num(), 2);

    // Non-forced, identical params: model kept, conversation reset to just the system prompt
    double ReuseSeconds = 0.0;
    TestEqual(TEXT("non-forced reuse load"), LoadAndWait(false, ReuseSeconds), 0);
    TestEqual(TEXT("history after reuse = system prompt"), LastState.ChatHistory.History.Num(), 1);
    TestEqual(TEXT("first turn tokens after reuse"), InsertUserTurn(), FirstTurnTokens);
    AddInfo(FString::Printf(TEXT("load %.2fs, non-forced reuse %.2fs"), Seconds, ReuseSeconds));
    TestTrue(TEXT("reuse is much faster than a load"), ReuseSeconds < Seconds * 0.5);

    // Non-forced with a changed load parameter: must actually reload
    Params.MaxContextLength = 4096;
    Native.SetModelParams(Params);
    double ChangedSeconds = 0.0;
    TestEqual(TEXT("non-forced load with changed params"), LoadAndWait(false, ChangedSeconds), 0);
    TestEqual(TEXT("history after param change reload"), LastState.ChatHistory.History.Num(), 1);
    TestTrue(TEXT("changed params took a real reload"), ChangedSeconds > ReuseSeconds * 5.0);
    TestEqual(TEXT("first turn tokens after param change reload"), InsertUserTurn(), FirstTurnTokens);

    // Forced: always a clean hard reload
    for (int32 i = 0; i < 3; i++)
    {
        TestEqual(FString::Printf(TEXT("forced reload #%d"), i), LoadAndWait(true, Seconds), 0);
        TestEqual(FString::Printf(TEXT("history after forced reload #%d"), i), LastState.ChatHistory.History.Num(), 1);
        TestEqual(FString::Printf(TEXT("first turn tokens after forced reload #%d"), i), InsertUserTurn(), FirstTurnTokens);
    }
    return true;
}

#endif // WITH_DEV_AUTOMATION_TESTS
