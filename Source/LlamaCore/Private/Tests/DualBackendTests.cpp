// Copyright 2025-current Getnamo.

#include "Misc/AutomationTest.h"

#if WITH_DEV_AUTOMATION_TESTS

#include "LlamaDualBackend.h"
#include "LlamaNative.h"

/**
 * Coverage for the dual-backend's pure state-machine behavior — toggle, history sync,
 * audio routing, and capability predicates. None of these require a loaded model or a
 * reachable HTTP server, so they run in CI on any machine.
 */

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FDualBackendDefaultLocalTest,
    "LlamaCore.DualBackend.DefaultLocal",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FDualBackendDefaultLocalTest::RunTest(const FString& /*Parameters*/)
{
    FLlamaDualBackend B;
    B.Initialize();

    TestFalse(TEXT("Defaults to local routing"), B.IsUsingRemote());
    TestNotNull(TEXT("LlamaNative allocated"), B.GetLlamaNative());
    TestFalse(TEXT("Not multimodal until model loaded"), B.IsMultimodalLoaded());
    TestFalse(TEXT("No vision until model loaded"), B.SupportsVision());
    TestFalse(TEXT("No audio until model loaded"), B.SupportsAudio());
    TestEqual(TEXT("Embedding dim 0 without model"), B.GetEmbeddingDimension(), 0);
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FDualBackendToggleHistorySyncTest,
    "LlamaCore.DualBackend.ToggleHistorySync",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FDualBackendToggleHistorySyncTest::RunTest(const FString& /*Parameters*/)
{
    FLlamaDualBackend B;
    B.Initialize();
    B.Endpoint.BaseUrl = TEXT(""); // empty URL → SetUseRemote(true) won't auto-load, just flips state
    B.ModelParams.PathToModel = TEXT(""); // also empty so reverse path won't load either

    // Seed history so toggle should mark history sync pending.
    FStructuredChatMessage M;
    M.Role = EChatTemplateRole::User;
    M.Content = TEXT("Hello");
    B.ModelState.ChatHistory.History.Add(M);

    AddExpectedError(TEXT("Endpoint.BaseUrl empty"), EAutomationExpectedErrorFlags::Contains, 1);
    B.SetUseRemote(true);
    TestTrue(TEXT("Now using remote"), B.IsUsingRemote());

    AddExpectedError(TEXT("ModelParams.PathToModel empty"), EAutomationExpectedErrorFlags::Contains, 1);
    B.SetUseRemote(false);
    TestFalse(TEXT("Back to local"), B.IsUsingRemote());

    // Idempotent: re-setting to same value is a no-op (no expected log).
    B.SetUseRemote(false);
    TestFalse(TEXT("Still local after no-op toggle"), B.IsUsingRemote());

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FDualBackendRemoteCapsTest,
    "LlamaCore.DualBackend.RemoteCapsBeforeLoad",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FDualBackendRemoteCapsTest::RunTest(const FString& /*Parameters*/)
{
    FLlamaDualBackend B;
    B.Initialize();
    B.Endpoint.BaseUrl = TEXT(""); // suppress auto-load (default BaseUrl is localhost:8080)

    AddExpectedError(TEXT("Endpoint.BaseUrl empty"), EAutomationExpectedErrorFlags::Contains, 1);
    B.SetUseRemote(true);

    // No /props fetch yet, so all remote capability flags should report false.
    TestFalse(TEXT("Remote: no multimodal until /props"), B.IsMultimodalLoaded());
    TestFalse(TEXT("Remote: no vision until /props"), B.SupportsVision());
    TestFalse(TEXT("Remote: no audio until /props"), B.SupportsAudio());
    TestEqual(TEXT("Remote audio default 16k"), B.GetAudioSampleRate(), 16000);
    TestFalse(TEXT("Model not loaded"), B.IsModelLoaded());
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FDualBackendRollbackBypassTest,
    "LlamaCore.DualBackend.RollbackBypass",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FDualBackendRollbackBypassTest::RunTest(const FString& /*Parameters*/)
{
    // Verifies state-only rollback path runs when ShouldBypassNativeKV is true
    // (impersonation mode here — covers the remote case too without needing a server).
    FLlamaDualBackend B;
    B.Initialize();
    B.ModelParams.bImpersonationMode = true;

    auto Push = [&B](EChatTemplateRole Role, const TCHAR* Text) {
        FStructuredChatMessage M; M.Role = Role; M.Content = Text;
        B.ModelState.ChatHistory.History.Add(MoveTemp(M));
    };
    Push(EChatTemplateRole::User,      TEXT("u1"));
    Push(EChatTemplateRole::Assistant, TEXT("a1"));
    Push(EChatTemplateRole::User,      TEXT("u2"));
    Push(EChatTemplateRole::Assistant, TEXT("a2"));

    TestEqual(TEXT("4 messages before rollback"), B.ModelState.ChatHistory.History.Num(), 4);
    B.RemoveLastReply();
    TestEqual(TEXT("3 after RemoveLastReply"), B.ModelState.ChatHistory.History.Num(), 3);

    B.RemoveLastUserInput();
    TestEqual(TEXT("1 after RemoveLastUserInput"), B.ModelState.ChatHistory.History.Num(), 1);
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FDualBackendResetContextTest,
    "LlamaCore.DualBackend.ResetContext",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FDualBackendResetContextTest::RunTest(const FString& /*Parameters*/)
{
    // Force the remote-mode reset path (which is pure state mutation, no HTTP if no slot).
    FLlamaDualBackend B;
    B.Initialize();
    B.Endpoint.BaseUrl = TEXT(""); // suppress auto-load
    AddExpectedError(TEXT("Endpoint.BaseUrl empty"), EAutomationExpectedErrorFlags::Contains, 1);
    B.SetUseRemote(true);

    auto Push = [&B](EChatTemplateRole Role, const TCHAR* Text) {
        FStructuredChatMessage M; M.Role = Role; M.Content = Text;
        B.ModelState.ChatHistory.History.Add(MoveTemp(M));
    };
    Push(EChatTemplateRole::System,    TEXT("sys"));
    Push(EChatTemplateRole::User,      TEXT("u"));
    Push(EChatTemplateRole::Assistant, TEXT("a"));

    int32 ResetCalls = 0;
    B.OnContextReset = [&ResetCalls]() { ++ResetCalls; };

    B.ResetContextHistory(/*bKeepSystemPrompt=*/ true);
    TestEqual(TEXT("OnContextReset fired once"), ResetCalls, 1);
    TestEqual(TEXT("Only system message remains"), B.ModelState.ChatHistory.History.Num(), 1);
    TestEqual(TEXT("System role preserved"), (int32)B.ModelState.ChatHistory.History[0].Role,
              (int32)EChatTemplateRole::System);

    Push(EChatTemplateRole::User, TEXT("u2"));
    B.ResetContextHistory(/*bKeepSystemPrompt=*/ false);
    TestEqual(TEXT("Full clear empties history"), B.ModelState.ChatHistory.History.Num(), 0);
    return true;
}

#endif // WITH_DEV_AUTOMATION_TESTS
