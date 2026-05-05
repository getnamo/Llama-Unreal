// Copyright 2025-current Getnamo.

#include "Misc/AutomationTest.h"

#if WITH_DEV_AUTOMATION_TESTS

#include "LlamaUtility.h"
#include "LlamaDualBackend.h"
#include "LlamaNative.h"
#include "Misc/Paths.h"
#include "Misc/FileHelper.h"
#include "Containers/Ticker.h"
#include "HAL/PlatformProcess.h"

#include <string>

namespace
{
    /** Strings that exercise every UTF-8 byte length and a few multi-codepoint shapes. */
    static const TArray<FString>& Utf8Specimens()
    {
        static const TArray<FString> Specimens = {
            // 1-byte ASCII
            TEXT("Hello, World!"),
            // 2-byte: Latin-1 supplement, accented forms, German + French
            TEXT("Café — naïve façade — Größe"),
            // 2-byte: Cyrillic
            TEXT("Привет, мир!"),
            // 2-byte: Greek + Hebrew
            TEXT("Καλημέρα κόσμε / שלום עולם"),
            // 3-byte: CJK (Chinese, Japanese, Korean)
            TEXT("你好世界 — 日本語のテスト — 안녕하세요"),
            // 3-byte: Arabic with combining marks
            TEXT("مرحبا بالعالم؟"),
            // 3-byte: Devanagari with combining marks (Hindi)
            TEXT("नमस्ते दुनिया।"),
            // 3-byte: Thai
            TEXT("สวัสดีชาวโลก"),
            // 4-byte: emoji (supplementary plane) + ZWJ sequences
            TEXT("👋🌏 Hello! — 👨‍👩‍👧‍👦 family — 🇺🇳 flag"),
            // Mix of all of the above in one string
            TEXT("Mixed: ascii + café + 你好 + Привет + 🚀 + مرحبا + नमस्ते."),
            // Whitespace + CJK punctuation only
            TEXT("\t你好。\n世界？\r\n！"),
            // Empty + single chars
            TEXT(""),
            TEXT("a"),
            TEXT("你"),
            TEXT("👋"),
        };
        return Specimens;
    }
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FLlamaUTF8ConversionRoundTripTest,
    "LlamaCore.UTF8.ConversionRoundTrip",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FLlamaUTF8ConversionRoundTripTest::RunTest(const FString& /*Parameters*/)
{
    for (const FString& Original : Utf8Specimens())
    {
        // FString → std::string (UTF-8) → FString
        const std::string Stage1 = FLlamaString::ToStd(Original);
        const FString Recovered  = FLlamaString::ToUE(Stage1);
        TestEqual(FString::Printf(TEXT("Round-trip preserves [%s] (%d chars)"), *Original, Original.Len()),
                  Recovered, Original);

        // Byte-level: UE TCHAR_TO_UTF8 must produce identical bytes to a manually computed
        // length (no truncation at high codepoints, no embedded NUL surprises).
        if (!Original.IsEmpty())
        {
            const int32 ExpectedBytes = FTCHARToUTF8(*Original).Length();
            TestEqual(FString::Printf(TEXT("UTF-8 byte length stable for [%s]"), *Original),
                      static_cast<int32>(Stage1.size()), ExpectedBytes);
        }

        // Triple round-trip — guards against any layer that truncates on second pass.
        const std::string Stage2 = FLlamaString::ToStd(Recovered);
        const FString  Recovered2 = FLlamaString::ToUE(Stage2);
        TestEqual(TEXT("Triple round-trip stable"), Recovered2, Original);
    }
    return true;
}

// ─── Model-gated: end-to-end user-message round-trip ─────────────────────────

namespace
{
    /** First chat model present under Saved/Models, in priority order. */
    static FString FindChatModel()
    {
        const FString Root = FPaths::ProjectSavedDir() / TEXT("Models");
        const TArray<FString> Candidates = {
            TEXT("google_gemma-3-4b-it-Q4_K_L.gguf"),     // ~2.5 GB, multilingual, fastest
            TEXT("gemma-4-E2B-it-Q6_K.gguf"),
            TEXT("Qwen3.5-9B-Q4_K_M.gguf"),
            TEXT("Qwen2.5-Omni-7B-Q4_K_M.gguf"),
        };
        for (const FString& F : Candidates)
        {
            const FString Full = Root / F;
            if (FPaths::FileExists(Full))
            {
                // Convert to absolute so FLlamaPaths::ParsePathIntoFullPath doesn't treat the
                // leading `..` as the Saved/Models-relative `.` prefix.
                return FPaths::ConvertRelativePathToFull(Full);
            }
        }
        return FString();
    }

    /** Pump the GT ticker for up to TimeoutSec while Predicate() returns false.
     *  Returns true if Predicate became true within the budget. */
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

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FLlamaUTF8UserMessageRoundTripTest,
    "LlamaCore.UTF8.UserMessageRoundTrip",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FLlamaUTF8UserMessageRoundTripTest::RunTest(const FString& /*Parameters*/)
{
    const FString ModelPath = FindChatModel();
    if (ModelPath.IsEmpty())
    {
        AddInfo(TEXT("LlamaCore.UTF8.UserMessageRoundTrip: no chat model in Saved/Models, skipping integration check (conversion-helper test still ran)."));
        return true;
    }
    AddInfo(FString::Printf(TEXT("Using model: %s"), *ModelPath));

    FLlamaDualBackend Backend;
    Backend.Initialize();
    if (!Backend.GetLlamaNative())
    {
        AddError(TEXT("FLlamaNative not constructed"));
        return false;
    }
    Backend.GetLlamaNative()->AddTicker(); // drain GT callbacks via core ticker

    Backend.ModelParams.PathToModel = ModelPath;
    Backend.ModelParams.MaxContextLength = 2048;
    Backend.ModelParams.GPULayers = 99;
    Backend.ModelParams.bAutoInsertSystemPromptOnLoad = false;
    Backend.ModelParams.SystemPrompt = TEXT("");

    bool bModelLoaded = false;
    bool bModelError = false;
    Backend.OnModelLoaded = [&bModelLoaded](const FString&) { bModelLoaded = true; };
    Backend.OnError       = [&bModelError](const FString& Err, int32) { bModelError = true; UE_LOG(LogTemp, Warning, TEXT("Backend error: %s"), *Err); };

    Backend.LoadModel(/*bForceReload=*/false);

    if (!WaitFor(120.0, [&]() { return bModelLoaded || bModelError; }))
    {
        AddError(TEXT("Model load timed out (120s)"));
        Backend.GetLlamaNative()->RemoveTicker();
        return false;
    }
    if (bModelError)
    {
        AddError(TEXT("Model load reported an error — check log"));
        Backend.GetLlamaNative()->RemoveTicker();
        return false;
    }

    // The strings we'll insert as USER messages. Each one we expect to find back
    // verbatim in ModelState.ChatHistory after the bGenerateReply=false call settles.
    const TArray<FString> Probes = {
        TEXT("你好世界,这是一条测试消息。"),                  // Simplified Chinese
        TEXT("こんにちは世界、テストメッセージです。"),         // Japanese
        TEXT("Привет мир — это тестовое сообщение."),        // Russian Cyrillic
        TEXT("مرحبا بالعالم، هذه رسالة اختبار."),            // Arabic (RTL)
        TEXT("नमस्ते दुनिया, यह एक परीक्षण संदेश है।"),         // Hindi (Devanagari combining)
        TEXT("Mixed 🚀 emoji + 你好 + ascii + ñoño."),       // 4-byte + mixed
    };

    bool bAllRoundTrippedExactly = true;

    for (const FString& Probe : Probes)
    {
        const int32 BeforeCount = Backend.ModelState.ChatHistory.History.Num();

        bool bPromptProcessed = false;
        Backend.OnPromptProcessed = [&bPromptProcessed](int32, EChatTemplateRole, float) { bPromptProcessed = true; };

        FLlamaChatPrompt P;
        P.Prompt = Probe;
        P.Role = EChatTemplateRole::User;
        P.bAddAssistantBOS = false;
        P.bGenerateReply = false; // history-only — fast and deterministic
        Backend.InsertTemplatedPrompt(P);

        // Wait for the BG task to land + GT callback to fire.
        const bool bSettled = WaitFor(30.0,
            [&]() { return bPromptProcessed && Backend.ModelState.ChatHistory.History.Num() > BeforeCount; });

        if (!bSettled)
        {
            AddError(FString::Printf(TEXT("Probe never landed in history: %s"), *Probe));
            bAllRoundTrippedExactly = false;
            continue;
        }

        const FStructuredChatMessage& Last = Backend.ModelState.ChatHistory.History.Last();
        const bool bRoleOk    = (Last.Role == EChatTemplateRole::User);
        const bool bContentOk = (Last.Content == Probe);

        if (!bRoleOk)
        {
            AddError(FString::Printf(TEXT("Wrong role on last message for probe: %s"), *Probe));
            bAllRoundTrippedExactly = false;
        }
        if (!bContentOk)
        {
            // Surface both for diagnosis — Equals will short-circuit; print byte counts too.
            AddError(FString::Printf(
                TEXT("Probe round-trip MISMATCH:\n  expected (%d chars / %d bytes): %s\n  got      (%d chars / %d bytes): %s"),
                Probe.Len(), FTCHARToUTF8(*Probe).Length(), *Probe,
                Last.Content.Len(), FTCHARToUTF8(*Last.Content).Length(), *Last.Content));
            bAllRoundTrippedExactly = false;
        }
        else
        {
            AddInfo(FString::Printf(TEXT("OK [%d chars / %d bytes]: %s"),
                Probe.Len(), FTCHARToUTF8(*Probe).Length(), *Probe));
        }
    }

    Backend.UnloadModel();
    Backend.GetLlamaNative()->RemoveTicker();

    return bAllRoundTrippedExactly;
}

#endif // WITH_DEV_AUTOMATION_TESTS
