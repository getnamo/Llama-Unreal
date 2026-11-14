// Copyright 2025-current Getnamo.

#include "Misc/AutomationTest.h"

#if WITH_DEV_AUTOMATION_TESTS

#include "Internal/LlamaInternal.h"
#include "LlamaDataTypes.h"
#include "LlamaUtility.h"
#include "HAL/PlatformFileManager.h"
#include "Misc/CommandLine.h"
#include "Misc/Parse.h"

// Small-model tests that also run in packaged games (ClientContext), e.g. on Android devices and the
// emulator via the LlamaAndroidTest project (-LlamaTests=LlamaCore.Device). The model is resolved like a
// user's "./" path: Saved/Models on desktop, <GamePersistentDownloadDir>/Models on Android.
// Optional command line: -LlamaGPULayers=N (default -1 = all layers on the GPU backend if there is one).
namespace LlamaDeviceTests
{
    static const TCHAR* ModelFile = TEXT("./Qwen3-0.6B-Q8_0.gguf");

    static FString FindModel(FAutomationTestBase& Test)
    {
        const FString Path = FLlamaPaths::ParsePathIntoFullPath(ModelFile);
        if (!FPlatformFileManager::Get().GetPlatformFile().FileExists(*Path))
        {
            Test.AddInfo(FString::Printf(TEXT("Skipping: model not found at %s"), *Path));
            return FString();
        }
        return Path;
    }

    static FLLMModelParams MakeParams(const FString& ModelPath)
    {
        FLLMModelParams Params;
        Params.PathToModel = ModelPath;
        Params.MaxContextLength = 2048;
        Params.Advanced.Sampling.Temp = 0.f;
        Params.Advanced.Thinking.bEnableThinking = false;
        int32 GPULayers = -1;
        if (FParse::Value(FCommandLine::Get(), TEXT("LlamaGPULayers="), GPULayers))
        {
            Params.GPULayers = GPULayers;
        }
        return Params;
    }

    struct FRun
    {
        FString Response;
        float TokensPerSecond = 0.f;
        FLLMSpeculativeStats Stats;
        FString Errors;
    };

    static bool Run(FAutomationTestBase& Test, const FLLMModelParams& Params, const char* Prompt, FRun& Out,
        TFunction<void(FLlamaInternal&)> AfterLoad = nullptr)
    {
        FLlamaInternal Internal;
        Internal.OnGenerationComplete = [&Out](const std::string&, float, int32, float Tps) { Out.TokensPerSecond = Tps; };
        Internal.OnError = [&Out](const FString& Message, int32 Code) { Out.Errors += FString::Printf(TEXT("[%d] %s "), Code, *Message); };
        if (!Internal.LoadModelFromParams(Params))
        {
            Test.AddError(FString::Printf(TEXT("model load failed: %s"), *Out.Errors));
            return false;
        }
        if (AfterLoad)
        {
            AfterLoad(Internal);
        }
        Out.Response = UTF8_TO_TCHAR(Internal.InsertTemplatedPrompt(Prompt, EChatTemplateRole::User, true, true).c_str());
        Out.Stats = Internal.LastSpeculativeStats;
        Internal.UnloadModel();
        return true;
    }

    static const char* EchoPrompt =
        "Repeat the following paragraph back word for word, twice, with nothing else:\n\n"
        "The lighthouse keeper climbed the spiral stairs every evening at dusk. He trimmed the wick, "
        "polished the great lens until it gleamed, and wound the clockwork that turned the light. "
        "Ships passing the rocky point counted on that steady beam to find their way home through the fog. "
        "In winter the storms rattled the windows, but the keeper never missed a single night.";
}

#define LLAMA_DEVICE_TEST_FLAGS (EAutomationTestFlags::EditorContext | EAutomationTestFlags::ClientContext | EAutomationTestFlags::EngineFilter)

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FLlamaDeviceGenerateTest, "LlamaCore.Device.Generate", LLAMA_DEVICE_TEST_FLAGS)

bool FLlamaDeviceGenerateTest::RunTest(const FString& /*Parameters*/)
{
    using namespace LlamaDeviceTests;
    const FString Model = FindModel(*this);
    if (Model.IsEmpty())
    {
        return true;
    }

    FRun Result;
    if (!Run(*this, MakeParams(Model), "What is the capital of France? Answer with one word.", Result,
        [this](FLlamaInternal&)
        {
            //Which compute backends llama.cpp found (CPU, Vulkan, ...)
            for (size_t i = 0; i < ggml_backend_dev_count(); i++)
            {
                ggml_backend_dev_t Dev = ggml_backend_dev_get(i);
                AddInfo(FString::Printf(TEXT("Backend device %d: %hs (%hs)"), (int32)i, ggml_backend_dev_name(Dev), ggml_backend_dev_description(Dev)));
            }
        }))
    {
        return false;
    }
    AddInfo(FString::Printf(TEXT("Reply: %s (%.1f t/s)"), *Result.Response.TrimStartAndEnd(), Result.TokensPerSecond));
    TestTrue(TEXT("reply names the capital"), Result.Response.Contains(TEXT("Paris"), ESearchCase::IgnoreCase));
    TestTrue(FString::Printf(TEXT("no errors (%s)"), *Result.Errors), Result.Errors.IsEmpty());
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FLlamaDeviceGrammarTest, "LlamaCore.Device.Grammar", LLAMA_DEVICE_TEST_FLAGS)

bool FLlamaDeviceGrammarTest::RunTest(const FString& /*Parameters*/)
{
    using namespace LlamaDeviceTests;
    const FString Model = FindModel(*this);
    if (Model.IsEmpty())
    {
        return true;
    }

    FLLMModelParams Params = MakeParams(Model);
    Params.Advanced.Sampling.Grammar = TEXT("root ::= \"yes\" | \"no\"");
    FRun Result;
    if (!Run(*this, Params, "Is water wet? Answer yes or no.", Result))
    {
        return false;
    }
    const FString Reply = Result.Response.TrimStartAndEnd();
    AddInfo(FString::Printf(TEXT("Reply: %s"), *Reply));
    TestTrue(TEXT("grammar constrains the reply"), Reply == TEXT("yes") || Reply == TEXT("no"));
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FLlamaDeviceSpeculativeTest, "LlamaCore.Device.SpeculativeNGram", LLAMA_DEVICE_TEST_FLAGS)

bool FLlamaDeviceSpeculativeTest::RunTest(const FString& /*Parameters*/)
{
    using namespace LlamaDeviceTests;
    const FString Model = FindModel(*this);
    if (Model.IsEmpty())
    {
        return true;
    }

    FLLMModelParams Params = MakeParams(Model);
    FRun Normal, Spec;
    if (!Run(*this, Params, EchoPrompt, Normal))
    {
        return false;
    }
    Params.Advanced.Speculative.Mode = ELLMSpeculativeMode::NGram;
    if (!Run(*this, Params, EchoPrompt, Spec))
    {
        return false;
    }
    AddInfo(FString::Printf(TEXT("normal %.1f t/s | n-gram %.1f t/s | accepted %d/%d, %d passes"), Normal.TokensPerSecond,
        Spec.TokensPerSecond, Spec.Stats.AcceptedTokens, Spec.Stats.DraftedTokens, Spec.Stats.VerificationSteps));
    const int32 PrefixLen = FMath::Min(100, Normal.Response.Len());
    TestFalse(TEXT("non-empty reply"), Normal.Response.IsEmpty());
    TestEqual(TEXT("greedy output matches normal generation"), Spec.Response.Left(PrefixLen), Normal.Response.Left(PrefixLen));
    TestTrue(TEXT("drafts accepted"), Spec.Stats.AcceptedTokens > 0);
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FLlamaDeviceJinjaTest, "LlamaCore.Device.JinjaTemplate", LLAMA_DEVICE_TEST_FLAGS)

bool FLlamaDeviceJinjaTest::RunTest(const FString& /*Parameters*/)
{
    using namespace LlamaDeviceTests;
    const FString Model = FindModel(*this);
    if (Model.IsEmpty())
    {
        return true;
    }

    //A template the built-in templater doesn't recognize, so it goes through llama.cpp's Jinja engine.
    //The empty think block keeps Qwen3 from reasoning (as its own template does with thinking off).
    FLLMModelParams Params = MakeParams(Model);
    Params.CustomChatTemplate.Jinja = TEXT(
        "{% for message in messages %}### {{ message.role }}:\n{{ message.content }}\n\n{% endfor %}"
        "{% if add_generation_prompt %}### assistant:\n<think>\n\n</think>\n\n{% endif %}");
    bool bJinja = false;
    FRun Result;
    if (!Run(*this, Params, "Say hello.", Result, [&bJinja](FLlamaInternal& Internal) { bJinja = Internal.IsUsingJinjaTemplate(); }))
    {
        return false;
    }
    AddInfo(FString::Printf(TEXT("Reply: %s"), *Result.Response.TrimStartAndEnd().Left(120)));
    TestTrue(TEXT("Jinja template in use"), bJinja);
    TestFalse(TEXT("non-empty reply"), Result.Response.TrimStartAndEnd().IsEmpty());
    TestTrue(FString::Printf(TEXT("no errors (%s)"), *Result.Errors), Result.Errors.IsEmpty());
    return true;
}

#endif // WITH_DEV_AUTOMATION_TESTS
