// Copyright 2025-current Getnamo.

#include "Misc/AutomationTest.h"

#if WITH_DEV_AUTOMATION_TESTS

#include "Internal/LlamaInternal.h"
#include "LlamaDataTypes.h"
#include "Misc/Paths.h"

// Chat templates: llama.cpp's built-in templater for every template it recognizes, and llama.cpp's
// Jinja engine (through the llama-unreal-chat shim) for the rest. Needs local models; skipped otherwise.
namespace LlamaChatTemplateTests
{
    static FString FindModel(const TCHAR* FileName)
    {
        const FString Path = FPaths::ConvertRelativePathToFull(FPaths::ProjectSavedDir() / TEXT("Models") / FileName);
        return FPaths::FileExists(Path) ? Path : FString();
    }

    static FLLMModelParams MakeParams(const FString& ModelPath)
    {
        FLLMModelParams Params;
        Params.PathToModel = ModelPath;
        Params.MaxContextLength = 4096;
        Params.Advanced.Sampling.Temp = 0.f;
        Params.Advanced.Thinking.bEnableThinking = false;
        return Params;
    }

    static FString Say(FLlamaInternal& Internal, const char* Prompt)
    {
        return UTF8_TO_TCHAR(Internal.InsertTemplatedPrompt(Prompt, EChatTemplateRole::User, true, true).c_str());
    }
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FLlamaChatTemplateJinjaTest,
    "LlamaCore.ChatTemplate.JinjaConversation",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FLlamaChatTemplateJinjaTest::RunTest(const FString& /*Parameters*/)
{
    using namespace LlamaChatTemplateTests;

    const FString Model = FindModel(TEXT("gemma-4-E2B-it-Q6_K.gguf"));
    if (Model.IsEmpty())
    {
        AddInfo(TEXT("Skipping: gemma-4-E2B-it-Q6_K.gguf not found in Saved/Models"));
        return true;
    }

    FString Errors;
    FLlamaInternal Internal;
    Internal.OnError = [&Errors](const FString& Message, int32 Code)
    {
        Errors += FString::Printf(TEXT("[%d] %s "), Code, *Message);
    };
    if (!TestTrue(TEXT("model loads"), Internal.LoadModelFromParams(MakeParams(Model))))
    {
        return false;
    }
    TestTrue(TEXT("Jinja template in use (Gemma 4's template isn't a built-in one)"), Internal.IsUsingJinjaTemplate());

    //System message first, as LlamaNative does on load
    Internal.InsertTemplatedPrompt("You are a terse assistant. Answer in as few words as possible.", EChatTemplateRole::System, false, false);

    const FString First = Say(Internal, "What is the capital of France?");
    const FString Second = Say(Internal, "And what is the capital of Italy?");
    AddInfo(FString::Printf(TEXT("Turn 1: %s | Turn 2: %s"), *First.TrimStartAndEnd(), *Second.TrimStartAndEnd()));
    TestTrue(TEXT("turn 1 answers"), First.Contains(TEXT("Paris"), ESearchCase::IgnoreCase));
    TestTrue(TEXT("turn 2 answers in context"), Second.Contains(TEXT("Rome"), ESearchCase::IgnoreCase));

    //Thinking is off: the template is told so through enable_thinking, and no reasoning block comes back
    if (!Internal.ThinkingOpenTag.empty())
    {
        const FString OpenTag = UTF8_TO_TCHAR(Internal.ThinkingOpenTag.c_str());
        TestFalse(TEXT("no reasoning block in the reply"), First.Contains(OpenTag) || Second.Contains(OpenTag));
    }

    Internal.UnloadModel();
    TestTrue(FString::Printf(TEXT("no errors (%s)"), *Errors), Errors.IsEmpty());
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FLlamaChatTemplateBuiltinTest,
    "LlamaCore.ChatTemplate.BuiltinStaysBuiltin",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FLlamaChatTemplateBuiltinTest::RunTest(const FString& /*Parameters*/)
{
    using namespace LlamaChatTemplateTests;

    //Templates the built-in templater recognizes keep using it, so existing models behave as before
    const FString Model = FindModel(TEXT("Qwen3-0.6B-Q8_0.gguf"));
    if (Model.IsEmpty())
    {
        AddInfo(TEXT("Skipping: Qwen3-0.6B-Q8_0.gguf not found in Saved/Models"));
        return true;
    }

    FLlamaInternal Internal;
    if (!TestTrue(TEXT("model loads"), Internal.LoadModelFromParams(MakeParams(Model))))
    {
        return false;
    }
    TestFalse(TEXT("built-in templater in use"), Internal.IsUsingJinjaTemplate());
    TestFalse(TEXT("non-empty reply"), Say(Internal, "Say hello.").TrimStartAndEnd().IsEmpty());
    Internal.UnloadModel();
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FLlamaChatTemplateInvalidTest,
    "LlamaCore.ChatTemplate.InvalidTemplateErrors",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FLlamaChatTemplateInvalidTest::RunTest(const FString& /*Parameters*/)
{
    using namespace LlamaChatTemplateTests;

    //Neither a built-in template nor valid Jinja: must report an error, not crash (it used to crash in
    //vector::resize on the -1 from llama_chat_apply_template)
    const FString Model = FindModel(TEXT("Qwen3-0.6B-Q8_0.gguf"));
    if (Model.IsEmpty())
    {
        AddInfo(TEXT("Skipping: Qwen3-0.6B-Q8_0.gguf not found in Saved/Models"));
        return true;
    }

    FString Errors;
    FLlamaInternal Internal;
    Internal.OnError = [&Errors](const FString& Message, int32 Code)
    {
        Errors += FString::Printf(TEXT("[%d] %s "), Code, *Message);
    };
    FLLMModelParams Params = MakeParams(Model);
    Params.CustomChatTemplate.Jinja = TEXT("{% for message in messages %}{{ message.content ");
    if (!TestTrue(TEXT("model loads"), Internal.LoadModelFromParams(Params)))
    {
        return false;
    }
    TestFalse(TEXT("broken template not loaded as Jinja"), Internal.IsUsingJinjaTemplate());

    AddExpectedError(TEXT("Failed to apply the chat template"), EAutomationExpectedErrorFlags::Contains, 0);
    const FString Reply = Say(Internal, "Say hello.");
    TestTrue(TEXT("no reply generated"), Reply.IsEmpty());
    TestTrue(FString::Printf(TEXT("template error reported (%s)"), *Errors), Errors.Contains(TEXT("chat template")));

    Internal.UnloadModel();
    return true;
}

#endif // WITH_DEV_AUTOMATION_TESTS
