// Copyright 2025-current Getnamo.

#include "Misc/AutomationTest.h"

#if WITH_DEV_AUTOMATION_TESTS

#include "Internal/LlamaInternal.h"
#include "LlamaDataTypes.h"
#include "Misc/Paths.h"

// GBNF grammar support (getnamo/Llama-Unreal#41) and sampling-param forwarding. Needs a small local
// model; skipped when it isn't present.
namespace LlamaGrammarTests
{
    static FString FindTestModel()
    {
        const FString Path = FPaths::ConvertRelativePathToFull(
            FPaths::ProjectSavedDir() / TEXT("Models") / TEXT("google_gemma-3-4b-it-Q4_K_L.gguf"));
        return FPaths::FileExists(Path) ? Path : FString();
    }

    static const TCHAR* YesNoGrammar = TEXT("root ::= \"yes\" | \"no\"");

    static const TCHAR* MoodJsonGrammar = TEXT(
        "root ::= \"{\\\"mood\\\": \" mood \", \\\"intensity\\\": \" digit \"}\"\n"
        "mood ::= \"\\\"happy\\\"\" | \"\\\"sad\\\"\" | \"\\\"angry\\\"\"\n"
        "digit ::= [0-9]");

    static FLLMModelParams MakeParams(const FString& ModelPath, bool bUseCommonSampler)
    {
        FLLMModelParams Params;
        Params.PathToModel = ModelPath;
        Params.MaxContextLength = 4096;
        Params.Advanced.Sampling.bUseCommonSampler = bUseCommonSampler;
        Params.Advanced.Sampling.Grammar = YesNoGrammar;
        return Params;
    }

    static FString Ask(FLlamaInternal& Internal, const TCHAR* Question)
    {
        const std::string Response = Internal.InsertTemplatedPrompt(TCHAR_TO_UTF8(Question), EChatTemplateRole::User, true, true);
        return FString(UTF8_TO_TCHAR(Response.c_str())).TrimStartAndEnd();
    }
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FLlamaGrammarConstrainsOutputTest,
    "LlamaCore.Grammar.ConstrainsOutput",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FLlamaGrammarConstrainsOutputTest::RunTest(const FString& /*Parameters*/)
{
    using namespace LlamaGrammarTests;

    const FString ModelPath = FindTestModel();
    if (ModelPath.IsEmpty())
    {
        AddInfo(TEXT("Skipping: test model not found in Saved/Models"));
        return true;
    }

    // Both sampler paths: the common sampler needs a rebuild per response, the raw chain a reset
    for (const bool bUseCommonSampler : { true, false })
    {
        const TCHAR* Path = bUseCommonSampler ? TEXT("common") : TEXT("raw chain");
        FLlamaInternal Internal;
        FLLMModelParams Params = MakeParams(ModelPath, bUseCommonSampler);
        Params.Advanced.Sampling.Temp = 0.f; //greedy: the grammar must restrict, not distort, the choice
        if (!TestTrue(FString::Printf(TEXT("[%s] load"), Path), Internal.LoadModelFromParams(Params)))
        {
            return false;
        }

        //Fresh context per question so earlier answers can't prime later ones
        const FString Blue = Ask(Internal, TEXT("Is the sky usually blue on a clear day? Answer yes or no."));
        Internal.ResetContextHistory(false);
        const FString Fire = Ask(Internal, TEXT("Is fire cold? Answer yes or no."));
        Internal.ResetContextHistory(false);
        AddInfo(FString::Printf(TEXT("[%s] greedy: sky blue='%s' fire cold='%s'"), Path, *Blue, *Fire));
        TestEqual(FString::Printf(TEXT("[%s] sky is blue"), Path), Blue, FString(TEXT("yes")));
        TestEqual(FString::Printf(TEXT("[%s] fire is not cold"), Path), Fire, FString(TEXT("no")));

        //Consecutive responses in one conversation (the #41 bug: second response came back empty)

        const FString First = Ask(Internal, TEXT("Is the sky usually blue on a clear day?"));
        const FString Second = Ask(Internal, TEXT("Is fire cold?"));
        const FString Third = Ask(Internal, TEXT("Do fish live in water?"));
        AddInfo(FString::Printf(TEXT("[%s] answers: '%s' '%s' '%s'"), Path, *First, *Second, *Third));

        for (const FString& Answer : { First, Second, Third })
        {
            TestTrue(FString::Printf(TEXT("[%s] answer '%s' matches the yes|no grammar"), Path, *Answer),
                Answer == TEXT("yes") || Answer == TEXT("no"));
        }
        Internal.UnloadModel();
    }
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FLlamaGrammarRuntimeSwapTest,
    "LlamaCore.Grammar.RuntimeSwapAndInvalid",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FLlamaGrammarRuntimeSwapTest::RunTest(const FString& /*Parameters*/)
{
    using namespace LlamaGrammarTests;

    const FString ModelPath = FindTestModel();
    if (ModelPath.IsEmpty())
    {
        AddInfo(TEXT("Skipping: test model not found in Saved/Models"));
        return true;
    }

    FLLMModelParams Params = MakeParams(ModelPath, true);
    FLlamaInternal Internal;
    int32 LastErrorCode = 0;
    Internal.OnError = [&LastErrorCode](const FString&, int32 Code) { LastErrorCode = Code; };
    if (!TestTrue(TEXT("load"), Internal.LoadModelFromParams(Params)))
    {
        return false;
    }

    const FString YesNo = Ask(Internal, TEXT("Is snow white?"));
    TestTrue(TEXT("yes|no grammar applied"), YesNo == TEXT("yes") || YesNo == TEXT("no"));
    const int32 MessagesBefore = (int32)Internal.Messages.size();

    // Swap to a structured grammar mid-conversation: history kept, new shape enforced
    FLLMSamplingParams Json = Params.Advanced.Sampling;
    Json.Grammar = MoodJsonGrammar;
    TestTrue(TEXT("UpdateSamplingParams on loaded model"), Internal.UpdateSamplingParams(Json));
    TestEqual(TEXT("history kept across sampling swap"), (int32)Internal.Messages.size(), MessagesBefore);

    const FString Mood = Ask(Internal, TEXT("How does a lost puppy feel? Reply as JSON."));
    AddInfo(FString::Printf(TEXT("json answer: '%s'"), *Mood));
    const bool bMoodShape = Mood.StartsWith(TEXT("{\"mood\": \"")) && Mood.EndsWith(TEXT("}")) &&
        (Mood.Contains(TEXT("\"happy\"")) || Mood.Contains(TEXT("\"sad\"")) || Mood.Contains(TEXT("\"angry\"")));
    TestTrue(TEXT("json grammar shape"), bMoodShape);

    // Invalid grammar: reported (error 12), generation continues unconstrained instead of crashing
    FLLMSamplingParams Invalid = Params.Advanced.Sampling;
    Invalid.Grammar = TEXT("root ::= this is not ( valid gbnf");
    AddExpectedError(TEXT("Invalid GBNF grammar"), EAutomationExpectedErrorFlags::Contains, 1);
    AddExpectedError(TEXT("[llama]"), EAutomationExpectedErrorFlags::Contains, 0);
    Internal.UpdateSamplingParams(Invalid);
    TestEqual(TEXT("invalid grammar reported"), LastErrorCode, 12);
    TestFalse(TEXT("no active grammar after invalid"), Internal.HasActiveGrammar());

    Internal.UnloadModel();
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FLlamaSamplingTempForwardedTest,
    "LlamaCore.Grammar.TempForwardedToCommonSampler",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FLlamaSamplingTempForwardedTest::RunTest(const FString& /*Parameters*/)
{
    using namespace LlamaGrammarTests;

    const FString ModelPath = FindTestModel();
    if (ModelPath.IsEmpty())
    {
        AddInfo(TEXT("Skipping: test model not found in Saved/Models"));
        return true;
    }

    // Temp 0 (greedy) must make the common sampler deterministic; it used to be silently ignored
    FLLMModelParams Params = MakeParams(ModelPath, true);
    Params.Advanced.Sampling.Grammar.Empty();
    Params.Advanced.Sampling.Temp = 0.f;

    FLlamaInternal Internal;
    if (!TestTrue(TEXT("load"), Internal.LoadModelFromParams(Params)))
    {
        return false;
    }

    const TCHAR* Question = TEXT("Name one planet in the solar system. Reply with a single word.");
    const FString A = Ask(Internal, Question);
    Internal.ResetContextHistory(false);
    const FString B = Ask(Internal, Question);
    AddInfo(FString::Printf(TEXT("greedy answers: '%s' / '%s'"), *A, *B));
    TestFalse(TEXT("non-empty answer"), A.IsEmpty());
    TestEqual(TEXT("temp 0 is deterministic"), B, A);

    Internal.UnloadModel();
    return true;
}

#endif // WITH_DEV_AUTOMATION_TESTS
