// Copyright 2025-current Getnamo.

#include "Misc/AutomationTest.h"

#if WITH_DEV_AUTOMATION_TESTS

#include "Internal/LlamaInternal.h"
#include "LlamaDataTypes.h"
#include "Misc/Paths.h"

// Speculative decoding (getnamo/Llama-Unreal#56). Greedy (temp 0) output must match normal generation
// and proposals must actually be accepted. Needs local models; skipped otherwise.
//
// Why "match" means a shared 100-char prefix rather than equality: verification decodes several tokens
// per batch, which uses different GPU kernels than one-token decoding, so near-tie greedy choices can
// flip later in a long response (e.g. "rotated"/"rotating"). llama.cpp has the same property.
namespace LlamaSpeculativeTests
{
    static FString FindModel(const TCHAR* FileName)
    {
        const FString Path = FPaths::ConvertRelativePathToFull(FPaths::ProjectSavedDir() / TEXT("Models") / FileName);
        return FPaths::FileExists(Path) ? Path : FString();
    }

    struct FRunResult
    {
        FString Response;
        float TokensPerSecond = 0.f;
        int32 Tokens = 0;
        FLLMSpeculativeStats Stats;
    };

    static bool RunOnce(FAutomationTestBase& Test, const FLLMModelParams& Params, const TCHAR* Prompt, FRunResult& Out)
    {
        FLlamaInternal Internal;
        Internal.OnGenerationComplete = [&Out](const std::string&, float, int32 Tokens, float Tps)
        {
            Out.Tokens = Tokens;
            Out.TokensPerSecond = Tps;
        };
        if (!Internal.LoadModelFromParams(Params))
        {
            Test.AddError(TEXT("model load failed"));
            return false;
        }
        const std::string Response = Internal.InsertTemplatedPrompt(TCHAR_TO_UTF8(Prompt), EChatTemplateRole::User, true, true);
        Out.Response = UTF8_TO_TCHAR(Response.c_str());
        Out.Stats = Internal.LastSpeculativeStats;
        Internal.UnloadModel();
        return true;
    }

    static void CompareRuns(FAutomationTestBase& Test, const TCHAR* Label, const FRunResult& Normal, const FRunResult& Spec)
    {
        Test.AddInfo(FString::Printf(TEXT("[%s] normal %d tok @ %.1f t/s | speculative %d tok @ %.1f t/s (x%.2f) | accepted %d/%d (%.0f%%), %.2f tok/pass"),
            Label, Normal.Tokens, Normal.TokensPerSecond, Spec.Tokens, Spec.TokensPerSecond,
            Normal.TokensPerSecond > 0.f ? Spec.TokensPerSecond / Normal.TokensPerSecond : 0.f,
            Spec.Stats.AcceptedTokens, Spec.Stats.DraftedTokens, Spec.Stats.AcceptanceRate * 100.f, Spec.Stats.TokensPerStep));

        Test.TestFalse(FString::Printf(TEXT("[%s] non-empty response"), Label), Normal.Response.IsEmpty());
        const int32 PrefixLen = FMath::Min(100, Normal.Response.Len());
        Test.TestEqual(FString::Printf(TEXT("[%s] greedy output matches normal generation"), Label), Spec.Response.Left(PrefixLen), Normal.Response.Left(PrefixLen));
        Test.TestEqual(FString::Printf(TEXT("[%s] normal run reports no speculation"), Label), Normal.Stats.VerificationSteps, 0);
        Test.TestTrue(FString::Printf(TEXT("[%s] speculative run verified batches"), Label), Spec.Stats.VerificationSteps > 0);
        Test.TestTrue(FString::Printf(TEXT("[%s] some drafted tokens accepted"), Label), Spec.Stats.AcceptedTokens > 0);
    }

    static const TCHAR* Passage = TEXT(
        "The lighthouse keeper climbed the spiral stairs every evening at dusk. He trimmed the wick, "
        "polished the great lens until it gleamed, and wound the clockwork that turned the light. "
        "Ships passing the rocky point counted on that steady beam to find their way home through the fog.");
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FLlamaSpeculativeNGramTest,
    "LlamaCore.Speculative.NGram",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FLlamaSpeculativeNGramTest::RunTest(const FString& /*Parameters*/)
{
    using namespace LlamaSpeculativeTests;

    const FString ModelPath = FindModel(TEXT("google_gemma-3-4b-it-Q4_K_L.gguf"));
    if (ModelPath.IsEmpty())
    {
        AddInfo(TEXT("Skipping: test model not found in Saved/Models"));
        return true;
    }

    FLLMModelParams Params;
    Params.PathToModel = ModelPath;
    Params.MaxContextLength = 4096;
    Params.Advanced.Sampling.Temp = 0.f;

    //Echoing text is where n-gram self-speculation pays off
    const FString Prompt = FString::Printf(TEXT("Repeat the following paragraph back word for word, twice, with nothing else:\n\n%s"), Passage);

    FRunResult Normal, Spec;
    if (!RunOnce(*this, Params, *Prompt, Normal))
    {
        return false;
    }
    Params.Advanced.Speculative.Mode = ELLMSpeculativeMode::NGram;
    if (!RunOnce(*this, Params, *Prompt, Spec))
    {
        return false;
    }
    CompareRuns(*this, TEXT("ngram"), Normal, Spec);
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FLlamaSpeculativeDraftModelTest,
    "LlamaCore.Speculative.DraftModel",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FLlamaSpeculativeDraftModelTest::RunTest(const FString& /*Parameters*/)
{
    using namespace LlamaSpeculativeTests;

    const FString Target = FindModel(TEXT("Qwen3-14B-Q6_K.gguf"));
    const FString Draft = FindModel(TEXT("Qwen3-0.6B-Q8_0.gguf"));
    if (Target.IsEmpty() || Draft.IsEmpty())
    {
        AddInfo(TEXT("Skipping: Qwen3-14B / Qwen3-0.6B not found in Saved/Models"));
        return true;
    }

    FLLMModelParams Params;
    Params.PathToModel = Target;
    Params.MaxContextLength = 4096;
    Params.Advanced.Sampling.Temp = 0.f;
    Params.Advanced.Thinking.bEnableThinking = false;

    const TCHAR* Prompt = TEXT("In one paragraph of about 120 words, explain how a lighthouse helps ships at night.");

    FRunResult Normal, Spec, SpecCombined;
    if (!RunOnce(*this, Params, Prompt, Normal))
    {
        return false;
    }

    Params.Advanced.Speculative.Mode = ELLMSpeculativeMode::DraftModel;
    Params.Advanced.Speculative.DraftModelPath = Draft;
    if (!RunOnce(*this, Params, Prompt, Spec))
    {
        return false;
    }
    CompareRuns(*this, TEXT("draft model"), Normal, Spec);

    Params.Advanced.Speculative.Mode = ELLMSpeculativeMode::DraftModelAndNGram;
    if (!RunOnce(*this, Params, Prompt, SpecCombined))
    {
        return false;
    }
    CompareRuns(*this, TEXT("draft model + ngram"), Normal, SpecCombined);
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FLlamaSpeculativeFallbackTest,
    "LlamaCore.Speculative.InvalidDraftFallsBack",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FLlamaSpeculativeFallbackTest::RunTest(const FString& /*Parameters*/)
{
    using namespace LlamaSpeculativeTests;

    const FString ModelPath = FindModel(TEXT("google_gemma-3-4b-it-Q4_K_L.gguf"));
    const FString OtherFamily = FindModel(TEXT("model-Qwen3-4b.gguf"));
    if (ModelPath.IsEmpty() || OtherFamily.IsEmpty())
    {
        AddInfo(TEXT("Skipping: test models not found in Saved/Models"));
        return true;
    }

    //A draft model with a different tokenizer must be rejected, and generation must still work
    FLLMModelParams Params;
    Params.PathToModel = ModelPath;
    Params.MaxContextLength = 2048;
    Params.Advanced.Sampling.Temp = 0.f;
    Params.Advanced.Speculative.Mode = ELLMSpeculativeMode::DraftModel;
    Params.Advanced.Speculative.DraftModelPath = OtherFamily;

    AddExpectedError(TEXT("tokenizer doesn't match"), EAutomationExpectedErrorFlags::Contains, 1);
    FRunResult Result;
    if (!RunOnce(*this, Params, TEXT("Name one planet in the solar system. Reply with a single word."), Result))
    {
        return false;
    }
    TestFalse(TEXT("still generates"), Result.Response.IsEmpty());
    TestEqual(TEXT("no speculation used"), Result.Stats.VerificationSteps, 0);
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FLlamaSpeculativeConversationTest,
    "LlamaCore.Speculative.ConversationAndRollback",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FLlamaSpeculativeConversationTest::RunTest(const FString& /*Parameters*/)
{
    using namespace LlamaSpeculativeTests;

    const FString ModelPath = FindModel(TEXT("Qwen3-14B-Q6_K.gguf"));
    const FString DraftPath = FindModel(TEXT("Qwen3-0.6B-Q8_0.gguf"));
    if (ModelPath.IsEmpty() || DraftPath.IsEmpty())
    {
        AddInfo(TEXT("Skipping: Qwen3-14B / Qwen3-0.6B not found in Saved/Models"));
        return true;
    }

    FLLMModelParams Params;
    Params.PathToModel = ModelPath;
    Params.MaxContextLength = 4096;
    Params.SystemPrompt = TEXT("You are a concise assistant.");
    Params.Advanced.Sampling.Temp = 0.f;
    Params.Advanced.Thinking.bEnableThinking = false;
    Params.Advanced.Speculative.Mode = ELLMSpeculativeMode::DraftModelAndNGram;
    Params.Advanced.Speculative.DraftModelPath = DraftPath;

    FLlamaInternal Internal;
    if (!TestTrue(TEXT("load"), Internal.LoadModelFromParams(Params)) || !TestTrue(TEXT("speculative active"), Internal.IsSpeculativeActive()))
    {
        return false;
    }
    Internal.InsertTemplatedPrompt(TCHAR_TO_UTF8(*Params.SystemPrompt), EChatTemplateRole::System, false, false);
    TestTrue(TEXT("mirror after system prompt"), Internal.IsTokenMirrorConsistent());

    const TCHAR* Turns[] = {
        TEXT("List three primary colors, comma separated."),
        TEXT("Now list them again in reverse order."),
        TEXT("And once more, in uppercase."),
    };
    for (const TCHAR* Turn : Turns)
    {
        const std::string Reply = Internal.InsertTemplatedPrompt(TCHAR_TO_UTF8(Turn), EChatTemplateRole::User, true, true);
        AddInfo(FString::Printf(TEXT("'%s' -> '%s' (accepted %d/%d)"), Turn, *FString(UTF8_TO_TCHAR(Reply.c_str())).TrimStartAndEnd(),
            Internal.LastSpeculativeStats.AcceptedTokens, Internal.LastSpeculativeStats.DraftedTokens));
        TestFalse(TEXT("reply not empty"), Reply.empty());
        TestTrue(TEXT("speculation used"), Internal.LastSpeculativeStats.VerificationSteps > 0);
        TestTrue(TEXT("mirror consistent after turn"), Internal.IsTokenMirrorConsistent());
    }

    //Roll back the last exchange and regenerate: KV, mirror and draft side must all rewind together
    Internal.RollbackContextHistoryByMessages(2);
    TestTrue(TEXT("mirror consistent after rollback"), Internal.IsTokenMirrorConsistent());
    const std::string Again = Internal.InsertTemplatedPrompt(TCHAR_TO_UTF8(Turns[2]), EChatTemplateRole::User, true, true);
    TestFalse(TEXT("regenerated reply not empty"), Again.empty());
    TestTrue(TEXT("mirror consistent after regenerate"), Internal.IsTokenMirrorConsistent());

    Internal.ResetContextHistory(false);
    TestTrue(TEXT("mirror empty after reset"), Internal.IsTokenMirrorConsistent());
    Internal.UnloadModel();
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FLlamaSpeculativeMTPTest,
    "LlamaCore.Speculative.MTP",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FLlamaSpeculativeMTPTest::RunTest(const FString& /*Parameters*/)
{
    using namespace LlamaSpeculativeTests;

    //Qwen3.5 is hybrid (attention + recurrent), so this also covers snapshot-based draft rollback
    const FString ModelPath = FindModel(TEXT("Qwen3.5-4B-MTP-Q4_K_M.gguf"));
    if (ModelPath.IsEmpty())
    {
        AddInfo(TEXT("Skipping: Qwen3.5-4B-MTP-Q4_K_M.gguf not found in Saved/Models"));
        return true;
    }

    FLLMModelParams Params;
    Params.PathToModel = ModelPath;
    Params.MaxContextLength = 4096;
    Params.Advanced.Sampling.Temp = 0.f;
    Params.Advanced.Thinking.bEnableThinking = false;

    const TCHAR* Prompt = TEXT("In one paragraph of about 120 words, explain how a lighthouse helps ships at night.");
    const FString EchoPrompt = FString::Printf(TEXT("Repeat the following paragraph back word for word, with nothing else:\n\n%s"), Passage);

    FRunResult Normal, MTP, NormalEcho, NGramHybrid;
    if (!RunOnce(*this, Params, Prompt, Normal) || !RunOnce(*this, Params, *EchoPrompt, NormalEcho))
    {
        return false;
    }

    Params.Advanced.Speculative.Mode = ELLMSpeculativeMode::MTP;
    if (!RunOnce(*this, Params, Prompt, MTP))
    {
        return false;
    }
    CompareRuns(*this, TEXT("mtp"), Normal, MTP);

    Params.Advanced.Speculative.Mode = ELLMSpeculativeMode::NGram;
    Params.Advanced.Speculative.DraftMaxTokens = 8;
    if (!RunOnce(*this, Params, *EchoPrompt, NGramHybrid))
    {
        return false;
    }
    CompareRuns(*this, TEXT("ngram on hybrid"), NormalEcho, NGramHybrid);
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FLlamaSpeculativeMTPConversationTest,
    "LlamaCore.Speculative.MTPConversation",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FLlamaSpeculativeMTPConversationTest::RunTest(const FString& /*Parameters*/)
{
    using namespace LlamaSpeculativeTests;

    const FString ModelPath = FindModel(TEXT("Qwen3.5-4B-MTP-Q4_K_M.gguf"));
    if (ModelPath.IsEmpty())
    {
        AddInfo(TEXT("Skipping: Qwen3.5-4B-MTP-Q4_K_M.gguf not found in Saved/Models"));
        return true;
    }

    FLLMModelParams Params;
    Params.PathToModel = ModelPath;
    Params.MaxContextLength = 4096;
    Params.SystemPrompt = TEXT("You are a concise assistant.");
    Params.Advanced.Sampling.Temp = 0.f;
    Params.Advanced.Thinking.bEnableThinking = false;
    Params.Advanced.Speculative.Mode = ELLMSpeculativeMode::MTPAndNGram;

    FLlamaInternal Internal;
    if (!TestTrue(TEXT("load"), Internal.LoadModelFromParams(Params)) || !TestTrue(TEXT("speculative active"), Internal.IsSpeculativeActive()))
    {
        return false;
    }
    Internal.InsertTemplatedPrompt(TCHAR_TO_UTF8(*Params.SystemPrompt), EChatTemplateRole::System, false, false);

    const TCHAR* Turns[] = {
        TEXT("List three primary colors, comma separated."),
        TEXT("Now list them again in reverse order."),
        TEXT("And once more, in uppercase."),
    };
    for (const TCHAR* Turn : Turns)
    {
        const std::string Reply = Internal.InsertTemplatedPrompt(TCHAR_TO_UTF8(Turn), EChatTemplateRole::User, true, true);
        AddInfo(FString::Printf(TEXT("'%s' -> '%s' (accepted %d/%d)"), Turn, *FString(UTF8_TO_TCHAR(Reply.c_str())).TrimStartAndEnd(),
            Internal.LastSpeculativeStats.AcceptedTokens, Internal.LastSpeculativeStats.DraftedTokens));
        TestFalse(TEXT("reply not empty"), Reply.empty());
        TestTrue(TEXT("speculation used"), Internal.LastSpeculativeStats.VerificationSteps > 0);
        TestTrue(TEXT("mirror consistent after turn"), Internal.IsTokenMirrorConsistent());
    }

    Internal.ResetContextHistory(false);
    TestTrue(TEXT("mirror empty after reset"), Internal.IsTokenMirrorConsistent());
    const std::string AfterReset = Internal.InsertTemplatedPrompt("Name one planet. Reply with a single word.", EChatTemplateRole::User, true, true);
    TestFalse(TEXT("reply after reset"), AfterReset.empty());
    TestTrue(TEXT("speculation used after reset"), Internal.LastSpeculativeStats.VerificationSteps > 0);
    Internal.UnloadModel();
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FLlamaSpeculativeMTPMissingHeadsTest,
    "LlamaCore.Speculative.MTPWithoutHeadsFallsBack",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FLlamaSpeculativeMTPMissingHeadsTest::RunTest(const FString& /*Parameters*/)
{
    using namespace LlamaSpeculativeTests;

    const FString ModelPath = FindModel(TEXT("google_gemma-3-4b-it-Q4_K_L.gguf"));
    if (ModelPath.IsEmpty())
    {
        AddInfo(TEXT("Skipping: test model not found in Saved/Models"));
        return true;
    }

    FLLMModelParams Params;
    Params.PathToModel = ModelPath;
    Params.MaxContextLength = 2048;
    Params.Advanced.Sampling.Temp = 0.f;
    Params.Advanced.Speculative.Mode = ELLMSpeculativeMode::MTP;

    AddExpectedError(TEXT("no MTP heads"), EAutomationExpectedErrorFlags::Contains, 1);
    FRunResult Result;
    if (!RunOnce(*this, Params, TEXT("Name one planet in the solar system. Reply with a single word."), Result))
    {
        return false;
    }
    TestFalse(TEXT("still generates"), Result.Response.IsEmpty());
    TestEqual(TEXT("no speculation used"), Result.Stats.VerificationSteps, 0);
    return true;
}

// Not part of LlamaCore.*: sweeps draft settings on a 14B target and logs a table (slow)
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FLlamaSpeculativeBenchmark,
    "LlamaBenchmark.Speculative.Qwen3",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FLlamaSpeculativeBenchmark::RunTest(const FString& /*Parameters*/)
{
    using namespace LlamaSpeculativeTests;

    const FString Target = FindModel(TEXT("Qwen3-14B-Q6_K.gguf"));
    const FString Small = FindModel(TEXT("Qwen3-0.6B-Q8_0.gguf"));
    const FString Medium = FindModel(TEXT("model-Qwen3-4b.gguf"));
    if (Target.IsEmpty() || Small.IsEmpty())
    {
        AddInfo(TEXT("Skipping: Qwen3-14B / Qwen3-0.6B not found in Saved/Models"));
        return true;
    }

    struct FConfig { const TCHAR* Name; ELLMSpeculativeMode Mode; FString Draft; int32 NMax; float PMin; };
    TArray<FConfig> Configs = {
        { TEXT("0.6B n3"),         ELLMSpeculativeMode::DraftModel, Small, 3, 0.f },
        { TEXT("0.6B n8"),         ELLMSpeculativeMode::DraftModel, Small, 8, 0.f },
        { TEXT("0.6B n8 p0.75"),   ELLMSpeculativeMode::DraftModel, Small, 8, 0.75f },
        { TEXT("0.6B n16 p0.75"),  ELLMSpeculativeMode::DraftModel, Small, 16, 0.75f },
        { TEXT("ngram"),           ELLMSpeculativeMode::NGram, FString(), 8, 0.f },
        { TEXT("0.6B n8 p0.75+ng"),ELLMSpeculativeMode::DraftModelAndNGram, Small, 8, 0.75f },
    };
    if (!Medium.IsEmpty())
    {
        Configs.Add({ TEXT("4B n4 p0.75"), ELLMSpeculativeMode::DraftModel, Medium, 4, 0.75f });
    }

    const TCHAR* Prompts[] = {
        TEXT("In one paragraph of about 120 words, explain how a lighthouse helps ships at night."),
        TEXT("Write a Python function that parses a CSV file into a list of dictionaries using the csv module, with a docstring and type hints. Only output the code."),
    };

    for (const TCHAR* Prompt : Prompts)
    {
        FLLMModelParams Params;
        Params.PathToModel = Target;
        Params.GPULayers = 999; //all layers: the plugin default (50) leaves part of a 27B on the CPU
        Params.MaxContextLength = 4096;
        Params.Advanced.Sampling.Temp = 0.f;
        Params.Advanced.Thinking.bEnableThinking = false;

        //Each measurement is the better of two runs: the first run of a new batch shape pays one-off
        //GPU pipeline warm-up that would otherwise dominate short generations
        auto BestOfTwo = [&](FRunResult& Out) -> bool
        {
            FRunResult A, B;
            if (!RunOnce(*this, Params, Prompt, A) || !RunOnce(*this, Params, Prompt, B))
            {
                return false;
            }
            Out = A.TokensPerSecond >= B.TokensPerSecond ? A : B;
            return true;
        };

        FRunResult Normal;
        if (!BestOfTwo(Normal))
        {
            return false;
        }
        AddInfo(FString::Printf(TEXT("--- %.40s... | baseline %d tok @ %.1f t/s"), Prompt, Normal.Tokens, Normal.TokensPerSecond));

        for (const FConfig& Config : Configs)
        {
            Params.Advanced.Speculative.Mode = Config.Mode;
            Params.Advanced.Speculative.DraftModelPath = Config.Draft;
            Params.Advanced.Speculative.DraftMaxTokens = Config.NMax;
            Params.Advanced.Speculative.DraftMinProbability = Config.PMin;
            FRunResult Spec;
            if (!BestOfTwo(Spec))
            {
                return false;
            }
            AddInfo(FString::Printf(TEXT("%-18s %6.1f t/s  x%.2f  accepted %3d/%3d (%3.0f%%)  %.2f tok/pass"),
                Config.Name, Spec.TokensPerSecond, Spec.TokensPerSecond / FMath::Max(Normal.TokensPerSecond, 0.01f),
                Spec.Stats.AcceptedTokens, Spec.Stats.DraftedTokens, Spec.Stats.AcceptanceRate * 100.f, Spec.Stats.TokensPerStep));
        }
    }
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FLlamaSpeculativeMTPBenchmark,
    "LlamaBenchmark.Speculative.MTP",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FLlamaSpeculativeMTPBenchmark::RunTest(const FString& /*Parameters*/)
{
    using namespace LlamaSpeculativeTests;

    const FString Target = FindModel(TEXT("Qwen3.8-27B-UD-Q4_K_M.gguf"));
    if (Target.IsEmpty())
    {
        AddInfo(TEXT("Skipping: Qwen3.8-27B-UD-Q4_K_M.gguf not found in Saved/Models"));
        return true;
    }

    struct FConfig { const TCHAR* Name; ELLMSpeculativeMode Mode; int32 NMax; };
    const FConfig Configs[] = {
        { TEXT("mtp n1"),       ELLMSpeculativeMode::MTP, 1 },
        { TEXT("mtp n2"),       ELLMSpeculativeMode::MTP, 2 },
        { TEXT("mtp n3"),       ELLMSpeculativeMode::MTP, 3 },
        { TEXT("mtp+ngram n1"), ELLMSpeculativeMode::MTPAndNGram, 1 },
        { TEXT("ngram n8"),     ELLMSpeculativeMode::NGram, 8 },
    };
    const TCHAR* Prompts[] = {
        TEXT("In one paragraph of about 120 words, explain how a lighthouse helps ships at night."),
        TEXT("Write a Python function that parses a CSV file into a list of dictionaries using the csv module, with a docstring and type hints. Only output the code."),
    };

    for (const TCHAR* Prompt : Prompts)
    {
        FLLMModelParams Params;
        Params.PathToModel = Target;
        Params.GPULayers = 999; //all layers: the plugin default (50) leaves part of a 27B on the CPU
        Params.MaxContextLength = 4096;
        Params.Advanced.Sampling.Temp = 0.f;
        Params.Advanced.Thinking.bEnableThinking = false;

        auto BestOfTwo = [&](FRunResult& Out) -> bool
        {
            FRunResult A, B;
            if (!RunOnce(*this, Params, Prompt, A) || !RunOnce(*this, Params, Prompt, B))
            {
                return false;
            }
            Out = A.TokensPerSecond >= B.TokensPerSecond ? A : B;
            return true;
        };

        FRunResult Normal;
        if (!BestOfTwo(Normal))
        {
            return false;
        }
        AddInfo(FString::Printf(TEXT("--- %.40s... | baseline %d tok @ %.1f t/s"), Prompt, Normal.Tokens, Normal.TokensPerSecond));

        for (const FConfig& Config : Configs)
        {
            Params.Advanced.Speculative.Mode = Config.Mode;
            Params.Advanced.Speculative.DraftMaxTokens = Config.NMax;
            FRunResult Spec;
            if (!BestOfTwo(Spec))
            {
                return false;
            }
            AddInfo(FString::Printf(TEXT("%-14s %6.1f t/s  x%.2f  accepted %3d/%3d (%3.0f%%)  %.2f tok/pass"),
                Config.Name, Spec.TokensPerSecond, Spec.TokensPerSecond / FMath::Max(Normal.TokensPerSecond, 0.01f),
                Spec.Stats.AcceptedTokens, Spec.Stats.DraftedTokens, Spec.Stats.AcceptanceRate * 100.f, Spec.Stats.TokensPerStep));
        }
    }
    return true;
}

#endif // WITH_DEV_AUTOMATION_TESTS
