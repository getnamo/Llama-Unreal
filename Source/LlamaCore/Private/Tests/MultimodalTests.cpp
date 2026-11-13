// Copyright 2025-current Getnamo.

#include "Misc/AutomationTest.h"

#if WITH_DEV_AUTOMATION_TESTS

#include "Internal/LlamaInternal.h"
#include "LlamaDataTypes.h"
#include "Misc/Paths.h"

// Multimodal (mtmd) smoke tests: an image and an audio clip go through mtmd_tokenize /
// mtmd_helper_eval_chunks, then a text-only follow-up checks that generation continues at the right
// KV position. Needs local models; skipped otherwise.
namespace LlamaMultimodalTests
{
    static FString FindFile(const FString& Path)
    {
        const FString Full = FPaths::ConvertRelativePathToFull(Path);
        return FPaths::FileExists(Full) ? Full : FString();
    }

    static FString FindModel(const TCHAR* FileName)
    {
        return FindFile(FPaths::ProjectSavedDir() / TEXT("Models") / FileName);
    }

    static FLLMModelParams MakeParams(const FString& ModelPath, const FString& MmprojPath)
    {
        FLLMModelParams Params;
        Params.PathToModel = ModelPath;
        Params.MmprojPath = MmprojPath;
        Params.MaxContextLength = 4096;
        Params.Advanced.Sampling.Temp = 0.f;
        Params.Advanced.Thinking.bEnableThinking = false;
        return Params;
    }

    static std::string Ask(FLlamaInternal& Internal, const TCHAR* Text, const TArray<FLlamaMediaEntry>& Media)
    {
        return Internal.InsertMultimodalPrompt(TCHAR_TO_UTF8(Text), Media, EChatTemplateRole::User, true, true);
    }
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FLlamaMultimodalImageTest,
    "LlamaCore.Multimodal.Image",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FLlamaMultimodalImageTest::RunTest(const FString& /*Parameters*/)
{
    using namespace LlamaMultimodalTests;

    const FString Model = FindModel(TEXT("Qwen2.5-Omni-7B-Q4_K_M.gguf"));
    const FString Mmproj = FindModel(TEXT("mmproj-Qwen2.5-Omni-7B-Q8_0.gguf"));
    if (Model.IsEmpty() || Mmproj.IsEmpty())
    {
        AddInfo(TEXT("Skipping: Qwen2.5-Omni-7B + mmproj not found in Saved/Models"));
        return true;
    }

    FString Errors;
    FLlamaInternal Internal;
    Internal.OnError = [&Errors](const FString& Message, int32 Code)
    {
        Errors += FString::Printf(TEXT("[%d] %s "), Code, *Message);
    };
    if (!TestTrue(TEXT("model loads"), Internal.LoadModelFromParams(MakeParams(Model, Mmproj))))
    {
        return false;
    }
    TestTrue(TEXT("projector loaded"), Internal.IsMultimodalLoaded());
    TestTrue(TEXT("vision supported"), Internal.SupportsVision());

    //Solid red RGB image, built in memory
    FLlamaMediaEntry Image;
    Image.MediaType = ELlamaMediaType::Image;
    Image.ImageWidth = 256;
    Image.ImageHeight = 256;
    Image.ImageRGBData.SetNumUninitialized(Image.ImageWidth * Image.ImageHeight * 3);
    for (int32 i = 0; i < Image.ImageWidth * Image.ImageHeight; i++)
    {
        Image.ImageRGBData[i * 3 + 0] = 230;
        Image.ImageRGBData[i * 3 + 1] = 20;
        Image.ImageRGBData[i * 3 + 2] = 20;
    }

    const FString ImageReply = UTF8_TO_TCHAR(Ask(Internal, TEXT("<__media__>\nWhat color is this image? Answer with one word."), { Image }).c_str());
    AddInfo(FString::Printf(TEXT("Image reply: %s"), *ImageReply));
    TestTrue(TEXT("image reply names the color"), ImageReply.Contains(TEXT("red"), ESearchCase::IgnoreCase));

    //Text-only follow-up in the same conversation
    const FString TextReply = UTF8_TO_TCHAR(Internal.InsertTemplatedPrompt("What color is the sky on a clear day? Answer with one word.", EChatTemplateRole::User, true, true).c_str());
    AddInfo(FString::Printf(TEXT("Follow-up reply: %s"), *TextReply));
    TestTrue(TEXT("follow-up reply is coherent"), TextReply.Contains(TEXT("blue"), ESearchCase::IgnoreCase));

    Internal.UnloadModel();
    TestTrue(FString::Printf(TEXT("no errors (%s)"), *Errors), Errors.IsEmpty());
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FLlamaMultimodalAudioTest,
    "LlamaCore.Multimodal.Audio",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FLlamaMultimodalAudioTest::RunTest(const FString& /*Parameters*/)
{
    using namespace LlamaMultimodalTests;

    const FString Model = FindModel(TEXT("Qwen2.5-Omni-7B-Q4_K_M.gguf"));
    const FString Mmproj = FindModel(TEXT("mmproj-Qwen2.5-Omni-7B-Q8_0.gguf"));
    const FString Wav = FindFile(FPaths::ProjectDir() / TEXT("Notes/AudioSamples/SileroTest.wav"));
    if (Model.IsEmpty() || Mmproj.IsEmpty() || Wav.IsEmpty())
    {
        AddInfo(TEXT("Skipping: needs Qwen2.5-Omni-7B + mmproj in Saved/Models and Notes/AudioSamples/SileroTest.wav"));
        return true;
    }

    FString Errors;
    FLlamaInternal Internal;
    Internal.OnError = [&Errors](const FString& Message, int32 Code)
    {
        Errors += FString::Printf(TEXT("[%d] %s "), Code, *Message);
    };
    if (!TestTrue(TEXT("model loads"), Internal.LoadModelFromParams(MakeParams(Model, Mmproj))))
    {
        return false;
    }
    if (!TestTrue(TEXT("audio supported"), Internal.SupportsAudio()))
    {
        Internal.UnloadModel();
        return false;
    }

    //File path input: mtmd decodes and resamples the WAV itself
    FLlamaMediaEntry Audio;
    Audio.MediaType = ELlamaMediaType::Audio;
    Audio.FilePath = Wav;

    const FString AudioReply = UTF8_TO_TCHAR(Ask(Internal, TEXT("<__media__>\nTranscribe what is said in this audio."), { Audio }).c_str());
    AddInfo(FString::Printf(TEXT("Audio reply: %s"), *AudioReply));
    TestFalse(TEXT("non-empty audio reply"), AudioReply.TrimStartAndEnd().IsEmpty());

    const FString TextReply = UTF8_TO_TCHAR(Internal.InsertTemplatedPrompt("What color is the sky on a clear day? Answer with one word.", EChatTemplateRole::User, true, true).c_str());
    AddInfo(FString::Printf(TEXT("Follow-up reply: %s"), *TextReply));
    TestTrue(TEXT("follow-up reply is coherent"), TextReply.Contains(TEXT("blue"), ESearchCase::IgnoreCase));

    Internal.UnloadModel();
    TestTrue(FString::Printf(TEXT("no errors (%s)"), *Errors), Errors.IsEmpty());
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FLlamaMultimodalJinjaTemplateTest,
    "LlamaCore.Multimodal.ImageJinjaTemplate",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FLlamaMultimodalJinjaTemplateTest::RunTest(const FString& /*Parameters*/)
{
    using namespace LlamaMultimodalTests;

    //Gemma 4 ships a Jinja-only chat template that llama_chat_apply_template doesn't recognize, so it
    //goes through the llama-unreal-chat Jinja shim (and used to crash before that existed)
    const FString Model = FindModel(TEXT("gemma-4-E2B-it-Q6_K.gguf"));
    const FString Mmproj = FindModel(TEXT("mmproj-gemma-4-E2B-it-Q6_K-F16.gguf"));
    if (Model.IsEmpty() || Mmproj.IsEmpty())
    {
        AddInfo(TEXT("Skipping: gemma-4-E2B-it + mmproj not found in Saved/Models"));
        return true;
    }

    FString Errors;
    FLlamaInternal Internal;
    Internal.OnError = [&Errors](const FString& Message, int32 Code)
    {
        Errors += FString::Printf(TEXT("[%d] %s "), Code, *Message);
    };
    if (!TestTrue(TEXT("model loads"), Internal.LoadModelFromParams(MakeParams(Model, Mmproj))))
    {
        return false;
    }
    TestTrue(TEXT("Jinja template in use"), Internal.IsUsingJinjaTemplate());

    FLlamaMediaEntry Image;
    Image.MediaType = ELlamaMediaType::Image;
    Image.ImageWidth = 256;
    Image.ImageHeight = 256;
    Image.ImageRGBData.SetNumUninitialized(Image.ImageWidth * Image.ImageHeight * 3);
    for (int32 i = 0; i < Image.ImageWidth * Image.ImageHeight; i++)
    {
        Image.ImageRGBData[i * 3 + 0] = 20;
        Image.ImageRGBData[i * 3 + 1] = 40;
        Image.ImageRGBData[i * 3 + 2] = 220;
    }

    const FString ImageReply = UTF8_TO_TCHAR(Ask(Internal, TEXT("<__media__>\nWhat color is this image? Answer with one word."), { Image }).c_str());
    AddInfo(FString::Printf(TEXT("Image reply: %s"), *ImageReply));
    TestTrue(TEXT("image reply names the color"), ImageReply.Contains(TEXT("blue"), ESearchCase::IgnoreCase));

    const FString TextReply = UTF8_TO_TCHAR(Internal.InsertTemplatedPrompt("What color is grass? Answer with one word.", EChatTemplateRole::User, true, true).c_str());
    AddInfo(FString::Printf(TEXT("Follow-up reply: %s"), *TextReply));
    TestTrue(TEXT("follow-up reply is coherent"), TextReply.Contains(TEXT("green"), ESearchCase::IgnoreCase));

    Internal.UnloadModel();
    TestTrue(FString::Printf(TEXT("no errors (%s)"), *Errors), Errors.IsEmpty());
    return true;
}

#endif // WITH_DEV_AUTOMATION_TESTS
