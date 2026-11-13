// Copyright 2025-current Getnamo.

#include "Misc/AutomationTest.h"

#if WITH_DEV_AUTOMATION_TESTS

#include "Internal/WhisperInternal.h"
#include "WhisperNative.h"
#include "LlamaAudioUtils.h"
#include "Misc/Paths.h"

// whisper.cpp is compiled into this module against the ggml that ships with llama.cpp, so a llama.cpp
// update can break speech-to-text at runtime even when everything links. Needs local files; skipped otherwise.
namespace LlamaWhisperTests
{
    static FString FindFile(const FString& Path)
    {
        const FString Full = FPaths::ConvertRelativePathToFull(Path);
        return FPaths::FileExists(Full) ? Full : FString();
    }
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FLlamaWhisperTranscribeWavTest,
    "LlamaWhisper.Transcribe.WavFile",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FLlamaWhisperTranscribeWavTest::RunTest(const FString& /*Parameters*/)
{
    using namespace LlamaWhisperTests;

    const FString ModelPath = FindFile(FPaths::ProjectSavedDir() / TEXT("Models/ggml-small.en.bin"));
    const FString WavPath = FindFile(FPaths::ProjectDir() / TEXT("Notes/AudioSamples/SileroTest.wav"));
    if (ModelPath.IsEmpty() || WavPath.IsEmpty())
    {
        AddInfo(TEXT("Skipping: needs Saved/Models/ggml-small.en.bin and Notes/AudioSamples/SileroTest.wav"));
        return true;
    }

    TArray<float> Samples;
    int32 SampleRate = 0;
    if (!TestTrue(TEXT("WAV file loads"), FWhisperNative::LoadWavFile(WavPath, Samples, SampleRate)))
    {
        return false;
    }
    TArray<float> Samples16k;
    if (SampleRate != 16000)
    {
        ULlamaAudioUtils::ResampleLinear(Samples.GetData(), Samples.Num(), SampleRate, Samples16k, 16000);
    }
    else
    {
        Samples16k = MoveTemp(Samples);
    }

    FString Transcript;
    FString Errors;
    FWhisperInternal Internal;
    Internal.OnTranscriptionResult = [&Transcript](const std::string& Text, bool /*bIsFinal*/)
    {
        Transcript += UTF8_TO_TCHAR(Text.c_str());
    };
    Internal.OnError = [&Errors](const std::string& Message)
    {
        Errors += UTF8_TO_TCHAR(Message.c_str());
    };

    if (!TestTrue(TEXT("whisper model loads"), Internal.LoadModel(TCHAR_TO_UTF8(*ModelPath), /*bUseGPU*/ true, 4)))
    {
        return false;
    }
    const bool bTranscribed = Internal.TranscribeAudio(Samples16k.GetData(), Samples16k.Num(), "en", false,
        /*MaxContext*/ -1, /*BestOf*/ 5, /*BeamSize*/ 5, /*bUseBeamSearch*/ false);
    Internal.UnloadModel();

    AddInfo(FString::Printf(TEXT("Transcript (%.1fs of audio): %s"), Samples16k.Num() / 16000.f, *Transcript.TrimStartAndEnd()));
    TestTrue(TEXT("transcription succeeded"), bTranscribed);
    TestTrue(FString::Printf(TEXT("no errors (%s)"), *Errors), Errors.IsEmpty());
    TestFalse(TEXT("non-empty transcript"), Transcript.TrimStartAndEnd().IsEmpty());
    return true;
}

#endif // WITH_DEV_AUTOMATION_TESTS
