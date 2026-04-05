// Copyright 2025-current Getnamo.

#include "AudioProcessors/LlamaSileroVAD.h"
#include "LlamaUtility.h"
#include "whisper.h"

static constexpr int32 SILERO_SAMPLE_RATE = 16000;

FLlamaSileroVAD::FLlamaSileroVAD(float InThreshold, float InHoldTimeSec,
                                   float InPreRollSec, float InMaxSegmentSec)
    : Threshold(InThreshold)
    , HoldTimeSec(InHoldTimeSec)
    , PreRollSec(InPreRollSec)
    , MaxSegmentSec(InMaxSegmentSec)
{
}

FLlamaSileroVAD::~FLlamaSileroVAD()
{
    UnloadModel();
}

bool FLlamaSileroVAD::LoadModel(const FString& ModelPath)
{
    UnloadModel();

    const FString FullPath = FLlamaPaths::ParsePathIntoFullPath(ModelPath);
    whisper_vad_context_params Params = whisper_vad_default_context_params();
    SileroContext = whisper_vad_init_from_file_with_params(TCHAR_TO_UTF8(*FullPath), Params);
    if (!SileroContext)
    {
        UE_LOG(LlamaLog, Error, TEXT("FLlamaSileroVAD: Failed to load model: %s"), *FullPath);
        return false;
    }

    UE_LOG(LlamaLog, Log, TEXT("FLlamaSileroVAD: Loaded model: %s"), *FullPath);
    return true;
}

void FLlamaSileroVAD::UnloadModel()
{
    if (SileroContext)
    {
        whisper_vad_free(SileroContext);
        SileroContext = nullptr;
    }
}

bool FLlamaSileroVAD::IsModelLoaded() const
{
    return SileroContext != nullptr;
}

FVADDecision FLlamaSileroVAD::ProcessAudioChunk(const float* Samples, int32 Count,
                                                  int64 AbsoluteWritePos, float /*DeltaSeconds*/)
{
    FVADDecision Decision;

    if (!SileroContext || Count <= 0)
    {
        return Decision;
    }

    // Append to residual and process complete 512-sample windows
    const int32 WindowSize = whisper_vad_n_window(SileroContext);
    Residual.Append(Samples, Count);

    bool bSpeechDetected = false;
    int32 WindowsProcessed = 0;

    while (Residual.Num() >= WindowSize)
    {
        const float Prob = whisper_vad_detect_speech_streaming(
            SileroContext, Residual.GetData());
        if (Prob >= Threshold)
        {
            bSpeechDetected = true;
        }
        Residual.RemoveAt(0, WindowSize, EAllowShrinking::No);
        WindowsProcessed++;
    }

    if (WindowsProcessed == 0)
    {
        return Decision;
    }

    const float ChunkDuration = static_cast<float>(WindowsProcessed * WindowSize) / SILERO_SAMPLE_RATE;

    // Onset/offset state machine
    if (!bSpeechActive)
    {
        if (bSpeechDetected)
        {
            bSpeechActive = true;
            Decision.Result = FVADDecision::EResult::SpeechOnset;

            const int32 PreRollSamples = FMath::RoundToInt(PreRollSec * SILERO_SAMPLE_RATE);
            SpeechStartSample = FMath::Max(static_cast<int64>(0),
                AbsoluteWritePos - Count - PreRollSamples);
            Decision.SegmentStartSample = SpeechStartSample;

            SilenceDuration = 0.0f;
            SegmentDuration = ChunkDuration;
        }
    }
    else
    {
        SegmentDuration += ChunkDuration;

        if (!bSpeechDetected)
        {
            SilenceDuration += ChunkDuration;

            if (SilenceDuration >= HoldTimeSec)
            {
                bSpeechActive = false;
                Decision.Result = FVADDecision::EResult::DispatchSegment;
                Decision.SegmentStartSample = SpeechStartSample;
                Decision.SegmentEndSample = AbsoluteWritePos;
                SpeechStartSample = AbsoluteWritePos;
                SilenceDuration = 0.0f;
                SegmentDuration = 0.0f;
            }
        }
        else
        {
            SilenceDuration = 0.0f;

            if (SegmentDuration >= MaxSegmentSec)
            {
                Decision.Result = FVADDecision::EResult::DispatchSegment;
                Decision.SegmentStartSample = SpeechStartSample;
                Decision.SegmentEndSample = AbsoluteWritePos;
                SpeechStartSample = AbsoluteWritePos;
                SegmentDuration = 0.0f;
            }
        }
    }

    return Decision;
}

void FLlamaSileroVAD::Reset()
{
    bSpeechActive = false;
    SilenceDuration = 0.0f;
    SegmentDuration = 0.0f;
    SpeechStartSample = 0;
    Residual.Reset();

    if (SileroContext)
    {
        whisper_vad_reset_state(SileroContext);
    }
}
