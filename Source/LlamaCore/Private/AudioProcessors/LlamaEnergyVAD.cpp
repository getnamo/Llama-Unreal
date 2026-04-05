// Copyright 2025-current Getnamo.

#include "AudioProcessors/LlamaEnergyVAD.h"

FLlamaEnergyVAD::FLlamaEnergyVAD(float InThreshold, float InHoldTimeSec,
                                   float InPreRollSec, float InMaxSegmentSec)
    : Threshold(InThreshold)
    , HoldTimeSec(InHoldTimeSec)
    , PreRollSec(InPreRollSec)
    , MaxSegmentSec(InMaxSegmentSec)
{
}

FVADDecision FLlamaEnergyVAD::ProcessAudioChunk(const float* Samples, int32 Count,
                                                  int64 AbsoluteWritePos, float DeltaSeconds)
{
    FVADDecision Decision;
    const float RMS = ComputeRMS(Samples, Count);
    const float ChunkDuration = DeltaSeconds;

    if (!bSpeechActive)
    {
        if (RMS > Threshold)
        {
            bSpeechActive = true;
            Decision.Result = FVADDecision::EResult::SpeechOnset;

            // Include pre-roll — the capture component handles the actual pre-roll buffer;
            // we just report where the segment should start.
            const int32 PreRollSamples = FMath::RoundToInt(PreRollSec * 16000.0f);
            SpeechStartSample = FMath::Max(static_cast<int64>(0), AbsoluteWritePos - Count - PreRollSamples);
            Decision.SegmentStartSample = SpeechStartSample;

            SilenceDuration = 0.0f;
            SegmentDuration = ChunkDuration;
        }
    }
    else
    {
        SegmentDuration += ChunkDuration;

        if (RMS < Threshold)
        {
            SilenceDuration += ChunkDuration;

            if (SilenceDuration >= HoldTimeSec)
            {
                // Voice offset
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

            // Safety valve: force dispatch if too long
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

void FLlamaEnergyVAD::Reset()
{
    bSpeechActive = false;
    SilenceDuration = 0.0f;
    SegmentDuration = 0.0f;
    SpeechStartSample = 0;
}

float FLlamaEnergyVAD::ComputeRMS(const float* Samples, int32 Count)
{
    if (Count <= 0)
    {
        return 0.0f;
    }

    double SumSq = 0.0;
    for (int32 i = 0; i < Count; ++i)
    {
        const double S = Samples[i];
        SumSq += S * S;
    }
    return static_cast<float>(FMath::Sqrt(SumSq / Count));
}
