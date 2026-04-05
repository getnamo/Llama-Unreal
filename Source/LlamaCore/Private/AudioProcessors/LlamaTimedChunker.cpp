// Copyright 2025-current Getnamo.

#include "AudioProcessors/LlamaTimedChunker.h"

FLlamaTimedChunker::FLlamaTimedChunker(float InMaxSegmentSec, float InOverlapSec)
    : MaxSegmentSec(InMaxSegmentSec)
    , OverlapSec(InOverlapSec)
{
}

FVADDecision FLlamaTimedChunker::ProcessAudioChunk(const float* Samples, int32 Count,
                                                     int64 AbsoluteWritePos, float DeltaSeconds)
{
    FVADDecision Decision;
    AccumulatedDuration += DeltaSeconds;

    if (AccumulatedDuration >= MaxSegmentSec)
    {
        Decision.Result = FVADDecision::EResult::DispatchSegment;
        Decision.SegmentStartSample = SegmentStartSample;
        Decision.SegmentEndSample = AbsoluteWritePos;

        const int64 OverlapSamples = static_cast<int64>(OverlapSec * 16000.0f);
        SegmentStartSample = FMath::Max(static_cast<int64>(0), AbsoluteWritePos - OverlapSamples);
        AccumulatedDuration = OverlapSec;
    }

    return Decision;
}

void FLlamaTimedChunker::Reset()
{
    AccumulatedDuration = 0.0f;
    SegmentStartSample = 0;
}
