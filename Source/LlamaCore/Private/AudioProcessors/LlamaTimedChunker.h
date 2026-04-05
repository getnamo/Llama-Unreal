// Copyright 2025-current Getnamo.

#pragma once

#include "LlamaMediaCaptureTypes.h"

/**
 * Timed audio chunker — no voice detection, just dispatches at fixed intervals.
 * Runs inline on the audio HW thread.
 * Ported from FWhisperNative Disabled VAD mode logic.
 */
class FLlamaTimedChunker : public ILlamaAudioProcessor
{
public:
    /** @param InMaxSegmentSec  Dispatch interval in seconds, default 15.0
     *  @param InOverlapSec     Overlap between consecutive chunks, default 0.5 */
    FLlamaTimedChunker(float InMaxSegmentSec = 15.0f, float InOverlapSec = 0.5f);

    virtual FVADDecision ProcessAudioChunk(const float* Samples, int32 Count,
                                            int64 AbsoluteWritePos, float DeltaSeconds) override;
    virtual void Reset() override;
    virtual bool RequiresBackgroundThread() const override { return false; }

    float MaxSegmentSec = 15.0f;
    float OverlapSec = 0.5f;

private:
    float AccumulatedDuration = 0.0f;
    int64 SegmentStartSample = 0;
};
