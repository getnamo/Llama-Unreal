// Copyright 2025-current Getnamo.

#pragma once

#include "LlamaMediaCaptureTypes.h"

/**
 * Energy-based VAD using RMS threshold detection.
 * Runs inline on the audio HW thread (fast, no model).
 * Ported from FWhisperNative energy VAD logic.
 */
class FLlamaEnergyVAD : public ILlamaAudioProcessor
{
public:
    /** @param InThreshold    RMS threshold for voice onset [0.0-1.0], default 0.02
     *  @param InHoldTimeSec  Silence hold time before offset, default 0.8
     *  @param InPreRollSec   Pre-roll seconds to include before onset, default 0.15
     *  @param InMaxSegmentSec  Max segment duration before forced dispatch, default 15.0 */
    FLlamaEnergyVAD(float InThreshold = 0.02f, float InHoldTimeSec = 0.8f,
                     float InPreRollSec = 0.15f, float InMaxSegmentSec = 15.0f);

    virtual FVADDecision ProcessAudioChunk(const float* Samples, int32 Count,
                                            int64 AbsoluteWritePos, float DeltaSeconds) override;
    virtual void Reset() override;
    virtual bool RequiresBackgroundThread() const override { return false; }

    float Threshold = 0.02f;
    float HoldTimeSec = 0.8f;
    float PreRollSec = 0.15f;
    float MaxSegmentSec = 15.0f;

private:
    bool bSpeechActive = false;
    float SilenceDuration = 0.0f;
    float SegmentDuration = 0.0f;
    int64 SpeechStartSample = 0;

    static float ComputeRMS(const float* Samples, int32 Count);
};
