// Copyright 2025-current Getnamo.

#pragma once

#include "LlamaMediaCaptureTypes.h"

struct whisper_vad_context;

/**
 * Neural VAD using the ggml Silero model via whisper.cpp's VAD API.
 * Must run on a background thread (RequiresBackgroundThread() = true).
 *
 * The capture component calls ProcessAudioChunk on its BG thread with
 * accumulated audio. This class manages the Silero LSTM streaming state
 * and 512-sample windowing internally.
 */
class FLlamaSileroVAD : public ILlamaAudioProcessor
{
public:
    FLlamaSileroVAD(float InThreshold = 0.5f, float InHoldTimeSec = 0.2f,
                     float InPreRollSec = 0.15f, float InMaxSegmentSec = 15.0f);
    ~FLlamaSileroVAD();

    /** Load the Silero VAD model from the given path. Must be called before ProcessAudioChunk. */
    bool LoadModel(const FString& ModelPath);
    void UnloadModel();
    bool IsModelLoaded() const;

    virtual FVADDecision ProcessAudioChunk(const float* Samples, int32 Count,
                                            int64 AbsoluteWritePos, float DeltaSeconds) override;
    virtual void Reset() override;
    virtual bool RequiresBackgroundThread() const override { return true; }

    float Threshold = 0.5f;
    float HoldTimeSec = 0.2f;
    float PreRollSec = 0.15f;
    float MaxSegmentSec = 15.0f;

private:
    whisper_vad_context* SileroContext = nullptr;
    TArray<float> Residual;

    // VAD state machine
    bool bSpeechActive = false;
    float SilenceDuration = 0.0f;
    float SegmentDuration = 0.0f;
    int64 SpeechStartSample = 0;
};
