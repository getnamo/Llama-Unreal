// Copyright 2025-current Getnamo.

#pragma once

#include "CoreMinimal.h"
#include "Kismet/BlueprintFunctionLibrary.h"
#include "Sound/SoundWave.h"

#include "LlamaAudioUtils.generated.h"

/**
 * Utility functions for audio conversion, primarily for feeding audio into multimodal LLM prompts.
 */
UCLASS()
class LLAMACORE_API ULlamaAudioUtils : public UBlueprintFunctionLibrary
{
    GENERATED_BODY()
public:

    /**
     * Convert a USoundWave to a float PCM array.
     * Supports 16-bit PCM WAV data stored in the SoundWave's bulk data.
     * OutSampleRate and OutNumChannels reflect the source audio — callers may need to resample
     * to 16kHz mono before passing to InsertTemplateAudioPrompt.
     * Returns false if the SoundWave has no raw PCM data (e.g. streaming or compressed assets).
     */
    UFUNCTION(BlueprintCallable, Category = "Llama|Audio")
    static bool SoundWaveToPCMFloat(USoundWave* SoundWave, TArray<float>& OutPCM, int32& OutSampleRate, int32& OutNumChannels);

    /**
     * Downmix a multi-channel PCM float array to mono by averaging channels.
     * If NumChannels is already 1, returns the input unchanged.
     */
    UFUNCTION(BlueprintCallable, Category = "Llama|Audio")
    static TArray<float> PCMFloatToMono(const TArray<float>& PCM, int32 NumChannels);

    /**
     * Simple linear resampler. Resamples PCM float data from InSampleRate to OutSampleRate.
     * Quality is sufficient for speech; for music use a proper resampling library.
     */
    UFUNCTION(BlueprintCallable, Category = "Llama|Audio")
    static TArray<float> ResamplePCMFloat(const TArray<float>& PCM, int32 InSampleRate, int32 OutSampleRate);

    /**
     * Convenience function: converts USoundWave to 16kHz mono float PCM ready for multimodal audio prompts.
     * Internally calls SoundWaveToPCMFloat -> PCMFloatToMono -> ResamplePCMFloat as needed.
     * Returns false if conversion fails.
     */
    UFUNCTION(BlueprintCallable, Category = "Llama|Audio")
    static bool SoundWaveToLLMAudio(USoundWave* SoundWave, TArray<float>& OutPCM, int32 TargetSampleRate = 16000);

    // -----------------------------------------------------------------------
    // Low-level utilities (pointer-based, suitable for audio HW thread use)
    // -----------------------------------------------------------------------

    /** Average interleaved N-channel frames to mono. */
    static void DownmixToMono(const float* InInterleaved, int32 NumFrames, int32 NumChannels,
                              TArray<float>& OutMono);

    /** Linear interpolation resampler: mono float32 from InRate to OutRate. */
    static void ResampleLinear(const float* InSamples, int32 InCount, int32 InRate,
                               TArray<float>& OutSamples, int32 OutRate);

    /** Compute root-mean-square energy of a sample block. */
    static float ComputeRMS(const float* Samples, int32 Count);
};
