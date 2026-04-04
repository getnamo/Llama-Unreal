// Copyright 2025-current Getnamo.

#include "LlamaAudioUtils.h"
#include "LlamaUtility.h"
#include "Audio.h"

bool ULlamaAudioUtils::SoundWaveToPCMFloat(USoundWave* SoundWave, TArray<float>& OutPCM, int32& OutSampleRate, int32& OutNumChannels)
{
    OutPCM.Reset();
    OutSampleRate = 0;
    OutNumChannels = 0;

    if (!SoundWave)
    {
        UE_LOG(LlamaLog, Warning, TEXT("SoundWaveToPCMFloat: null SoundWave passed"));
        return false;
    }

#if WITH_EDITORONLY_DATA
    // Editor path: asset stores the original WAV in RawData bulk data
    if (SoundWave->RawData.GetPayloadSize() > 0)
    {
        TFuture<FSharedBuffer> Future = SoundWave->RawData.GetPayload();
        FSharedBuffer WavBytes = Future.Get();

        if (!WavBytes.IsNull() && WavBytes.GetSize() > 0)
        {
            FWaveModInfo WaveInfo;
            FString ErrorReason;
            if (!WaveInfo.ReadWaveInfo(static_cast<const uint8*>(WavBytes.GetData()), (int32)WavBytes.GetSize(), &ErrorReason))
            {
                UE_LOG(LlamaLog, Warning, TEXT("SoundWaveToPCMFloat: failed to parse WAV for '%s': %s"), *SoundWave->GetName(), *ErrorReason);
                return false;
            }

            const int32 BitsPerSample = (int32)*WaveInfo.pBitsPerSample;
            if (BitsPerSample != 16)
            {
                UE_LOG(LlamaLog, Warning, TEXT("SoundWaveToPCMFloat: only 16-bit PCM supported (got %d-bit) for '%s'"), BitsPerSample, *SoundWave->GetName());
                return false;
            }

            OutSampleRate = (int32)*WaveInfo.pSamplesPerSec;
            OutNumChannels = (int32)*WaveInfo.pChannels;
            const int32 NumSamples = WaveInfo.SampleDataSize / sizeof(int16);
            OutPCM.SetNum(NumSamples);
            const int16* Src = reinterpret_cast<const int16*>(WaveInfo.SampleDataStart);
            for (int32 i = 0; i < NumSamples; i++)
            {
                OutPCM[i] = Src[i] / 32768.f;
            }
            return true;
        }
    }
#endif

    // Runtime path: decoded PCM is available after the sound has been loaded for playback
    if (SoundWave->RawPCMData && SoundWave->RawPCMDataSize > 0)
    {
        OutSampleRate = SoundWave->GetSampleRateForCurrentPlatform();
        OutNumChannels = SoundWave->NumChannels;

        if (OutSampleRate <= 0 || OutNumChannels <= 0)
        {
            UE_LOG(LlamaLog, Warning, TEXT("SoundWaveToPCMFloat: invalid sample rate or channel count for '%s'"), *SoundWave->GetName());
            return false;
        }

        const int32 NumSamples = SoundWave->RawPCMDataSize / sizeof(int16);
        OutPCM.SetNum(NumSamples);
        const int16* Src = reinterpret_cast<const int16*>(SoundWave->RawPCMData);
        for (int32 i = 0; i < NumSamples; i++)
        {
            OutPCM[i] = Src[i] / 32768.f;
        }
        return true;
    }

    UE_LOG(LlamaLog, Warning, TEXT("SoundWaveToPCMFloat: no PCM data available on '%s'. Ensure the asset is not streaming."), *SoundWave->GetName());
    return false;
}

TArray<float> ULlamaAudioUtils::PCMFloatToMono(const TArray<float>& PCM, int32 NumChannels)
{
    if (NumChannels <= 1 || PCM.Num() == 0)
    {
        return PCM;
    }

    const int32 NumFrames = PCM.Num() / NumChannels;
    TArray<float> Mono;
    Mono.SetNum(NumFrames);

    const float Scale = 1.f / NumChannels;
    for (int32 Frame = 0; Frame < NumFrames; Frame++)
    {
        float Sum = 0.f;
        for (int32 Ch = 0; Ch < NumChannels; Ch++)
        {
            Sum += PCM[Frame * NumChannels + Ch];
        }
        Mono[Frame] = Sum * Scale;
    }
    return Mono;
}

TArray<float> ULlamaAudioUtils::ResamplePCMFloat(const TArray<float>& PCM, int32 InSampleRate, int32 OutSampleRate)
{
    if (InSampleRate == OutSampleRate || PCM.Num() == 0)
    {
        return PCM;
    }

    if (InSampleRate <= 0 || OutSampleRate <= 0)
    {
        UE_LOG(LlamaLog, Warning, TEXT("ResamplePCMFloat: invalid sample rates %d -> %d"), InSampleRate, OutSampleRate);
        return PCM;
    }

    const double Ratio = (double)OutSampleRate / (double)InSampleRate;
    const int32 OutNumSamples = FMath::RoundToInt(PCM.Num() * Ratio);
    TArray<float> Out;
    Out.SetNum(OutNumSamples);

    for (int32 i = 0; i < OutNumSamples; i++)
    {
        const double SrcPos = i / Ratio;
        const int32 SrcIdx = (int32)SrcPos;
        const float Frac = (float)(SrcPos - SrcIdx);

        const float A = PCM[SrcIdx];
        const float B = (SrcIdx + 1 < PCM.Num()) ? PCM[SrcIdx + 1] : A;
        Out[i] = A + Frac * (B - A);
    }
    return Out;
}

bool ULlamaAudioUtils::SoundWaveToLLMAudio(USoundWave* SoundWave, TArray<float>& OutPCM, int32 TargetSampleRate)
{
    int32 SampleRate = 0;
    int32 NumChannels = 0;

    if (!SoundWaveToPCMFloat(SoundWave, OutPCM, SampleRate, NumChannels))
    {
        return false;
    }

    if (NumChannels > 1)
    {
        OutPCM = PCMFloatToMono(OutPCM, NumChannels);
    }

    if (SampleRate != TargetSampleRate)
    {
        OutPCM = ResamplePCMFloat(OutPCM, SampleRate, TargetSampleRate);
    }

    return true;
}
