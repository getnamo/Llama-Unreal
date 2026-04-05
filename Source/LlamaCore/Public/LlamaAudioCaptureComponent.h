// Copyright 2025-current Getnamo.

#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "LlamaMediaCaptureTypes.h"
#include "HAL/ThreadSafeBool.h"
#include "Containers/Queue.h"
#include "LlamaDataTypes.h"

#include "LlamaAudioCaptureComponent.generated.h"

namespace Audio { class FAudioCapture; }

/**
 * Standalone microphone capture component with optional voice activity detection.
 * Captures audio, resamples to 16 kHz mono, and dispatches segments to registered consumers.
 *
 * Consumer dispatch:
 *   - C++ consumers implement ILlamaAudioConsumer and register via AddConsumer/RemoveConsumer.
 *     Segments are delivered directly on the BG thread (zero game-thread hop).
 *   - Blueprint consumers bind OnAudioSegmentReady (requires bEnableGameThreadDelegate = true).
 *
 * Threading: Audio HW thread -> ring buffer -> VAD -> BG thread dispatch -> consumers.
 */
UCLASS(Category = "LLM", BlueprintType, meta = (BlueprintSpawnableComponent))
class LLAMACORE_API ULlamaAudioCaptureComponent : public UActorComponent
{
    GENERATED_BODY()

public:
    ULlamaAudioCaptureComponent(const FObjectInitializer& ObjectInitializer);
    ~ULlamaAudioCaptureComponent();

    virtual void TickComponent(float DeltaTime, ELevelTick TickType,
                               FActorComponentTickFunction* ThisTickFunction) override;
    virtual void BeginDestroy() override;

    // -----------------------------------------------------------------------
    // Configuration
    // -----------------------------------------------------------------------

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Audio Capture")
    ELlamaVADMode VADMode = ELlamaVADMode::EnergyBased;

    /** RMS threshold for energy-based VAD [0.0-1.0]. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Audio Capture",
        meta = (ClampMin = "0.0", ClampMax = "1.0", EditCondition = "VADMode == ELlamaVADMode::EnergyBased"))
    float VADThreshold = 0.02f;

    /** Silence hold time before speech offset (energy mode). */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Audio Capture",
        meta = (ClampMin = "0.1", EditCondition = "VADMode == ELlamaVADMode::EnergyBased"))
    float VADHoldTimeSec = 0.8f;

    /** Silero speech probability threshold [0.0-1.0]. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Audio Capture",
        meta = (ClampMin = "0.0", ClampMax = "1.0", EditCondition = "VADMode == ELlamaVADMode::Silero"))
    float SileroThreshold = 0.5f;

    /** Silence hold time for Silero mode. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Audio Capture",
        meta = (ClampMin = "0.05", EditCondition = "VADMode == ELlamaVADMode::Silero"))
    float SileroHoldTimeSec = 0.2f;

    /** Path to ggml Silero VAD model (relative to Saved/Models or absolute). */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Audio Capture",
        meta = (EditCondition = "VADMode == ELlamaVADMode::Silero"))
    FString PathToVADModel = TEXT("./ggml-silero-v6.2.0.bin");

    /** Pre-roll seconds to include before speech onset. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Audio Capture",
        meta = (ClampMin = "0.0", ClampMax = "2.0"))
    float VADPreRollSec = 0.15f;

    /** Max segment duration before forced dispatch. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Audio Capture",
        meta = (ClampMin = "1.0"))
    float MaxSpeechSegmentSec = 15.0f;

    /** Overlap between consecutive chunks (Disabled VAD mode only). */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Audio Capture",
        meta = (ClampMin = "0.0", ClampMax = "5.0"))
    float NonVADOverlapSec = 0.5f;

    /** Ring buffer capacity in samples at 16 kHz. Default 30 seconds. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Audio Capture",
        meta = (ClampMin = "16000"))
    int32 RingBufferCapacitySamples = 16000 * 30;

    /** If true, OnAudioSegmentReady fires on the game thread. Off by default —
     *  C++ consumers via AddConsumer() are the preferred zero-GT-hop path. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Audio Capture")
    bool bEnableGameThreadDelegate = false;

    // -----------------------------------------------------------------------
    // Delegates (game thread, opt-in via bEnableGameThreadDelegate)
    // -----------------------------------------------------------------------

    UPROPERTY(BlueprintAssignable, Category = "Audio Capture")
    FOnLlamaAudioSegmentReady OnAudioSegmentReady;

    UPROPERTY(BlueprintAssignable, Category = "Audio Capture")
    FOnLlamaVADStateChanged OnVADStateChanged;

    // -----------------------------------------------------------------------
    // C++ consumer registration (zero GT hop)
    // -----------------------------------------------------------------------

    void AddConsumer(ILlamaAudioConsumer* Consumer);
    void RemoveConsumer(ILlamaAudioConsumer* Consumer);

    // -----------------------------------------------------------------------
    // Blueprint consumer registration
    // -----------------------------------------------------------------------

    /** Register an actor component (e.g. WhisperComponent, LlamaComponent) as an audio consumer.
     *  The component must have registered itself via ILlamaAudioConsumer::RegisterComponent.
     *  Returns true if the component was successfully added. */
    UFUNCTION(BlueprintCallable, Category = "Audio Capture")
    bool AddConsumerComponent(UActorComponent* Component);

    /** Remove a previously registered consumer component. Returns true if found and removed. */
    UFUNCTION(BlueprintCallable, Category = "Audio Capture")
    bool RemoveConsumerComponent(UActorComponent* Component);

    // -----------------------------------------------------------------------
    // Blueprint API
    // -----------------------------------------------------------------------

    UFUNCTION(BlueprintCallable, Category = "Audio Capture")
    void StartCapture();

    UFUNCTION(BlueprintCallable, Category = "Audio Capture")
    void StopCapture();

    UFUNCTION(BlueprintPure, Category = "Audio Capture")
    bool IsCaptureActive() const;

    UFUNCTION(BlueprintCallable, Category = "Audio Capture")
    void SetMuted(bool bMuted);

    UFUNCTION(BlueprintPure, Category = "Audio Capture")
    bool IsMuted() const;

    /** Snapshot the last N seconds from the ring buffer as a segment. */
    UFUNCTION(BlueprintCallable, Category = "Audio Capture")
    FLlamaAudioSegment SnapshotAudio(float Seconds = 5.0f);

private:
    // -----------------------------------------------------------------------
    // Background thread
    // -----------------------------------------------------------------------
    void StartBGThread();

    TQueue<FLLMThreadTask, EQueueMode::Mpsc> BackgroundTasks;
    TQueue<FLLMThreadTask>                   GameThreadTasks;
    FThreadSafeBool bThreadIsActive = false;
    FThreadSafeBool bThreadShouldRun = false;
    FThreadSafeCounter TaskIdCounter = 0;

    int64 GetNextTaskId();
    void EnqueueBGTask(TFunction<void(int64)> Task);
    void EnqueueGTTask(TFunction<void()> Task);

    // -----------------------------------------------------------------------
    // Audio capture
    // -----------------------------------------------------------------------
    Audio::FAudioCapture* AudioCapture = nullptr;
    FThreadSafeBool bCaptureActive = false;
    FThreadSafeBool bMicMuted = false;
    int32 CaptureDeviceSampleRate = 48000;
    int32 CaptureDeviceChannels = 1;

    void HandleAudioCaptureCallback(const float* InAudio, int32 NumFrames,
                                     int32 NumChannels, int32 SampleRate);

    // -----------------------------------------------------------------------
    // Ring buffer (protected by AudioBufferMutex)
    // -----------------------------------------------------------------------
    FCriticalSection AudioBufferMutex;
    TArray<float> RingBuffer;
    int64 TotalSamplesWritten = 0;
    TArray<float> PreRollBuffer;
    int32 PreRollWritePos = 0;

    void AppendToRingBuffer_Locked(const TArray<float>& Samples16k);
    void CopyRingBufferSegment_Locked(int64 AbsStart, int64 AbsEnd, TArray<float>& OutSegment) const;

    // -----------------------------------------------------------------------
    // VAD processor
    // -----------------------------------------------------------------------
    ILlamaAudioProcessor* AudioProcessor = nullptr;
    void CreateAudioProcessor();
    void DestroyAudioProcessor();

    // For Silero BG thread processing
    int64 SileroChunkStart = 0;
    FThreadSafeBool bSileroInFlight = false;
    void ProcessSileroBGChunk(int64 ChunkEnd);

    // -----------------------------------------------------------------------
    // Consumers
    // -----------------------------------------------------------------------
    FCriticalSection ConsumersMutex;
    TArray<ILlamaAudioConsumer*> Consumers;

    void DispatchSegment(int64 AbsStart, int64 AbsEnd, bool bVADTriggered);
};
