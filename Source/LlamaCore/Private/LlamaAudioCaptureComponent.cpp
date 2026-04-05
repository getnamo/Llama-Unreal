// Copyright 2025-current Getnamo.

#include "LlamaAudioCaptureComponent.h"
#include "LlamaAudioUtils.h"
#include "LlamaUtility.h"
#include "AudioProcessors/LlamaEnergyVAD.h"
#include "AudioProcessors/LlamaTimedChunker.h"
#include "AudioCaptureCore.h"
#include "HAL/PlatformProcess.h"
#include "Async/Async.h"

static constexpr int32 CAPTURE_SAMPLE_RATE = 16000;
static constexpr float BG_THREAD_IDLE_SLEEP = 0.005f;

// ---------------------------------------------------------------------------
// Construction / Destruction
// ---------------------------------------------------------------------------

ULlamaAudioCaptureComponent::ULlamaAudioCaptureComponent(const FObjectInitializer& ObjectInitializer)
    : UActorComponent(ObjectInitializer)
{
    PrimaryComponentTick.bCanEverTick = true;
    PrimaryComponentTick.bStartWithTickEnabled = true;
}

ULlamaAudioCaptureComponent::~ULlamaAudioCaptureComponent()
{
    // Safety net — normal cleanup should happen in BeginDestroy
    if (AudioCapture)
    {
        delete AudioCapture;
        AudioCapture = nullptr;
    }
    DestroyAudioProcessor();
}

void ULlamaAudioCaptureComponent::BeginDestroy()
{
    // Stop capture if still active
    if (bCaptureActive)
    {
        StopCapture();
    }

    // Stop BG thread with spin-wait
    bThreadShouldRun = false;
    {
        const double WaitStart = FPlatformTime::Seconds();
        constexpr double MaxWaitSec = 3.0;
        while (bThreadIsActive)
        {
            FPlatformProcess::Sleep(0.01f);
            if (FPlatformTime::Seconds() - WaitStart > MaxWaitSec)
            {
                UE_LOG(LlamaLog, Warning, TEXT("LlamaAudioCapture: BG thread did not exit within %.1f sec"), MaxWaitSec);
                break;
            }
        }
    }

    if (AudioCapture)
    {
        delete AudioCapture;
        AudioCapture = nullptr;
    }
    DestroyAudioProcessor();

    Super::BeginDestroy();
}

// ---------------------------------------------------------------------------
// Tick — drain game-thread task queue
// ---------------------------------------------------------------------------

void ULlamaAudioCaptureComponent::TickComponent(float DeltaTime, ELevelTick TickType,
                                                  FActorComponentTickFunction* ThisTickFunction)
{
    Super::TickComponent(DeltaTime, TickType, ThisTickFunction);

    // Drain game-thread callbacks (only populated when bEnableGameThreadDelegate)
    while (!GameThreadTasks.IsEmpty())
    {
        FLLMThreadTask Task;
        GameThreadTasks.Dequeue(Task);
        if (Task.TaskFunction)
        {
            Task.TaskFunction(Task.TaskId);
        }
    }
}

// ---------------------------------------------------------------------------
// Background thread (matches FLlamaNative pattern)
// ---------------------------------------------------------------------------

void ULlamaAudioCaptureComponent::StartBGThread()
{
    bThreadShouldRun = true;
    Async(EAsyncExecution::Thread, [this]
    {
        bThreadIsActive = true;

        while (bThreadShouldRun)
        {
            while (!BackgroundTasks.IsEmpty())
            {
                FLLMThreadTask Task;
                BackgroundTasks.Dequeue(Task);
                if (Task.TaskFunction)
                {
                    Task.TaskFunction(Task.TaskId);
                }
            }

            FPlatformProcess::Sleep(BG_THREAD_IDLE_SLEEP);
        }

        bThreadIsActive = false;
    });
}

int64 ULlamaAudioCaptureComponent::GetNextTaskId()
{
    return TaskIdCounter.Increment();
}

void ULlamaAudioCaptureComponent::EnqueueBGTask(TFunction<void(int64)> TaskFunction)
{
    // Lazy start the thread on first enqueue
    if (!bThreadIsActive)
    {
        StartBGThread();
    }

    FLLMThreadTask Task;
    Task.TaskId = GetNextTaskId();
    Task.TaskFunction = TaskFunction;
    BackgroundTasks.Enqueue(Task);
}

void ULlamaAudioCaptureComponent::EnqueueGTTask(TFunction<void()> TaskFunction)
{
    FLLMThreadTask Task;
    Task.TaskId = GetNextTaskId();
    Task.TaskFunction = [TaskFunction](int64 /*InTaskId*/)
    {
        TaskFunction();
    };
    GameThreadTasks.Enqueue(Task);
}

// ---------------------------------------------------------------------------
// Audio processor lifecycle
// ---------------------------------------------------------------------------

void ULlamaAudioCaptureComponent::CreateAudioProcessor()
{
    DestroyAudioProcessor();

    switch (VADMode)
    {
    case ELlamaVADMode::Disabled:
        AudioProcessor = new FLlamaTimedChunker(MaxSpeechSegmentSec, NonVADOverlapSec);
        break;

    case ELlamaVADMode::EnergyBased:
        AudioProcessor = new FLlamaEnergyVAD(VADThreshold, VADHoldTimeSec, VADPreRollSec, MaxSpeechSegmentSec);
        break;

    case ELlamaVADMode::Silero:
    {
        const FString FullPath = FLlamaPaths::ParsePathIntoFullPath(PathToVADModel);
        AudioProcessor = FLlamaAudioProcessorRegistry::Create(TEXT("Silero"), FullPath);
        if (!AudioProcessor)
        {
            UE_LOG(LlamaLog, Warning,
                TEXT("LlamaAudioCapture: Silero VAD not registered — falling back to EnergyBased."));
            AudioProcessor = new FLlamaEnergyVAD(VADThreshold, VADHoldTimeSec, VADPreRollSec, MaxSpeechSegmentSec);
        }
        break;
    }
    }
}

void ULlamaAudioCaptureComponent::DestroyAudioProcessor()
{
    if (AudioProcessor)
    {
        delete AudioProcessor;
        AudioProcessor = nullptr;
    }
}

// ---------------------------------------------------------------------------
// Blueprint consumer registration
// ---------------------------------------------------------------------------

bool ULlamaAudioCaptureComponent::AddConsumerComponent(UActorComponent* Component)
{
    ILlamaAudioConsumer* Consumer = ILlamaAudioConsumer::FindConsumerForComponent(Component);
    if (Consumer)
    {
        AddConsumer(Consumer);
        return true;
    }
    UE_LOG(LlamaLog, Warning, TEXT("LlamaAudioCapture: Component '%s' is not a registered audio consumer."), *GetNameSafe(Component));
    return false;
}

bool ULlamaAudioCaptureComponent::RemoveConsumerComponent(UActorComponent* Component)
{
    ILlamaAudioConsumer* Consumer = ILlamaAudioConsumer::FindConsumerForComponent(Component);
    if (Consumer)
    {
        RemoveConsumer(Consumer);
        return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
// StartCapture / StopCapture
// ---------------------------------------------------------------------------

void ULlamaAudioCaptureComponent::StartCapture()
{
    if (bCaptureActive)
    {
        UE_LOG(LlamaLog, Warning, TEXT("LlamaAudioCapture: Capture already active."));
        return;
    }

    // Create the VAD / chunker processor
    CreateAudioProcessor();

    // Allocate ring buffer
    {
        FScopeLock Lock(&AudioBufferMutex);
        RingBuffer.SetNumZeroed(RingBufferCapacitySamples);
        TotalSamplesWritten = 0;

        // Pre-roll buffer
        const int32 PreRollSamples = FMath::Max(1, FMath::CeilToInt(VADPreRollSec * CAPTURE_SAMPLE_RATE));
        PreRollBuffer.SetNumZeroed(PreRollSamples);
        PreRollWritePos = 0;
    }

    // Reset Silero state
    SileroChunkStart = 0;
    bSileroInFlight = false;

    // Create and open AudioCapture
    AudioCapture = new Audio::FAudioCapture();

    Audio::FCaptureDeviceInfo DeviceInfo;
    if (AudioCapture->GetCaptureDeviceInfo(DeviceInfo))
    {
        CaptureDeviceSampleRate = DeviceInfo.PreferredSampleRate;
        CaptureDeviceChannels = DeviceInfo.InputChannels;
    }
    else
    {
        UE_LOG(LlamaLog, Warning, TEXT("LlamaAudioCapture: Could not query capture device info, using defaults."));
        CaptureDeviceSampleRate = 48000;
        CaptureDeviceChannels = 1;
    }

    Audio::FAudioCaptureDeviceParams DeviceParams;

    const bool bOpened = AudioCapture->OpenAudioCaptureStream(
        DeviceParams,
        [this](const void* InAudio, int32 NumFrames, int32 NumChannels,
               int32 SampleRate, double /*StreamTime*/, bool /*bOverFlow*/)
        {
            if (SampleRate > 0) { CaptureDeviceSampleRate = SampleRate; }
            if (NumChannels > 0) { CaptureDeviceChannels = NumChannels; }
            HandleAudioCaptureCallback(
                reinterpret_cast<const float*>(InAudio), NumFrames, NumChannels, SampleRate);
        },
        /*NumFramesDesired=*/1024);

    if (!bOpened)
    {
        UE_LOG(LlamaLog, Error, TEXT("LlamaAudioCapture: Failed to open audio capture stream."));
        delete AudioCapture;
        AudioCapture = nullptr;
        DestroyAudioProcessor();
        return;
    }

    if (!AudioCapture->StartStream())
    {
        UE_LOG(LlamaLog, Error, TEXT("LlamaAudioCapture: Failed to start audio capture stream."));
        AudioCapture->CloseStream();
        delete AudioCapture;
        AudioCapture = nullptr;
        DestroyAudioProcessor();
        return;
    }

    bCaptureActive = true;
    UE_LOG(LlamaLog, Log, TEXT("LlamaAudioCapture: Started — device %d Hz %d ch, resampling to %d Hz mono."),
        CaptureDeviceSampleRate, CaptureDeviceChannels, CAPTURE_SAMPLE_RATE);
}

void ULlamaAudioCaptureComponent::StopCapture()
{
    if (!bCaptureActive)
    {
        return;
    }

    // Stop audio stream
    if (AudioCapture)
    {
        AudioCapture->StopStream();
        AudioCapture->CloseStream();
        delete AudioCapture;
        AudioCapture = nullptr;
    }

    bCaptureActive = false;

    // Flush remaining audio as final segment
    {
        FScopeLock Lock(&AudioBufferMutex);
        if (TotalSamplesWritten > 0)
        {
            // Dispatch everything from the last segment start (or 0) to current write pos
            const int64 BufferCapacity = RingBuffer.Num();
            const int64 AbsEnd = TotalSamplesWritten;
            // Dispatch the last few seconds as a final segment
            const int64 MaxFlushSamples = FMath::Min((int64)(MaxSpeechSegmentSec * CAPTURE_SAMPLE_RATE), TotalSamplesWritten);
            const int64 AbsStart = AbsEnd - MaxFlushSamples;

            if (AbsEnd > AbsStart)
            {
                // Release lock before dispatch (dispatch will re-lock briefly)
                Lock.Unlock();
                DispatchSegment(FMath::Max(AbsStart, (int64)0), AbsEnd, false);
            }
        }
    }

    DestroyAudioProcessor();

    UE_LOG(LlamaLog, Log, TEXT("LlamaAudioCapture: Stopped."));
}

bool ULlamaAudioCaptureComponent::IsCaptureActive() const
{
    return bCaptureActive;
}

void ULlamaAudioCaptureComponent::SetMuted(bool bMuted)
{
    bMicMuted = bMuted;
}

bool ULlamaAudioCaptureComponent::IsMuted() const
{
    return bMicMuted;
}

// ---------------------------------------------------------------------------
// Audio HW callback
// ---------------------------------------------------------------------------

void ULlamaAudioCaptureComponent::HandleAudioCaptureCallback(const float* InAudio, int32 NumFrames,
                                                              int32 NumChannels, int32 SampleRate)
{
    if (!bCaptureActive || bMicMuted)
    {
        return;
    }

    // Downmix to mono
    TArray<float> Mono;
    ULlamaAudioUtils::DownmixToMono(InAudio, NumFrames, NumChannels, Mono);

    // Resample to 16 kHz
    TArray<float> Resampled;
    if (SampleRate != CAPTURE_SAMPLE_RATE)
    {
        ULlamaAudioUtils::ResampleLinear(Mono.GetData(), Mono.Num(), SampleRate, Resampled, CAPTURE_SAMPLE_RATE);
    }
    else
    {
        Resampled = MoveTemp(Mono);
    }

    if (Resampled.Num() == 0)
    {
        return;
    }

    // VAD decision variables (set under lock, used outside)
    FVADDecision Decision;
    Decision.Result = FVADDecision::EResult::Continue;
    bool bNeedsBGProcessing = false;
    int64 CurrentWritePos = 0;

    {
        FScopeLock Lock(&AudioBufferMutex);

        // Update pre-roll circular buffer
        for (int32 i = 0; i < Resampled.Num(); ++i)
        {
            if (PreRollBuffer.Num() > 0)
            {
                PreRollBuffer[PreRollWritePos % PreRollBuffer.Num()] = Resampled[i];
                PreRollWritePos++;
            }
        }

        // Append to ring buffer
        AppendToRingBuffer_Locked(Resampled);
        CurrentWritePos = TotalSamplesWritten;

        // Process through VAD if available
        if (AudioProcessor)
        {
            const float ChunkDuration = (float)Resampled.Num() / (float)CAPTURE_SAMPLE_RATE;

            if (!AudioProcessor->RequiresBackgroundThread())
            {
                // Inline VAD processing (Energy, TimedChunker)
                Decision = AudioProcessor->ProcessAudioChunk(
                    Resampled.GetData(), Resampled.Num(), CurrentWritePos, ChunkDuration);
            }
            else
            {
                // Silero — needs BG thread
                bNeedsBGProcessing = true;
            }
        }
    }
    // Lock released

    // Handle inline VAD decision
    if (Decision.Result == FVADDecision::EResult::SpeechOnset)
    {
        // Fire VAD state change
        if (bEnableGameThreadDelegate)
        {
            EnqueueGTTask([this]()
            {
                OnVADStateChanged.Broadcast(true);
            });
        }
    }
    else if (Decision.Result == FVADDecision::EResult::DispatchSegment)
    {
        // Fire VAD offset
        if (bEnableGameThreadDelegate)
        {
            EnqueueGTTask([this]()
            {
                OnVADStateChanged.Broadcast(false);
            });
        }

        DispatchSegment(Decision.SegmentStartSample, Decision.SegmentEndSample, true);
    }

    // Enqueue Silero BG processing if needed
    if (bNeedsBGProcessing && !bSileroInFlight)
    {
        bSileroInFlight = true;
        const int64 ChunkEnd = CurrentWritePos;
        EnqueueBGTask([this, ChunkEnd](int64 /*TaskId*/)
        {
            ProcessSileroBGChunk(ChunkEnd);
        });
    }
}

// ---------------------------------------------------------------------------
// Silero BG thread processing
// ---------------------------------------------------------------------------

void ULlamaAudioCaptureComponent::ProcessSileroBGChunk(int64 ChunkEnd)
{
    if (!AudioProcessor || !bCaptureActive)
    {
        bSileroInFlight = false;
        return;
    }

    // Copy the audio chunk from ring buffer
    TArray<float> ChunkData;
    {
        FScopeLock Lock(&AudioBufferMutex);
        const int64 Start = SileroChunkStart;
        const int64 End = FMath::Min(ChunkEnd, TotalSamplesWritten);
        if (End <= Start)
        {
            bSileroInFlight = false;
            return;
        }
        CopyRingBufferSegment_Locked(Start, End, ChunkData);
    }

    const float ChunkDuration = (float)ChunkData.Num() / (float)CAPTURE_SAMPLE_RATE;

    // Run the Silero processor on the BG thread
    FVADDecision Decision = AudioProcessor->ProcessAudioChunk(
        ChunkData.GetData(), ChunkData.Num(), ChunkEnd, ChunkDuration);

    // Advance the Silero read cursor
    SileroChunkStart = ChunkEnd;
    bSileroInFlight = false;

    // Handle decision
    if (Decision.Result == FVADDecision::EResult::SpeechOnset)
    {
        if (bEnableGameThreadDelegate)
        {
            EnqueueGTTask([this]()
            {
                OnVADStateChanged.Broadcast(true);
            });
        }
    }
    else if (Decision.Result == FVADDecision::EResult::DispatchSegment)
    {
        if (bEnableGameThreadDelegate)
        {
            EnqueueGTTask([this]()
            {
                OnVADStateChanged.Broadcast(false);
            });
        }

        DispatchSegment(Decision.SegmentStartSample, Decision.SegmentEndSample, true);
    }
}

// ---------------------------------------------------------------------------
// Ring buffer operations
// ---------------------------------------------------------------------------

void ULlamaAudioCaptureComponent::AppendToRingBuffer_Locked(const TArray<float>& Samples16k)
{
    const int32 Capacity = RingBuffer.Num();
    if (Capacity == 0)
    {
        return;
    }

    for (int32 i = 0; i < Samples16k.Num(); ++i)
    {
        const int32 WriteIndex = (int32)(TotalSamplesWritten % (int64)Capacity);
        RingBuffer[WriteIndex] = Samples16k[i];
        TotalSamplesWritten++;
    }
}

void ULlamaAudioCaptureComponent::CopyRingBufferSegment_Locked(int64 AbsStart, int64 AbsEnd,
                                                                 TArray<float>& OutSegment) const
{
    const int32 Capacity = RingBuffer.Num();
    if (Capacity == 0 || AbsEnd <= AbsStart)
    {
        OutSegment.Reset();
        return;
    }

    // Clamp to what is actually available in the ring buffer
    const int64 OldestAvailable = FMath::Max((int64)0, TotalSamplesWritten - (int64)Capacity);
    const int64 ClampedStart = FMath::Max(AbsStart, OldestAvailable);
    const int64 ClampedEnd = FMath::Min(AbsEnd, TotalSamplesWritten);

    if (ClampedEnd <= ClampedStart)
    {
        OutSegment.Reset();
        return;
    }

    const int32 SegmentLen = (int32)(ClampedEnd - ClampedStart);
    OutSegment.SetNumUninitialized(SegmentLen);

    for (int32 i = 0; i < SegmentLen; ++i)
    {
        const int32 RingIndex = (int32)((ClampedStart + i) % (int64)Capacity);
        OutSegment[i] = RingBuffer[RingIndex];
    }
}

// ---------------------------------------------------------------------------
// Segment dispatch
// ---------------------------------------------------------------------------

void ULlamaAudioCaptureComponent::DispatchSegment(int64 AbsStart, int64 AbsEnd, bool bVADTriggered)
{
    EnqueueBGTask([this, AbsStart, AbsEnd, bVADTriggered](int64 /*TaskId*/)
    {
        // Copy segment from ring buffer (short mutex hold)
        TArray<float> SegmentPCM;
        {
            FScopeLock Lock(&AudioBufferMutex);
            CopyRingBufferSegment_Locked(AbsStart, AbsEnd, SegmentPCM);
        }

        if (SegmentPCM.Num() == 0)
        {
            return;
        }

        // Build the segment struct
        FLlamaAudioSegment Segment;
        Segment.PCMSamples = MoveTemp(SegmentPCM);
        Segment.DurationSeconds = (float)Segment.PCMSamples.Num() / (float)CAPTURE_SAMPLE_RATE;
        Segment.bIsVADTriggered = bVADTriggered;

        // Dispatch to C++ consumers on BG thread (zero GT hop)
        {
            FScopeLock Lock(&ConsumersMutex);
            for (ILlamaAudioConsumer* Consumer : Consumers)
            {
                if (Consumer)
                {
                    Consumer->OnAudioSegment(Segment);
                }
            }
        }

        // Optionally dispatch to Blueprint delegate on game thread
        if (bEnableGameThreadDelegate)
        {
            // Copy segment for GT delivery (consumers may have moved data)
            FLlamaAudioSegment GTSegment;
            GTSegment.PCMSamples = Segment.PCMSamples;
            GTSegment.DurationSeconds = Segment.DurationSeconds;
            GTSegment.bIsVADTriggered = Segment.bIsVADTriggered;

            EnqueueGTTask([this, GTSegment = MoveTemp(GTSegment)]()
            {
                OnAudioSegmentReady.Broadcast(GTSegment);
            });
        }
    });
}

// ---------------------------------------------------------------------------
// Snapshot
// ---------------------------------------------------------------------------

FLlamaAudioSegment ULlamaAudioCaptureComponent::SnapshotAudio(float Seconds)
{
    FLlamaAudioSegment Segment;

    const int32 RequestedSamples = FMath::CeilToInt(FMath::Max(0.0f, Seconds) * CAPTURE_SAMPLE_RATE);

    FScopeLock Lock(&AudioBufferMutex);

    if (TotalSamplesWritten == 0 || RequestedSamples == 0)
    {
        return Segment;
    }

    const int64 AbsEnd = TotalSamplesWritten;
    const int64 AbsStart = FMath::Max((int64)0, AbsEnd - (int64)RequestedSamples);

    CopyRingBufferSegment_Locked(AbsStart, AbsEnd, Segment.PCMSamples);
    Segment.DurationSeconds = (float)Segment.PCMSamples.Num() / (float)CAPTURE_SAMPLE_RATE;
    Segment.bIsVADTriggered = false;

    return Segment;
}

// ---------------------------------------------------------------------------
// Consumer registration
// ---------------------------------------------------------------------------

void ULlamaAudioCaptureComponent::AddConsumer(ILlamaAudioConsumer* Consumer)
{
    if (!Consumer)
    {
        return;
    }

    FScopeLock Lock(&ConsumersMutex);
    Consumers.AddUnique(Consumer);
}

void ULlamaAudioCaptureComponent::RemoveConsumer(ILlamaAudioConsumer* Consumer)
{
    if (!Consumer)
    {
        return;
    }

    FScopeLock Lock(&ConsumersMutex);
    Consumers.Remove(Consumer);
}
