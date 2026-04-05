// Copyright 2025-current Getnamo.

#include "WhisperNative.h"
#include "Internal/WhisperInternal.h"
#include "LlamaUtility.h"       // FLlamaPaths, FLlamaString
#include "LlamaAudioCaptureComponent.h"
#include "LlamaAudioUtils.h"

// whisper.h is on the public include path (ThirdParty/WhisperCpp/include).
// We include it here to access WHISPER_SAMPLE_RATE (always 16000).
#include "whisper.h"

#include "Async/Async.h"
#include "Misc/FileHelper.h"
#include "HAL/PlatformProcess.h"
#include "Tickable.h"

// ---------------------------------------------------------------------------
// Construction / Destruction
// ---------------------------------------------------------------------------

FWhisperNative::FWhisperNative()
{
	Internal = new FWhisperInternal();

	Internal->OnTranscriptionResult = [this](const std::string& Text, bool bIsFinal)
	{
		// Called on BG thread -- forward to GT
		const FString UEText = FLlamaString::ToUE(Text);
		EnqueueGTTask([this, UEText, bIsFinal]()
		{
			if (OnTranscriptionResult)
			{
				OnTranscriptionResult(UEText, bIsFinal);
			}
		});
	};

	Internal->OnError = [this](const std::string& ErrorMessage)
	{
		// Called on BG thread -- forward to GT
		const FString UEError = FLlamaString::ToUE(ErrorMessage);
		EnqueueGTTask([this, UEError]()
		{
			if (OnError)
			{
				OnError(UEError);
			}
		});
	};
}

FWhisperNative::~FWhisperNative()
{
	// Stop microphone capture if active (unsubscribes and cleans up internal source)
	if (bMicCaptureActive)
	{
		StopMicrophoneCapture();
	}

	// Stop background thread
	bThreadShouldRun = false;

	// Wait for the thread to finish (spin-wait; BG thread will exit on next idle sleep)
	const double StartTime = FPlatformTime::Seconds();
	while (bThreadIsActive && (FPlatformTime::Seconds() - StartTime) < 3.0)
	{
		FPlatformProcess::Sleep(0.01f);
	}

	RemoveTicker();

	delete Internal;
	Internal = nullptr;
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

void FWhisperNative::SetModelParams(const FWhisperModelParams& Params)
{
	ModelParams = Params;
}

// ---------------------------------------------------------------------------
// Threading helpers (mirrors FLlamaNative)
// ---------------------------------------------------------------------------

void FWhisperNative::StartWhisperThread()
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
			FPlatformProcess::Sleep(ThreadIdleSleepDuration);
		}

		bThreadIsActive = false;
	});
}

int64 FWhisperNative::GetNextTaskId()
{
	return TaskIdCounter.Increment();
}

void FWhisperNative::EnqueueBGTask(TFunction<void(int64)> TaskFunction)
{
	if (!bThreadIsActive)
	{
		StartWhisperThread();
	}

	FLLMThreadTask Task;
	Task.TaskId      = GetNextTaskId();
	Task.TaskFunction = TaskFunction;
	BackgroundTasks.Enqueue(Task);
}

void FWhisperNative::EnqueueGTTask(TFunction<void()> TaskFunction, int64 LinkedTaskId)
{
	FLLMThreadTask Task;
	Task.TaskId = (LinkedTaskId == -1) ? GetNextTaskId() : LinkedTaskId;
	Task.TaskFunction = [TaskFunction](int64) { TaskFunction(); };
	GameThreadTasks.Enqueue(Task);
}

// ---------------------------------------------------------------------------
// Model control
// ---------------------------------------------------------------------------

bool FWhisperNative::IsModelLoaded() const
{
	return bModelLoadInitiated && Internal && Internal->IsModelLoaded();
}

void FWhisperNative::LoadModel(bool bForceReload,
	TFunction<void(const FString&, int32)> Callback)
{
	if (Internal->IsModelLoaded() && !bForceReload)
	{
		if (Callback)
		{
			Callback(ModelParams.PathToModel, 0);
		}
		return;
	}

	bModelLoadInitiated = true;

	const FWhisperModelParams ParamsAtLoad = ModelParams;
	const FString             FullPath     = FLlamaPaths::ParsePathIntoFullPath(ParamsAtLoad.PathToModel);

	EnqueueBGTask([this, ParamsAtLoad, FullPath, Callback](int64 TaskId)
	{
		const bool bSuccess = Internal->LoadModel(
			FLlamaString::ToStd(FullPath),
			ParamsAtLoad.bUseGPU,
			ParamsAtLoad.Threads);

		EnqueueGTTask([this, bSuccess, FullPath, Callback]()
		{
			if (bSuccess)
			{
				if (OnModelLoaded)
				{
					OnModelLoaded(FullPath);
				}
				if (Callback)
				{
					Callback(FullPath, 0);
				}
			}
			else
			{
				if (Callback)
				{
					Callback(FullPath, -1);
				}
			}
		}, TaskId);
	});
}

void FWhisperNative::UnloadModel(TFunction<void(int32)> Callback)
{
	if (bMicCaptureActive)
	{
		StopMicrophoneCapture();
	}

	EnqueueBGTask([this, Callback](int64 TaskId)
	{
		Internal->UnloadModel();
		bModelLoadInitiated = false;

		EnqueueGTTask([Callback]()
		{
			if (Callback)
			{
				Callback(0);
			}
		}, TaskId);
	});
}

// ---------------------------------------------------------------------------
// One-shot transcription
// ---------------------------------------------------------------------------

void FWhisperNative::TranscribeAudioData(TArray<float> PCMSamples, int32 SampleRate)
{
	const FWhisperModelParams ParamsAtCall = ModelParams;

	EnqueueBGTask([this, PCMSamples = MoveTemp(PCMSamples), SampleRate, ParamsAtCall](int64 TaskId)
	{
		// Resample to 16 kHz if needed
		TArray<float> Samples16k;
		if (SampleRate != WHISPER_SAMPLE_RATE)
		{
			ULlamaAudioUtils::ResampleLinear(PCMSamples.GetData(), PCMSamples.Num(), SampleRate,
			               Samples16k, WHISPER_SAMPLE_RATE);
		}
		else
		{
			Samples16k = PCMSamples;
		}

		const bool bBeam = (ParamsAtCall.SamplingStrategy == EWhisperSamplingStrategy::BeamSearch);

		bIsTranscribing = true;
		EnqueueGTTask([this]() { if (OnTranscribingStateChanged) OnTranscribingStateChanged(true); }, TaskId);

		Internal->TranscribeAudio(
			Samples16k.GetData(), Samples16k.Num(),
			FLlamaString::ToStd(ParamsAtCall.Language),
			ParamsAtCall.bTranslate,
			ParamsAtCall.MaxContext,
			ParamsAtCall.BestOf,
			ParamsAtCall.BeamSize,
			bBeam);

		bIsTranscribing = false;
		EnqueueGTTask([this]() { if (OnTranscribingStateChanged) OnTranscribingStateChanged(false); });
	});
}

void FWhisperNative::TranscribeWaveFile(const FString& FilePath)
{
	const FString FullPath = FLlamaPaths::ParsePathIntoFullPath(FilePath);

	EnqueueBGTask([this, FullPath](int64 TaskId)
	{
		TArray<float> Samples;
		int32 FileSampleRate = 0;

		if (!LoadWavFile(FullPath, Samples, FileSampleRate))
		{
			EnqueueGTTask([this, FullPath]()
			{
				if (OnError)
				{
					OnError(FString::Printf(TEXT("Failed to load WAV file: %s"), *FullPath));
				}
			}, TaskId);
			return;
		}

		// Resample to 16 kHz if needed
		TArray<float> Samples16k;
		if (FileSampleRate != WHISPER_SAMPLE_RATE)
		{
			ULlamaAudioUtils::ResampleLinear(Samples.GetData(), Samples.Num(), FileSampleRate,
			               Samples16k, WHISPER_SAMPLE_RATE);
		}
		else
		{
			Samples16k = MoveTemp(Samples);
		}

		const FWhisperModelParams Params = ModelParams; // copy on BG
		const bool bBeam = (Params.SamplingStrategy == EWhisperSamplingStrategy::BeamSearch);

		bIsTranscribing = true;
		EnqueueGTTask([this]() { if (OnTranscribingStateChanged) OnTranscribingStateChanged(true); }, TaskId);

		Internal->TranscribeAudio(
			Samples16k.GetData(), Samples16k.Num(),
			FLlamaString::ToStd(Params.Language),
			Params.bTranslate,
			Params.MaxContext,
			Params.BestOf,
			Params.BeamSize,
			bBeam);

		bIsTranscribing = false;
		EnqueueGTTask([this]() { if (OnTranscribingStateChanged) OnTranscribingStateChanged(false); });
	});
}

// ---------------------------------------------------------------------------
// Microphone capture -- delegates to ULlamaAudioCaptureComponent
// ---------------------------------------------------------------------------

void FWhisperNative::StartMicrophoneCapture()
{
	if (bMicCaptureActive)
	{
		return;
	}

	ULlamaAudioCaptureComponent* Source = GetActiveAudioSource();
	if (!Source)
	{
		// Create internal audio source. ULlamaAudioCaptureComponent is a UActorComponent;
		// when used standalone (no owning actor), we create it with NewObject and root it.
		// VAD and capture settings use the component's own UPROPERTY defaults.
		InternalAudioSource = NewObject<ULlamaAudioCaptureComponent>();
		InternalAudioSource->AddToRoot(); // prevent GC

		Source = InternalAudioSource;
	}

	Source->AddConsumer(this);
	Source->StartCapture();
	bMicCaptureActive = true;
}

void FWhisperNative::StopMicrophoneCapture()
{
	if (!bMicCaptureActive)
	{
		return;
	}

	ULlamaAudioCaptureComponent* Source = GetActiveAudioSource();
	if (Source)
	{
		Source->RemoveConsumer(this);
		// Only stop the capture if we own it (internal source)
		if (Source == InternalAudioSource)
		{
			Source->StopCapture();
		}
	}

	bMicCaptureActive = false;

	// Clean up internal source
	if (InternalAudioSource)
	{
		InternalAudioSource->RemoveFromRoot();
		InternalAudioSource->ConditionalBeginDestroy();
		InternalAudioSource = nullptr;
	}
}

bool FWhisperNative::IsMicrophoneCaptureActive() const
{
	return bMicCaptureActive;
}

void FWhisperNative::SetMicrophoneMuted(bool bMuted)
{
	bMicMuted = bMuted;
	ULlamaAudioCaptureComponent* Source = GetActiveAudioSource();
	if (Source)
	{
		Source->SetMuted(bMuted);
	}
}

bool FWhisperNative::IsMicrophoneMuted() const
{
	return bMicMuted;
}

// ---------------------------------------------------------------------------
// External audio source management
// ---------------------------------------------------------------------------

void FWhisperNative::SetExternalAudioSource(ULlamaAudioCaptureComponent* Source)
{
	// If currently capturing, stop first to unsubscribe from the current source
	if (bMicCaptureActive)
	{
		StopMicrophoneCapture();
	}
	ExternalAudioSource = Source;
}

ULlamaAudioCaptureComponent* FWhisperNative::GetActiveAudioSource() const
{
	return ExternalAudioSource ? ExternalAudioSource : InternalAudioSource;
}

// ---------------------------------------------------------------------------
// ILlamaAudioConsumer implementation
// ---------------------------------------------------------------------------

void FWhisperNative::OnAudioSegment(const FLlamaAudioSegment& Segment)
{
	// Called on the capture component's BG thread -- enqueue transcription directly
	const FWhisperModelParams ParamsAtDispatch = ModelParams;
	TArray<float> PCMCopy = Segment.PCMSamples;

	EnqueueBGTask([this, PCMCopy = MoveTemp(PCMCopy), ParamsAtDispatch](int64 TaskId)
	{
		if (PCMCopy.IsEmpty())
		{
			return;
		}

		const bool bBeam = (ParamsAtDispatch.SamplingStrategy == EWhisperSamplingStrategy::BeamSearch);

		bIsTranscribing = true;
		EnqueueGTTask([this]() { if (OnTranscribingStateChanged) OnTranscribingStateChanged(true); }, TaskId);

		Internal->TranscribeAudio(
			PCMCopy.GetData(), PCMCopy.Num(),
			FLlamaString::ToStd(ParamsAtDispatch.Language),
			ParamsAtDispatch.bTranslate,
			ParamsAtDispatch.MaxContext,
			ParamsAtDispatch.BestOf,
			ParamsAtDispatch.BeamSize,
			bBeam);

		bIsTranscribing = false;
		EnqueueGTTask([this]() { if (OnTranscribingStateChanged) OnTranscribingStateChanged(false); });
	});
}

// ---------------------------------------------------------------------------
// WAV file loading
// ---------------------------------------------------------------------------

bool FWhisperNative::LoadWavFile(const FString& FilePath, TArray<float>& OutSamples, int32& OutSampleRate)
{
	TArray<uint8> RawBytes;
	if (!FFileHelper::LoadFileToArray(RawBytes, *FilePath))
	{
		return false;
	}

	const uint8* Data = RawBytes.GetData();
	const int32  Size = RawBytes.Num();

	// Minimum RIFF header size is 44 bytes
	if (Size < 44)
	{
		return false;
	}

	// Validate RIFF and WAVE identifiers
	if (FMemory::Memcmp(Data,     "RIFF", 4) != 0 ||
	    FMemory::Memcmp(Data + 8, "WAVE", 4) != 0)
	{
		return false;
	}

	// Parse fmt chunk (assume standard 44-byte header, no extra chunks before data)
	const uint16 AudioFormat   = *reinterpret_cast<const uint16*>(Data + 20);
	const uint16 NumChannels   = *reinterpret_cast<const uint16*>(Data + 22);
	const uint32 SampleRate    = *reinterpret_cast<const uint32*>(Data + 24);
	const uint16 BitsPerSample = *reinterpret_cast<const uint16*>(Data + 34);

	if (AudioFormat != 1 || BitsPerSample != 16)
	{
		// Only 16-bit PCM WAV supported
		return false;
	}

	if (NumChannels == 0 || SampleRate == 0)
	{
		return false;
	}

	// Find the "data" sub-chunk (search past the fmt chunk for robustness)
	int32 DataOffset = -1;
	uint32 DataSize  = 0;

	for (int32 Offset = 12; Offset + 8 <= Size; )
	{
		if (FMemory::Memcmp(Data + Offset, "data", 4) == 0)
		{
			DataSize   = *reinterpret_cast<const uint32*>(Data + Offset + 4);
			DataOffset = Offset + 8;
			break;
		}
		// Skip this chunk
		const uint32 ChunkSize = *reinterpret_cast<const uint32*>(Data + Offset + 4);
		Offset += 8 + static_cast<int32>(ChunkSize);
	}

	if (DataOffset < 0 || DataSize == 0)
	{
		return false;
	}

	const int32 NumSamples = static_cast<int32>(DataSize) / (BitsPerSample / 8);
	const int32 NumFrames  = NumSamples / NumChannels;

	// Convert int16 samples to float32
	const int16* SampleData = reinterpret_cast<const int16*>(Data + DataOffset);

	TArray<float> AllChannelSamples;
	AllChannelSamples.SetNumUninitialized(NumSamples);

	constexpr float kInt16ToFloat = 1.0f / 32768.0f;
	for (int32 i = 0; i < NumSamples; ++i)
	{
		AllChannelSamples[i] = static_cast<float>(SampleData[i]) * kInt16ToFloat;
	}

	// Downmix to mono if needed
	if (NumChannels > 1)
	{
		ULlamaAudioUtils::DownmixToMono(AllChannelSamples.GetData(), NumFrames, NumChannels, OutSamples);
	}
	else
	{
		OutSamples = MoveTemp(AllChannelSamples);
	}

	OutSampleRate = static_cast<int32>(SampleRate);
	return true;
}

// ---------------------------------------------------------------------------
// Game thread tick
// ---------------------------------------------------------------------------

void FWhisperNative::OnGameThreadTick(float /*DeltaTime*/)
{
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

void FWhisperNative::AddTicker()
{
	TickDelegateHandle = FTSTicker::GetCoreTicker().AddTicker(
		FTickerDelegate::CreateLambda([this](float DeltaTime)
		{
			OnGameThreadTick(DeltaTime);
			return true;
		}));
}

void FWhisperNative::RemoveTicker()
{
	if (IsNativeTickerActive())
	{
		FTSTicker::GetCoreTicker().RemoveTicker(TickDelegateHandle);
		TickDelegateHandle = nullptr;
	}
}

bool FWhisperNative::IsNativeTickerActive() const
{
	return TickDelegateHandle.IsValid();
}
