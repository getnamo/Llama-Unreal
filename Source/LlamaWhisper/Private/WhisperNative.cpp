// Copyright 2025-current Getnamo.

#include "WhisperNative.h"
#include "Internal/WhisperInternal.h"
#include "LlamaUtility.h"       // FLlamaPaths, FLlamaString

// whisper.h is on the public include path (ThirdParty/WhisperCpp/include).
// We include it here to access WHISPER_SAMPLE_RATE (always 16000).
#include "whisper.h"

#include "Async/Async.h"
#include "Misc/FileHelper.h"
#include "HAL/PlatformProcess.h"
#include "Tickable.h"

// AudioCapture
#include "AudioCaptureCore.h"

// ---------------------------------------------------------------------------
// Construction / Destruction
// ---------------------------------------------------------------------------

FWhisperNative::FWhisperNative()
{
	Internal = new FWhisperInternal();

	Internal->OnTranscriptionResult = [this](const std::string& Text, bool bIsFinal)
	{
		// Called on BG thread — forward to GT
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
		// Called on BG thread — forward to GT
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
	// Stop microphone capture if active
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

	// Free Silero VAD context if loaded (BG thread is stopped by this point)
	if (SileroContext)
	{
		whisper_vad_free(SileroContext);
		SileroContext = nullptr;
	}

	delete Internal;
	Internal = nullptr;

	if (AudioCapture)
	{
		delete AudioCapture;
		AudioCapture = nullptr;
	}
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

void FWhisperNative::SetModelParams(const FWhisperModelParams& Params)
{
	ModelParams = Params;
}

void FWhisperNative::SetStreamParams(const FWhisperStreamParams& Params)
{
	StreamParams = Params;
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
	// This is a heuristic — bModelLoadInitiated is set on GT, actual load is BG.
	// The component's ModelState.bModelLoaded is the authoritative GT state.
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
				// Auto-load Silero VAD model when whisper model loads in Silero mode
				if (StreamParams.VADMode == EWhisperVADMode::Silero)
				{
					LoadVADModel();
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
// Silero VAD model control
// ---------------------------------------------------------------------------

bool FWhisperNative::IsVADModelLoaded() const
{
	return SileroContext != nullptr;
}

void FWhisperNative::LoadVADModel(TFunction<void(const FString&, int32)> Callback)
{
	if (SileroContext)
	{
		// Already loaded — report success without reloading
		if (Callback) Callback(StreamParams.PathToVADModel, 0);
		return;
	}

	const FString FullPath = FLlamaPaths::ParsePathIntoFullPath(StreamParams.PathToVADModel);
	const FWhisperModelParams ParamsAtLoad = ModelParams;

	EnqueueBGTask([this, FullPath, ParamsAtLoad, Callback](int64 TaskId)
	{
		whisper_vad_context_params VadCtxParams = whisper_vad_default_context_params();
		VadCtxParams.n_threads  = 1; // Silero is a tiny model; 1 thread avoids contention with game thread
		VadCtxParams.use_gpu    = ParamsAtLoad.bUseGPU;

		SileroContext = whisper_vad_init_from_file_with_params(
			TCHAR_TO_UTF8(*FullPath), VadCtxParams);

		const bool bSuccess = (SileroContext != nullptr);

		EnqueueGTTask([this, bSuccess, FullPath, Callback]()
		{
			if (OnVADModelLoaded)
			{
				OnVADModelLoaded(FullPath, bSuccess);
			}
			if (Callback)
			{
				Callback(FullPath, bSuccess ? 0 : -1);
			}
			if (!bSuccess && OnError)
			{
				OnError(FString::Printf(TEXT("FWhisperNative: Failed to load Silero VAD model: %s"), *FullPath));
			}
		}, TaskId);
	});
}

void FWhisperNative::UnloadVADModel(TFunction<void(int32)> Callback)
{
	EnqueueBGTask([this, Callback](int64 TaskId)
	{
		if (SileroContext)
		{
			whisper_vad_free(SileroContext);
			SileroContext    = nullptr;
			SileroChunkStart = 0;
		}

		EnqueueGTTask([Callback]()
		{
			if (Callback) Callback(0);
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
			ResampleLinear(PCMSamples.GetData(), PCMSamples.Num(), SampleRate,
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
			ResampleLinear(Samples.GetData(), Samples.Num(), FileSampleRate,
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
// Microphone capture
// ---------------------------------------------------------------------------

void FWhisperNative::StartMicrophoneCapture()
{
	if (bMicCaptureActive)
	{
		return;
	}

	// Allocate ring buffer
	{
		FScopeLock Lock(&AudioBufferMutex);

		const int32 Capacity = StreamParams.RingBufferCapacitySamples;
		RingBuffer.SetNumZeroed(Capacity);
		TotalSamplesWritten = 0;

		// Pre-roll buffer: PreRollSec * 16000 samples
		const int32 PreRollSamples = FMath::Max(1,
			FMath::RoundToInt(StreamParams.VADPreRollSec * WHISPER_SAMPLE_RATE));
		PreRollBuffer.SetNumZeroed(PreRollSamples);
		PreRollWritePos = 0;

		// Reset VAD state
		bVADSpeechActive     = false;
		VADSilenceDuration   = 0.0f;
		VADSegmentDuration   = 0.0f;
		VADSpeechStartSample = 0;   // no-VAD and Silero modes accumulate from sample 0
	}

	// Reset Silero state — safe here because mic is not yet active so no BG tasks running
	SileroChunkStart = 0;
	SileroResidual.Reset();
	if (SileroContext)
	{
		whisper_vad_reset_state(SileroContext);
	}

	// Create AudioCapture object
	if (!AudioCapture)
	{
		AudioCapture = new Audio::FAudioCapture();
	}

	// Open the stream using a lambda callback
	Audio::FAudioCaptureDeviceParams DeviceParams;
	// Use system default sample rate
	DeviceParams.SampleRate = Audio::InvalidDeviceSampleRate;

	// Get the actual device info to know the sample rate we'll receive
	Audio::FCaptureDeviceInfo DeviceInfo;
	AudioCapture->GetCaptureDeviceInfo(DeviceInfo);
	if (DeviceInfo.PreferredSampleRate > 0)
	{
		CaptureDeviceSampleRate = DeviceInfo.PreferredSampleRate;
	}
	CaptureDeviceChannels = DeviceInfo.InputChannels > 0 ? DeviceInfo.InputChannels : 1;

	const bool bOpened = AudioCapture->OpenAudioCaptureStream(
		DeviceParams,
		[this](const void* InAudio, int32 NumFrames, int32 NumChannels,
		       int32 SampleRate, double /*StreamTime*/, bool /*bOverFlow*/)
		{
			// Update cached capture info on first call
			if (SampleRate > 0)
			{
				CaptureDeviceSampleRate = SampleRate;
			}
			if (NumChannels > 0)
			{
				CaptureDeviceChannels = NumChannels;
			}
			HandleAudioCaptureCallback(
				reinterpret_cast<const float*>(InAudio), NumFrames, NumChannels, SampleRate);
		},
		/*NumFramesDesired=*/1024);

	if (!bOpened)
	{
		EnqueueGTTask([this]()
		{
			if (OnError)
			{
				OnError(TEXT("FWhisperNative: Failed to open audio capture stream."));
			}
		});
		return;
	}

	AudioCapture->StartStream();
	bMicCaptureActive = true;

	EnqueueGTTask([this]()
	{
		if (OnVADStateChanged)
		{
			OnVADStateChanged(false); // Start in silence state
		}
	});
}

void FWhisperNative::StopMicrophoneCapture()
{
	if (!bMicCaptureActive)
	{
		return;
	}

	if (AudioCapture)
	{
		AudioCapture->StopStream();
		AudioCapture->CloseStream();
	}

	bMicCaptureActive = false;

	// Prevent any queued BG Silero tasks from advancing the cursor past the flush window
	SileroChunkStart = TotalSamplesWritten;

	// Flush any remaining buffered audio as a final segment.
	// VADSpeechStartSample is advanced after every completed dispatch (VAD offset, force chunk,
	// no-VAD chunk), so this window only contains truly unprocessed audio.  This covers:
	//   VAD mode  — mic stopped mid-speech (bVADSpeechActive true) or during post-speech silence
	//   No-VAD    — whatever accumulated since the last forced chunk (or since mic start)
	// Extract dispatch params under the lock, then call dispatch outside to avoid deadlock.
	int64 PendingSegStart = -1;
	int64 PendingSegEnd   = -1;
	{
		FScopeLock Lock(&AudioBufferMutex);
		if (TotalSamplesWritten > VADSpeechStartSample)
		{
			PendingSegStart = VADSpeechStartSample;
			PendingSegEnd   = TotalSamplesWritten;
		}
		bVADSpeechActive = false;
	}

	if (PendingSegStart >= 0 && PendingSegEnd > PendingSegStart)
	{
		DispatchSegmentForTranscription(PendingSegStart, PendingSegEnd);
	}

	EnqueueGTTask([this]()
	{
		if (OnVADStateChanged)
		{
			OnVADStateChanged(false);
		}
	});
}

bool FWhisperNative::IsMicrophoneCaptureActive() const
{
	return bMicCaptureActive;
}

// ---------------------------------------------------------------------------
// Audio capture callback (runs on audio HW thread)
// ---------------------------------------------------------------------------

void FWhisperNative::SetMicrophoneMuted(bool bMuted)
{
	bMicMuted = bMuted;
}

bool FWhisperNative::IsMicrophoneMuted() const
{
	return bMicMuted;
}

void FWhisperNative::HandleAudioCaptureCallback(const float* InAudio, int32 NumFrames,
                                                 int32 NumChannels, int32 SampleRate)
{
	if (!InAudio || NumFrames <= 0 || bMicMuted)
	{
		return;
	}

	const int32 ActualChannels = FMath::Max(1, NumChannels);
	const int32 ActualRate     = (SampleRate > 0) ? SampleRate : CaptureDeviceSampleRate;

	// Step 1: Downmix to mono
	TArray<float> MonoSamples;
	DownmixToMono(InAudio, NumFrames, ActualChannels, MonoSamples);

	// Step 2: Resample to 16 kHz
	TArray<float> Samples16k;
	if (ActualRate != WHISPER_SAMPLE_RATE)
	{
		ResampleLinear(MonoSamples.GetData(), MonoSamples.Num(), ActualRate,
		               Samples16k, WHISPER_SAMPLE_RATE);
	}
	else
	{
		Samples16k = MoveTemp(MonoSamples);
	}

	if (Samples16k.IsEmpty())
	{
		return;
	}

	// Step 3: Append to ring buffer; run energy VAD inline or note position for Silero/Disabled
	int64 DispatchStart      = -1;
	int64 DispatchEnd        = -1;
	bool  bVADOnset          = false;
	bool  bVADOffset         = false;
	int64 SileroChunkEnd     = -1; // set in Silero branch; BG task uses this

	{
		FScopeLock Lock(&AudioBufferMutex);

		// Update pre-roll circular buffer (always runs regardless of VAD mode)
		const int32 PreRollCap = PreRollBuffer.Num();
		for (int32 i = 0; i < Samples16k.Num() && PreRollCap > 0; ++i)
		{
			PreRollBuffer[PreRollWritePos % PreRollCap] = Samples16k[i];
			PreRollWritePos++;
		}

		AppendToRingBuffer_Locked(Samples16k);

		if (StreamParams.VADMode == EWhisperVADMode::EnergyBased)
		{
			// Inline RMS VAD — fast enough to run on the audio HW thread
			const float RMS = ComputeRMS(Samples16k.GetData(), Samples16k.Num());
			const float ChunkDuration = static_cast<float>(Samples16k.Num()) / WHISPER_SAMPLE_RATE;

			if (!bVADSpeechActive)
			{
				if (RMS > StreamParams.VADThreshold)
				{
					// Voice onset: include pre-roll
					bVADSpeechActive = true;
					bVADOnset        = true;

					const int32 PreRollSamples = FMath::Min(
						PreRollBuffer.Num(),
						FMath::RoundToInt(StreamParams.VADPreRollSec * WHISPER_SAMPLE_RATE));

					VADSpeechStartSample = FMath::Max(
						TotalSamplesWritten - Samples16k.Num() - PreRollSamples,
						TotalSamplesWritten - RingBuffer.Num()); // clamp to ring buffer range

					VADSilenceDuration = 0.0f;
					VADSegmentDuration = ChunkDuration;
				}
			}
			else
			{
				VADSegmentDuration += ChunkDuration;

				if (RMS < StreamParams.VADThreshold)
				{
					VADSilenceDuration += ChunkDuration;

					if (VADSilenceDuration >= StreamParams.VADHoldTimeSec)
					{
						// Voice offset: dispatch the segment
						bVADSpeechActive     = false;
						bVADOffset           = true;
						DispatchStart        = VADSpeechStartSample;
						DispatchEnd          = TotalSamplesWritten;
						// Advance cursor so StopMicrophoneCapture won't re-dispatch this audio
						VADSpeechStartSample = TotalSamplesWritten;
						VADSilenceDuration   = 0.0f;
						VADSegmentDuration   = 0.0f;
					}
				}
				else
				{
					VADSilenceDuration = 0.0f;

					// Safety valve: force dispatch if segment is too long
					if (VADSegmentDuration >= StreamParams.MaxSpeechSegmentSec)
					{
						DispatchStart        = VADSpeechStartSample;
						DispatchEnd          = TotalSamplesWritten;
						VADSpeechStartSample = TotalSamplesWritten;
						VADSegmentDuration   = 0.0f;
					}
				}
			}
		}
		else if (StreamParams.VADMode == EWhisperVADMode::Silero)
		{
			// Neural VAD must NOT run on the audio HW thread (up to ~15ms per chunk).
			// Just capture the write cursor; the BG thread runs ProcessSileroVADChunk().
			SileroChunkEnd = TotalSamplesWritten;
		}
		else // EWhisperVADMode::Disabled
		{
			// Accumulate from mic start to mic stop.
			// Force-dispatch at MaxSpeechSegmentSec with optional overlap.
			const float ChunkDuration = static_cast<float>(Samples16k.Num()) / WHISPER_SAMPLE_RATE;
			VADSegmentDuration += ChunkDuration;

			if (VADSegmentDuration >= StreamParams.MaxSpeechSegmentSec)
			{
				DispatchStart = VADSpeechStartSample;
				DispatchEnd   = TotalSamplesWritten;

				const int64 OverlapSamples = static_cast<int64>(
					StreamParams.NonVADOverlapSec * WHISPER_SAMPLE_RATE);
				VADSpeechStartSample = FMath::Max(0LL, TotalSamplesWritten - OverlapSamples);
				VADSegmentDuration   = StreamParams.NonVADOverlapSec;
			}
		}
	} // end of mutex scope

	// Step 4: Fire callbacks and dispatch tasks OUTSIDE the mutex

	if (SileroChunkEnd > 0 && !bSileroInFlight)
	{
		// Enqueue a lightweight BG task — all VAD logic runs there.
		// Guard prevents flooding the queue while a previous chunk is still processing.
		bSileroInFlight = true;
		EnqueueBGTask([this, SileroChunkEnd](int64) { ProcessSileroVADChunk(SileroChunkEnd); });
	}

	if (bVADOnset)
	{
		EnqueueGTTask([this]() { if (OnVADStateChanged) OnVADStateChanged(true); });
	}
	if (bVADOffset)
	{
		EnqueueGTTask([this]() { if (OnVADStateChanged) OnVADStateChanged(false); });
	}
	if (DispatchStart >= 0 && DispatchEnd > DispatchStart)
	{
		DispatchSegmentForTranscription(DispatchStart, DispatchEnd);
	}
}

// ---------------------------------------------------------------------------
// Silero VAD chunk processor (BG thread only)
// ---------------------------------------------------------------------------

void FWhisperNative::ProcessSileroVADChunk(int64 ChunkEnd)
{
	if (!SileroContext || ChunkEnd <= SileroChunkStart)
	{
		SileroChunkStart = ChunkEnd;
		bSileroInFlight = false;
		return;
	}

	// Step 1: Short mutex hold — copy new audio and snapshot VAD state
	TArray<float> Chunk;
	{
		FScopeLock Lock(&AudioBufferMutex);
		CopyRingBufferSegment_Locked(SileroChunkStart, ChunkEnd, Chunk);
	}

	if (Chunk.IsEmpty())
	{
		SileroChunkStart = ChunkEnd;
		bSileroInFlight = false;
		return;
	}

	// Step 2: Run Silero inference in streaming mode — does NOT reset LSTM state.
	// Accumulate samples into n_window-sized (512) frames and run one inference per frame.
	const int WindowSize = whisper_vad_n_window(SileroContext);
	bool bSpeechDetected = false;
	const float SileroThreshold = StreamParams.SileroThreshold;

	// Append chunk to residual buffer, then process complete windows
	SileroResidual.Append(Chunk.GetData(), Chunk.Num());

	int WindowsProcessed = 0;
	while (SileroResidual.Num() >= WindowSize)
	{
		const float Prob = whisper_vad_detect_speech_streaming(
			SileroContext, SileroResidual.GetData());
		if (Prob >= SileroThreshold)
		{
			bSpeechDetected = true;
		}

		SileroResidual.RemoveAt(0, WindowSize, EAllowShrinking::No);
		WindowsProcessed++;
	}

	// If no complete window was processed, skip the state machine — we have no new
	// information. Running it with bSpeechDetected=false would incorrectly accumulate
	// silence duration between windows.
	if (WindowsProcessed == 0)
	{
		SileroChunkStart = ChunkEnd;
		bSileroInFlight = false;
		return;
	}

	// Use actual processed sample count (complete Silero windows), not raw chunk size.
	// The residual buffer means we may process more or fewer samples than the chunk contains.
	const float ChunkDuration = static_cast<float>(WindowsProcessed * WindowSize) / WHISPER_SAMPLE_RATE;

	// Step 3: Onset/offset state machine — short mutex hold to read/write shared VAD state
	int64 DispatchStart = -1;
	int64 DispatchEnd   = -1;
	bool  bVADOnset     = false;
	bool  bVADOffset    = false;

	{
		FScopeLock Lock(&AudioBufferMutex);

		if (!bVADSpeechActive)
		{
			if (bSpeechDetected)
			{
				bVADSpeechActive = true;
				bVADOnset        = true;

				const int32 PreRollSamples = FMath::Min(
					PreRollBuffer.Num(),
					FMath::RoundToInt(StreamParams.VADPreRollSec * WHISPER_SAMPLE_RATE));

				VADSpeechStartSample = FMath::Max(
					SileroChunkStart - PreRollSamples,
					TotalSamplesWritten - RingBuffer.Num()); // clamp to ring buffer range

				VADSilenceDuration = 0.0f;
				VADSegmentDuration = ChunkDuration;
			}
		}
		else
		{
			VADSegmentDuration += ChunkDuration;

			if (!bSpeechDetected)
			{
				VADSilenceDuration += ChunkDuration;

				if (VADSilenceDuration >= StreamParams.SileroHoldTimeSec)
				{
					// Voice offset: dispatch segment
					bVADSpeechActive     = false;
					bVADOffset           = true;
					DispatchStart        = VADSpeechStartSample;
					DispatchEnd          = ChunkEnd;
					VADSpeechStartSample = ChunkEnd;
					VADSilenceDuration   = 0.0f;
					VADSegmentDuration   = 0.0f;
				}
			}
			else
			{
				VADSilenceDuration = 0.0f;

				// Safety valve: force dispatch if segment is too long
				if (VADSegmentDuration >= StreamParams.MaxSpeechSegmentSec)
				{
					DispatchStart        = VADSpeechStartSample;
					DispatchEnd          = ChunkEnd;
					VADSpeechStartSample = ChunkEnd;
					VADSegmentDuration   = 0.0f;
				}
			}
		}
	} // end of mutex scope

	// Step 4: Advance BG read cursor (BG thread only — no mutex needed)
	SileroChunkStart = ChunkEnd;
	bSileroInFlight = false;

	// Step 5: Fire GT callbacks and dispatch outside mutex
	if (bVADOnset)
	{
		EnqueueGTTask([this]() { if (OnVADStateChanged) OnVADStateChanged(true); });
	}
	if (bVADOffset)
	{
		EnqueueGTTask([this]() { if (OnVADStateChanged) OnVADStateChanged(false); });
	}
	if (DispatchStart >= 0 && DispatchEnd > DispatchStart)
	{
		DispatchSegmentForTranscription(DispatchStart, DispatchEnd);
	}
}

// ---------------------------------------------------------------------------
// Segment dispatch
// ---------------------------------------------------------------------------

void FWhisperNative::DispatchSegmentForTranscription(int64 AbsStart, int64 AbsEnd)
{
	const FWhisperModelParams ParamsAtDispatch = ModelParams;

	EnqueueBGTask([this, AbsStart, AbsEnd, ParamsAtDispatch](int64 TaskId)
	{
		// Copy the segment out of the ring buffer (short mutex hold)
		TArray<float> Segment;
		{
			FScopeLock Lock(&AudioBufferMutex);
			CopyRingBufferSegment_Locked(AbsStart, AbsEnd, Segment);
		}

		if (Segment.IsEmpty())
		{
			return;
		}

		const bool bBeam = (ParamsAtDispatch.SamplingStrategy == EWhisperSamplingStrategy::BeamSearch);

		bIsTranscribing = true;
		EnqueueGTTask([this]() { if (OnTranscribingStateChanged) OnTranscribingStateChanged(true); }, TaskId);

		Internal->TranscribeAudio(
			Segment.GetData(), Segment.Num(),
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
// Ring buffer helpers
// ---------------------------------------------------------------------------

void FWhisperNative::AppendToRingBuffer_Locked(const TArray<float>& Samples16k)
{
	const int32 Capacity = RingBuffer.Num();
	if (Capacity <= 0)
	{
		return;
	}

	for (int32 i = 0; i < Samples16k.Num(); ++i)
	{
		const int32 Idx     = static_cast<int32>(TotalSamplesWritten % Capacity);
		RingBuffer[Idx]     = Samples16k[i];
		TotalSamplesWritten++;
	}
}

void FWhisperNative::CopyRingBufferSegment_Locked(int64 AbsStart, int64 AbsEnd,
                                                   TArray<float>& OutSegment) const
{
	const int32 Capacity = RingBuffer.Num();
	if (Capacity <= 0 || AbsEnd <= AbsStart)
	{
		return;
	}

	// Clamp to what's still in the ring buffer
	const int64 OldestSample = TotalSamplesWritten - Capacity;
	const int64 ClampedStart = FMath::Max(AbsStart, OldestSample);
	const int64 ClampedEnd   = FMath::Min(AbsEnd, TotalSamplesWritten);

	if (ClampedEnd <= ClampedStart)
	{
		return;
	}

	const int32 Count = static_cast<int32>(ClampedEnd - ClampedStart);
	OutSegment.SetNumUninitialized(Count);

	const int32 StartIdx = static_cast<int32>(ClampedStart % Capacity);

	if (StartIdx + Count <= Capacity)
	{
		// Contiguous
		FMemory::Memcpy(OutSegment.GetData(), RingBuffer.GetData() + StartIdx,
		                Count * sizeof(float));
	}
	else
	{
		// Wraps around
		const int32 FirstPart = Capacity - StartIdx;
		FMemory::Memcpy(OutSegment.GetData(),
		                RingBuffer.GetData() + StartIdx,
		                FirstPart * sizeof(float));
		FMemory::Memcpy(OutSegment.GetData() + FirstPart,
		                RingBuffer.GetData(),
		                (Count - FirstPart) * sizeof(float));
	}
}

// ---------------------------------------------------------------------------
// Audio processing utilities
// ---------------------------------------------------------------------------

void FWhisperNative::DownmixToMono(const float* InInterleaved, int32 NumFrames, int32 NumChannels,
                                    TArray<float>& OutMono)
{
	OutMono.SetNumUninitialized(NumFrames);

	if (NumChannels == 1)
	{
		FMemory::Memcpy(OutMono.GetData(), InInterleaved, NumFrames * sizeof(float));
	}
	else
	{
		const float InvN = 1.0f / static_cast<float>(NumChannels);
		for (int32 Frame = 0; Frame < NumFrames; ++Frame)
		{
			float Sum = 0.0f;
			for (int32 Ch = 0; Ch < NumChannels; ++Ch)
			{
				Sum += InInterleaved[Frame * NumChannels + Ch];
			}
			OutMono[Frame] = Sum * InvN;
		}
	}
}

void FWhisperNative::ResampleLinear(const float* InSamples, int32 InCount, int32 InRate,
                                     TArray<float>& OutSamples, int32 OutRate)
{
	if (InCount <= 0 || InRate <= 0 || OutRate <= 0)
	{
		OutSamples.Empty();
		return;
	}

	if (InRate == OutRate)
	{
		OutSamples.SetNumUninitialized(InCount);
		FMemory::Memcpy(OutSamples.GetData(), InSamples, InCount * sizeof(float));
		return;
	}

	const int32 OutCount = static_cast<int32>(
		static_cast<double>(InCount) * OutRate / InRate);

	OutSamples.SetNumUninitialized(OutCount);

	for (int32 i = 0; i < OutCount; ++i)
	{
		const double SrcPos = static_cast<double>(i) * InRate / OutRate;
		const int32  SrcIdx = static_cast<int32>(SrcPos);
		const float  Frac   = static_cast<float>(SrcPos - SrcIdx);

		const float S0 = InSamples[SrcIdx];
		const float S1 = (SrcIdx + 1 < InCount) ? InSamples[SrcIdx + 1] : S0;

		OutSamples[i] = S0 + Frac * (S1 - S0);
	}
}

float FWhisperNative::ComputeRMS(const float* Samples, int32 Count)
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
		OutSamples.SetNumUninitialized(NumFrames);
		const float InvN = 1.0f / static_cast<float>(NumChannels);
		for (int32 Frame = 0; Frame < NumFrames; ++Frame)
		{
			float Sum = 0.0f;
			for (int32 Ch = 0; Ch < NumChannels; ++Ch)
			{
				Sum += AllChannelSamples[Frame * NumChannels + Ch];
			}
			OutSamples[Frame] = Sum * InvN;
		}
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
