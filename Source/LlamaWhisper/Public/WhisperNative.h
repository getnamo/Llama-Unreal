// Copyright 2025-current Getnamo.

#pragma once

#include "CoreMinimal.h"
#include "WhisperDataTypes.h"
#include "LlamaDataTypes.h"  // FLLMThreadTask (reused from LlamaCore)
#include "LlamaMediaCaptureTypes.h"
#include "Containers/Queue.h"
#include "HAL/ThreadSafeBool.h"
#include "HAL/ThreadSafeCounter.h"
#include "Containers/Ticker.h"

/**
 * Threading wrapper for FWhisperInternal.
 *
 * Architecture mirrors FLlamaNative exactly:
 *   - Owns one background worker thread (started lazily on first task enqueue)
 *   - Two task queues: BackgroundTasks (MPSC -- game thread + audio thread produce)
 *                       GameThreadTasks (SPSC -- BG thread produces, GT consumes)
 *   - OnGameThreadTick() drains the GT queue; called from UWhisperComponent::TickComponent
 *
 * Audio pipeline:
 *   ULlamaAudioCaptureComponent handles mic capture, ring buffer, and VAD.
 *   When a speech segment is ready, it calls OnAudioSegment() on registered consumers.
 *   FWhisperNative implements ILlamaAudioConsumer and enqueues transcription tasks
 *   on the BG thread.
 *
 * Thread safety:
 *   OnAudioSegment is called from the capture component's BG thread.
 *   It enqueues work to this class's own BG thread via EnqueueBGTask -- no shared
 *   mutable state between the two threads.
 */
class LLAMAWHISPER_API FWhisperNative : public ILlamaAudioConsumer
{
public:
	// ---------------------------------------------------------------------------
	// Callbacks -- all fired on the game thread via the GT task queue
	// ---------------------------------------------------------------------------

	TFunction<void(const FString& Text, bool bIsFinal)>      OnTranscriptionResult;
	TFunction<void(const FString& ModelPath)>                OnModelLoaded;
	TFunction<void(const FString& ErrorMessage)>             OnError;
	TFunction<void(bool bIsTranscribing)>                    OnTranscribingStateChanged;

	// ---------------------------------------------------------------------------
	// Configuration
	// ---------------------------------------------------------------------------

	void SetModelParams(const FWhisperModelParams& Params);

	// ---------------------------------------------------------------------------
	// Model control (safe to call from game thread)
	// ---------------------------------------------------------------------------

	void LoadModel(bool bForceReload = false,
		TFunction<void(const FString& ModelPath, int32 StatusCode)> Callback = nullptr);
	void UnloadModel(TFunction<void(int32 StatusCode)> Callback = nullptr);
	bool IsModelLoaded() const;


	// ---------------------------------------------------------------------------
	// One-shot transcription (safe to call from game thread)
	// ---------------------------------------------------------------------------

	/** Submit a float32 PCM array for transcription. Will be resampled to 16 kHz if needed. */
	void TranscribeAudioData(TArray<float> PCMSamples, int32 SampleRate);

	/** Load a .wav file from disk and transcribe it. Handles 16-bit PCM RIFF WAVE only. */
	void TranscribeWaveFile(const FString& FilePath);

	// ---------------------------------------------------------------------------
	// Microphone streaming (safe to call from game thread)
	// ---------------------------------------------------------------------------

	void StartMicrophoneCapture();
	void StopMicrophoneCapture();
	bool IsMicrophoneCaptureActive() const;

	/** Mute/unmute the microphone input. Forwards to the active audio source. */
	void SetMicrophoneMuted(bool bMuted);
	bool IsMicrophoneMuted() const;

	// ---------------------------------------------------------------------------
	// External audio source
	// ---------------------------------------------------------------------------

	/** Set an externally-owned ULlamaAudioCaptureComponent as the audio source.
	 *  When set, StartMicrophoneCapture will subscribe to this source instead of
	 *  creating its own internal capture component. Pass nullptr to revert to
	 *  internal capture. If currently capturing, capture is stopped first. */
	void SetExternalAudioSource(class ULlamaAudioCaptureComponent* Source);

	// ---------------------------------------------------------------------------
	// ILlamaAudioConsumer interface
	// ---------------------------------------------------------------------------

	/** Called on the capture component's BG thread when a speech segment is ready. */
	virtual void OnAudioSegment(const FLlamaAudioSegment& Segment) override;

	// ---------------------------------------------------------------------------
	// Game thread pump -- call from component/subsystem tick
	// ---------------------------------------------------------------------------

	void OnGameThreadTick(float DeltaTime);
	void AddTicker();     // Use if no component tick is available (subsystem case)
	void RemoveTicker();
	bool IsNativeTickerActive() const;

	FWhisperNative();
	~FWhisperNative();

	float ThreadIdleSleepDuration = 0.005f;

private:
	// ---------------------------------------------------------------------------
	// Background thread (mirrors FLlamaNative exactly)
	// ---------------------------------------------------------------------------

	void StartWhisperThread();

	// MPSC queue: game thread AND audio consumer thread may enqueue tasks
	TQueue<FLLMThreadTask, EQueueMode::Mpsc> BackgroundTasks;
	// SPSC queue: only BG thread enqueues, GT consumes
	TQueue<FLLMThreadTask>                   GameThreadTasks;

	FThreadSafeBool    bThreadIsActive  = false;
	FThreadSafeBool    bThreadShouldRun = false;
	FThreadSafeCounter TaskIdCounter    = 0;

	int64 GetNextTaskId();
	void  EnqueueBGTask(TFunction<void(int64)> Task);
	void  EnqueueGTTask(TFunction<void()> Task, int64 LinkedTaskId = -1);

	// ---------------------------------------------------------------------------
	// Cached parameters (GT-owned, copied before BG use)
	// ---------------------------------------------------------------------------

	FWhisperModelParams ModelParams;

	bool bModelLoadInitiated = false;

	// ---------------------------------------------------------------------------
	// Audio source management
	// ---------------------------------------------------------------------------

	/** Externally-owned audio source (set via SetExternalAudioSource). */
	class ULlamaAudioCaptureComponent* ExternalAudioSource = nullptr;

	/** Internally-created audio source (created when no external source is set). */
	class ULlamaAudioCaptureComponent* InternalAudioSource = nullptr;

	/** Returns whichever audio source is active (external preferred over internal). */
	ULlamaAudioCaptureComponent* GetActiveAudioSource() const;

	/** Tracks whether we have subscribed to an audio source. */
	FThreadSafeBool bMicCaptureActive = false;
	FThreadSafeBool bMicMuted         = false;

	// ---------------------------------------------------------------------------
	// Internal whisper wrapper -- only accessed from the BG thread
	// ---------------------------------------------------------------------------

	class FWhisperInternal* Internal = nullptr;

	// BG thread state flags (set on BG thread, readable on GT via atomic)
	FThreadSafeBool bIsTranscribing = false;

	// ---------------------------------------------------------------------------
	// Ticker handle (for optional standalone tick via AddTicker)
	// ---------------------------------------------------------------------------

	FTSTicker::FDelegateHandle TickDelegateHandle = nullptr;

	// ---------------------------------------------------------------------------
	// WAV file loading utility
	// ---------------------------------------------------------------------------

	/** Load a RIFF/WAVE file into a float32 16 kHz mono array.
	 *  Returns true on success; fills OutSamples and OutSampleRate. */
	static bool LoadWavFile(const FString& FilePath,
	                        TArray<float>& OutSamples, int32& OutSampleRate);
};
