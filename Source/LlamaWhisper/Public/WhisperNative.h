// Copyright 2025-current Getnamo.

#pragma once

#include "CoreMinimal.h"
#include "WhisperDataTypes.h"
#include "LlamaDataTypes.h"  // FLLMThreadTask (reused from LlamaCore)

// Forward-declare AudioCapture types to avoid heavy header pull in downstream includes
namespace Audio
{
	class FAudioCapture;
}

/**
 * Threading wrapper for FWhisperInternal.
 *
 * Architecture mirrors FLlamaNative exactly:
 *   - Owns one background worker thread (started lazily on first task enqueue)
 *   - Two task queues: BackgroundTasks (MPSC — game thread + audio thread produce)
 *                       GameThreadTasks (SPSC — BG thread produces, GT consumes)
 *   - OnGameThreadTick() drains the GT queue; called from UWhisperComponent::TickComponent
 *
 * Audio pipeline:
 *   AudioCapture callback (HW thread) -> ring buffer (16 kHz mono float32)
 *     -> VAD state machine -> EnqueueBGTask dispatch
 *   BG thread: copies segment from ring buffer, calls FWhisperInternal::TranscribeAudio
 *   Result: EnqueueGTTask -> OnTranscriptionResult -> component delegates
 *
 * Thread safety:
 *   AudioBufferMutex guards the ring buffer and VAD state, which are shared between the
 *   audio HW thread (writer) and the BG thread (reader for segment copy).
 *   The BG thread holds the mutex only while copying the segment — NOT during inference.
 */
class LLAMAWHISPER_API FWhisperNative
{
public:
	// ---------------------------------------------------------------------------
	// Callbacks — all fired on the game thread via the GT task queue
	// ---------------------------------------------------------------------------

	TFunction<void(const FString& Text, bool bIsFinal)>      OnTranscriptionResult;
	TFunction<void(const FString& ModelPath)>                OnModelLoaded;
	TFunction<void(const FString& ErrorMessage)>             OnError;
	TFunction<void(bool bIsSpeechDetected)>                  OnVADStateChanged;
	TFunction<void(bool bIsTranscribing)>                    OnTranscribingStateChanged;
	TFunction<void(const FString& VADModelPath, bool bSuccess)> OnVADModelLoaded;

	// ---------------------------------------------------------------------------
	// Configuration
	// ---------------------------------------------------------------------------

	void SetModelParams(const FWhisperModelParams& Params);
	void SetStreamParams(const FWhisperStreamParams& Params);

	// ---------------------------------------------------------------------------
	// Model control (safe to call from game thread)
	// ---------------------------------------------------------------------------

	void LoadModel(bool bForceReload = false,
		TFunction<void(const FString& ModelPath, int32 StatusCode)> Callback = nullptr);
	void UnloadModel(TFunction<void(int32 StatusCode)> Callback = nullptr);
	bool IsModelLoaded() const;

	/** Load the Silero VAD model specified in StreamParams.PathToVADModel.
	 *  Called automatically by LoadModel when VADMode == Silero. */
	void LoadVADModel(TFunction<void(const FString& VADModelPath, int32 StatusCode)> Callback = nullptr);
	void UnloadVADModel(TFunction<void(int32 StatusCode)> Callback = nullptr);
	bool IsVADModelLoaded() const;

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

	/** Mute/unmute the microphone input. While muted the audio callback discards all
	 *  incoming samples — capture remains open and no flush/reset occurs. */
	void SetMicrophoneMuted(bool bMuted);
	bool IsMicrophoneMuted() const;

	// ---------------------------------------------------------------------------
	// Game thread pump — call from component/subsystem tick
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

	// MPSC queue: game thread AND audio HW thread may enqueue tasks
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
	FWhisperStreamParams StreamParams;

	bool bModelLoadInitiated = false;

	// ---------------------------------------------------------------------------
	// Audio ring buffer — shared between audio HW thread (write) and BG thread (read)
	// Protected by AudioBufferMutex.
	// ---------------------------------------------------------------------------

	FCriticalSection AudioBufferMutex;

	TArray<float> RingBuffer;           // Circular, capacity = StreamParams.RingBufferCapacitySamples
	int64         TotalSamplesWritten = 0; // Ever-increasing; ring index = TotalSamplesWritten % capacity

	// Pre-roll buffer: rolling window of pre-speech audio for VAD onset capture
	TArray<float> PreRollBuffer;
	int32         PreRollWritePos = 0;

	// ---------------------------------------------------------------------------
	// VAD state — protected by AudioBufferMutex
	// ---------------------------------------------------------------------------

	bool  bVADSpeechActive     = false;
	float VADSilenceDuration   = 0.0f;   // Accumulated silence seconds since last speech chunk
	int64 VADSpeechStartSample = 0;      // TotalSamplesWritten value when speech started (incl. pre-roll)
	float VADSegmentDuration   = 0.0f;   // Duration of current speech segment in seconds

	// ---------------------------------------------------------------------------
	// Audio capture state
	// ---------------------------------------------------------------------------

	Audio::FAudioCapture* AudioCapture         = nullptr;
	FThreadSafeBool       bMicCaptureActive    = false;
	FThreadSafeBool       bMicMuted            = false;
	int32                 CaptureDeviceSampleRate = 48000;
	int32                 CaptureDeviceChannels   = 1;

	// ---------------------------------------------------------------------------
	// Internal whisper wrapper — only accessed from the BG thread
	// ---------------------------------------------------------------------------

	class FWhisperInternal* Internal = nullptr;

	// BG thread state flags (set on BG thread, readable on GT via atomic)
	FThreadSafeBool bIsTranscribing = false;

	// ---------------------------------------------------------------------------
	// Silero VAD — only accessed from the BG thread
	// ---------------------------------------------------------------------------

	// Forward-declared to avoid pulling whisper.h into all consumers of this header
	struct whisper_vad_context* SileroContext = nullptr;

	// Ring buffer read cursor for Silero: position up to which the BG thread has processed.
	// Written and read exclusively on the BG thread — no mutex needed.
	int64 SileroChunkStart = 0;

	// Guards against flooding the BG queue with overlapping Silero tasks.
	FThreadSafeBool bSileroInFlight = false;

	// Residual samples not yet forming a complete Silero window (512 samples).
	// Written and read exclusively on the BG thread.
	TArray<float> SileroResidual;

	// ---------------------------------------------------------------------------
	// Ticker handle (for optional standalone tick via AddTicker)
	// ---------------------------------------------------------------------------

	FTSTicker::FDelegateHandle TickDelegateHandle = nullptr;

	// ---------------------------------------------------------------------------
	// Audio processing helpers (called on audio HW thread or BG thread)
	// ---------------------------------------------------------------------------

	/** Average interleaved stereo (or N-channel) frames to mono. */
	static void DownmixToMono(const float* InInterleaved, int32 NumFrames, int32 NumChannels,
	                          TArray<float>& OutMono);

	/** Linear interpolation resampler: mono float32 from InRate to OutRate. */
	static void ResampleLinear(const float* InSamples, int32 InCount, int32 InRate,
	                           TArray<float>& OutSamples, int32 OutRate);

	/** Compute root-mean-square energy of a sample block. */
	static float ComputeRMS(const float* Samples, int32 Count);

	/** Append 16 kHz mono samples to the ring buffer.
	 *  Must be called with AudioBufferMutex held. */
	void AppendToRingBuffer_Locked(const TArray<float>& Samples16k);

	/** Copy ring buffer samples from [AbsoluteStart, AbsoluteEnd) into OutSegment.
	 *  Must be called with AudioBufferMutex held. */
	void CopyRingBufferSegment_Locked(int64 AbsoluteStart, int64 AbsoluteEnd,
	                                  TArray<float>& OutSegment) const;

	/** Process new 16 kHz audio through the VAD state machine.
	 *  Optionally fires a dispatch by enqueuing a BG task.
	 *  Must be called with AudioBufferMutex held. */
	void ProcessVAD_Locked(const TArray<float>& NewSamples16k);

	/** Enqueue a transcription task for the segment [AbsStart, AbsEnd).
	 *  Should be called WITHOUT AudioBufferMutex held (acquires it internally for the copy). */
	void DispatchSegmentForTranscription(int64 AbsStart, int64 AbsEnd);

	/** Run Silero VAD on ring buffer audio up to ChunkEnd.
	 *  Drives the same onset/offset state machine as EnergyBased mode.
	 *  Called exclusively from the BG thread. */
	void ProcessSileroVADChunk(int64 ChunkEnd);

	/** Internal callback registered with FAudioCapture. Runs on audio HW thread. */
	void HandleAudioCaptureCallback(const float* InAudio, int32 NumFrames,
	                                int32 NumChannels, int32 SampleRate);

	/** Load a RIFF/WAVE file into a float32 16 kHz mono array.
	 *  Returns true on success; fills OutSamples and OutSampleRate. */
	static bool LoadWavFile(const FString& FilePath,
	                        TArray<float>& OutSamples, int32& OutSampleRate);
};
