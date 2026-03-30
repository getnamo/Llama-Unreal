// Copyright 2025-current Getnamo.

#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "WhisperDataTypes.h"

#include "WhisperComponent.generated.h"

/**
 * Actor component providing Blueprint-accessible speech-to-text via whisper.cpp.
 *
 * Each component owns its own whisper model instance and runs inference on an independent
 * background thread — safe to run in parallel with ULlamaComponent on the same actor.
 *
 * Usage modes:
 *   1. One-shot:   Call TranscribeAudioData() or TranscribeWaveFile(). Bind OnTranscriptionResult.
 *   2. Streaming:  Call StartMicrophoneCapture(). Bind OnTranscriptionResult and OnVADStateChanged.
 *
 * Lifetime: inherits from the owning actor (same as ULlamaComponent). For persistence across
 * level transitions, wrap this component in an always-persistent actor or subsystem.
 */
UCLASS(Category = "Whisper", BlueprintType, meta = (BlueprintSpawnableComponent))
class LLAMAWHISPER_API UWhisperComponent : public UActorComponent
{
	GENERATED_BODY()

public:
	UWhisperComponent(const FObjectInitializer& ObjectInitializer);
	~UWhisperComponent();

	virtual void BeginPlay() override;
	virtual void Activate(bool bReset) override;
	virtual void Deactivate() override;
	virtual void TickComponent(float DeltaTime, ELevelTick TickType,
	                           FActorComponentTickFunction* ThisTickFunction) override;

	// ---------------------------------------------------------------------------
	// Delegates
	// ---------------------------------------------------------------------------

	/** Fires for each completed transcription. bIsFinal is always true for one-shot;
	 *  in streaming mode it fires each time a VAD-gated segment is transcribed. */
	UPROPERTY(BlueprintAssignable, Category = "Whisper Component")
	FOnWhisperTranscriptionResult OnTranscriptionResult;

	/** Fires once when the model has loaded and is ready. */
	UPROPERTY(BlueprintAssignable, Category = "Whisper Component")
	FOnWhisperModelLoaded OnModelLoaded;

	/** Fires on internal errors (load failures, inference errors). */
	UPROPERTY(BlueprintAssignable, Category = "Whisper Component")
	FOnWhisperError OnError;

	/** Fires when VAD transitions: true = speech started, false = speech ended. */
	UPROPERTY(BlueprintAssignable, Category = "Whisper Component")
	FOnWhisperVADStateChanged OnVADStateChanged;

	/** Fires when the Silero VAD model finishes loading (Silero mode only). */
	UPROPERTY(BlueprintAssignable, Category = "Whisper Component")
	FOnWhisperVADModelLoaded OnVADModelLoaded;

	// ---------------------------------------------------------------------------
	// Properties
	// ---------------------------------------------------------------------------

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Whisper Component")
	FWhisperModelParams ModelParams;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Whisper Component")
	FWhisperStreamParams StreamParams;

	/** Read-only runtime state. Updated on the game thread from background thread callbacks. */
	UPROPERTY(BlueprintReadOnly, Category = "Whisper Component")
	FWhisperModelState ModelState;

	// ---------------------------------------------------------------------------
	// Blueprint API
	// ---------------------------------------------------------------------------

	/** Load (or reload) the whisper model specified in ModelParams. */
	UFUNCTION(BlueprintCallable, Category = "Whisper Component")
	void LoadModel(bool bForceReload = true);

	/** Unload the model and free its memory. */
	UFUNCTION(BlueprintCallable, Category = "Whisper Component")
	void UnloadModel();

	/** Returns true if the model is currently loaded and ready for inference. */
	UFUNCTION(BlueprintPure, Category = "Whisper Component")
	bool IsModelLoaded() const;

	/**
	 * Transcribe float32 PCM audio data. Automatically resampled to 16 kHz if needed.
	 * @param PCMSamples  Float32 mono or stereo audio (stereo is averaged to mono).
	 * @param SampleRate  Sample rate of the supplied audio.
	 */
	UFUNCTION(BlueprintCallable, Category = "Whisper Component")
	void TranscribeAudioData(TArray<float> PCMSamples, int32 SampleRate);

	/**
	 * Load a RIFF/WAVE file from disk and transcribe it.
	 * Only 16-bit PCM mono/stereo WAV files are supported.
	 * @param FilePath  Absolute path or relative path starting with '.' (resolved to Saved/Models/).
	 */
	UFUNCTION(BlueprintCallable, Category = "Whisper Component")
	void TranscribeWaveFile(const FString& FilePath);

	/** Start listening to the default microphone. Fires OnTranscriptionResult per speech segment. */
	UFUNCTION(BlueprintCallable, Category = "Whisper Component")
	void StartMicrophoneCapture();

	/** Stop microphone capture. Any in-progress speech segment is dispatched as a final result. */
	UFUNCTION(BlueprintCallable, Category = "Whisper Component")
	void StopMicrophoneCapture();

	/** Returns true if microphone capture is currently active. */
	UFUNCTION(BlueprintPure, Category = "Whisper Component")
	bool IsMicrophoneCaptureActive() const;

	/** Manually load (or reload) the Silero VAD model specified in StreamParams.PathToVADModel.
	 *  Not needed in normal use — the VAD model loads automatically when LoadModel is called
	 *  with VADMode set to Silero. */
	UFUNCTION(BlueprintCallable, Category = "Whisper Component")
	void LoadVADModel();

	/** Unload the Silero VAD model and free its memory. */
	UFUNCTION(BlueprintCallable, Category = "Whisper Component")
	void UnloadVADModel();

	/** Returns true if the Silero VAD model is currently loaded. */
	UFUNCTION(BlueprintPure, Category = "Whisper Component")
	bool IsVADModelLoaded() const;

private:
	class FWhisperNative* WhisperNative;
};
