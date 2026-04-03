// Copyright 2025-current Getnamo.

#pragma once

#include "WhisperDataTypes.generated.h"

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

UENUM(BlueprintType)
enum class EWhisperSamplingStrategy : uint8
{
	Greedy     UMETA(DisplayName = "Greedy"),
	BeamSearch UMETA(DisplayName = "Beam Search"),
};

UENUM(BlueprintType)
enum class EWhisperVADMode : uint8
{
	/** No VAD. Audio accumulates from StartMicrophoneCapture to Stop, dispatched as one chunk
	 *  (or at MaxSpeechSegmentSec boundaries with optional NonVADOverlapSec overlap). */
	Disabled    UMETA(DisplayName = "Disabled"),

	/** Lightweight onset/offset detection using RMS energy threshold.
	 *  Fast, zero extra model, works best in quiet environments. */
	EnergyBased UMETA(DisplayName = "Energy-Based (RMS)"),

	/** Neural VAD using the ggml Silero model. More accurate in noisy environments.
	 *  Requires PathToVADModel to point to a ggml-silero-vX.X.X.bin file. */
	Silero      UMETA(DisplayName = "Silero Neural VAD"),
};

// ---------------------------------------------------------------------------
// Delegates
// ---------------------------------------------------------------------------

/** Fires for each completed transcription segment. bIsFinal is true for one-shot modes;
 *  in streaming mode it may fire multiple times before a definitive final result. */
DECLARE_DYNAMIC_MULTICAST_DELEGATE_TwoParams(FOnWhisperTranscriptionResult,
	const FString&, Text,
	bool, bIsFinal);

/** Fires once when the whisper model has finished loading. */
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnWhisperModelLoaded,
	const FString&, ModelPath);

/** Fires on internal whisper errors (model load failures, inference errors, etc.). */
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnWhisperError,
	const FString&, ErrorMessage);

/** Fires when voice activity detection transitions between speech and silence. */
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnWhisperVADStateChanged,
	bool, bIsSpeechDetected);

/** Fires once when the Silero VAD model has finished loading (Silero mode only). */
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnWhisperVADModelLoaded,
	const FString&, VADModelPath);

// ---------------------------------------------------------------------------
// Structs
// ---------------------------------------------------------------------------

/** Parameters for loading and running a whisper model. */
USTRUCT(BlueprintType)
struct LLAMAWHISPER_API FWhisperModelParams
{
	GENERATED_USTRUCT_BODY()

	/** Path to the .bin whisper model file.
	 *  Paths beginning with '.' are relative to Saved/Models/ (same convention as LlamaCore). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Whisper Model Params")
	FString PathToModel = TEXT("./whisper-base.en.bin");

	/** BCP-47 language code for the spoken language, e.g. "en", "fr", "auto".
	 *  Use "auto" to let whisper detect the language automatically. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Whisper Model Params")
	FString Language = TEXT("en");

	/** Translate the transcription to English regardless of the input language. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Whisper Model Params")
	bool bTranslate = false;

	/** Number of CPU threads to use during inference. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Whisper Model Params",
		meta = (ClampMin = "1", ClampMax = "64"))
	int32 Threads = 4;

	/** Attempt GPU acceleration via the ggml Vulkan/CUDA backend (if available). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Whisper Model Params")
	bool bUseGPU = true;

	/** Maximum context tokens from previous transcriptions used as decoder prompt (0 = no limit). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Whisper Model Params")
	int32 MaxContext = 0;

	/** Decoding strategy: Greedy is faster; Beam Search is more accurate. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Whisper Model Params")
	EWhisperSamplingStrategy SamplingStrategy = EWhisperSamplingStrategy::Greedy;

	/** Best-of candidates for Greedy decoding (higher = slightly more accurate, slower). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Whisper Model Params",
		meta = (ClampMin = "1", ClampMax = "10", EditCondition = "SamplingStrategy == EWhisperSamplingStrategy::Greedy"))
	int32 BestOf = 2;

	/** Beam size for Beam Search decoding. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Whisper Model Params",
		meta = (ClampMin = "1", ClampMax = "10", EditCondition = "SamplingStrategy == EWhisperSamplingStrategy::BeamSearch"))
	int32 BeamSize = 5;

	/** Automatically load the model when the component activates. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Whisper Model Params")
	bool bAutoLoadModelOnStartup = true;
};

/** Parameters controlling the streaming audio pipeline and voice activity detection. */
USTRUCT(BlueprintType)
struct LLAMAWHISPER_API FWhisperStreamParams
{
	GENERATED_USTRUCT_BODY()

	/** Voice activity detection mode. Controls how speech segments are detected for transcription. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Whisper Stream Params")
	EWhisperVADMode VADMode = EWhisperVADMode::EnergyBased;

	/** Path to the ggml Silero VAD model file (e.g. ggml-silero-v6.2.0.bin).
	 *  Paths beginning with '.' are relative to Saved/Models/. Only used when VADMode is Silero. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Whisper Stream Params",
		meta = (EditCondition = "VADMode == EWhisperVADMode::Silero"))
	FString PathToVADModel = TEXT("./ggml-silero-v6.2.0.bin");

	/** Silero speech probability threshold [0.0–1.0].
	 *  A window is considered speech if its probability exceeds this value.
	 *  Lower values are more sensitive; raise to reduce false positives in noisy environments. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Whisper Stream Params",
		meta = (ClampMin = "0.0", ClampMax = "1.0", EditCondition = "VADMode == EWhisperVADMode::Silero"))
	float SileroThreshold = 0.5f;

	/** RMS energy threshold for voice onset/offset detection [0.0–1.0].
	 *  Lower values are more sensitive; raise if background noise causes false triggers.
	 *  Only used when VADMode is EnergyBased. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Whisper Stream Params",
		meta = (ClampMin = "0.0", ClampMax = "1.0", EditCondition = "VADMode == EWhisperVADMode::EnergyBased"))
	float VADThreshold = 0.02f;

	/** Seconds of silence before an active speech segment is considered ended (EnergyBased mode).
	 *  Higher values avoid premature cutoffs in paused speech. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Whisper Stream Params",
		meta = (ClampMin = "0.1", EditCondition = "VADMode == EWhisperVADMode::EnergyBased"))
	float VADHoldTimeSec = 0.8f;

	/** Seconds of silence before an active speech segment is considered ended (Silero mode).
	 *  Silero's neural detection is more precise, so a shorter hold time is appropriate. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Whisper Stream Params",
		meta = (ClampMin = "0.05", EditCondition = "VADMode == EWhisperVADMode::Silero"))
	float SileroHoldTimeSec = 0.2f;

	/** Pre-roll: seconds of audio before VAD onset to include at the start of a segment,
	 *  ensuring consonant attack sounds are captured. Used by EnergyBased and Silero modes. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Whisper Stream Params",
		meta = (ClampMin = "0.0", ClampMax = "2.0"))
	float VADPreRollSec = 0.15f;

	/** Maximum segment length (seconds) before a forced mid-stream dispatch.
	 *  In VAD mode this acts as a safety valve when speech runs too long.
	 *  In no-VAD mode this is the chunk boundary: audio is split here when the mic
	 *  session exceeds this duration, and capture continues into the next chunk. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Whisper Stream Params",
		meta = (ClampMin = "1.0"))
	float MaxSpeechSegmentSec = 15.0f;

	/** Overlap (seconds) between consecutive forced chunks when VAD is disabled.
	 *  The tail of each dispatched chunk is re-included at the start of the next chunk,
	 *  preventing words at boundaries from being cut off. Set to 0 to disable overlap. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Whisper Stream Params",
		meta = (ClampMin = "0.0", ClampMax = "5.0", EditCondition = "VADMode == EWhisperVADMode::Disabled"))
	float NonVADOverlapSec = 0.5f;

	/** Ring buffer capacity in samples at 16 kHz. 30 seconds = 480,000 samples (~1.8 MB). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Whisper Stream Params",
		meta = (ClampMin = "16000"))
	int32 RingBufferCapacitySamples = 16000 * 30;
};

/** Runtime state exposed to Blueprint consumers. */
USTRUCT(BlueprintType)
struct LLAMAWHISPER_API FWhisperModelState
{
	GENERATED_USTRUCT_BODY()

	/** True once a model has been loaded and is ready for inference. */
	UPROPERTY(BlueprintReadOnly, Category = "Whisper Model State")
	bool bModelLoaded = false;

	/** True once the Silero VAD model has been loaded (only relevant in Silero VAD mode). */
	UPROPERTY(BlueprintReadOnly, Category = "Whisper Model State")
	bool bVADModelLoaded = false;

	/** True while a transcription task is executing on the background thread. */
	UPROPERTY(BlueprintReadOnly, Category = "Whisper Model State")
	bool bIsTranscribing = false;

	/** True while the microphone capture stream is open and running. */
	UPROPERTY(BlueprintReadOnly, Category = "Whisper Model State")
	bool bMicrophoneActive = false;

	/** Current VAD state: true when voice activity is being detected. */
	UPROPERTY(BlueprintReadOnly, Category = "Whisper Model State")
	bool bVADSpeechDetected = false;

	/** Text from the most recently completed transcription. */
	UPROPERTY(BlueprintReadOnly, Category = "Whisper Model State")
	FString LastTranscription;
};
