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

	/** Enable voice activity detection. When false, audio is dispatched in fixed-size chunks. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Whisper Stream Params")
	bool bUseVAD = true;

	/** RMS energy threshold for voice onset/offset detection [0.0–1.0].
	 *  Lower values are more sensitive; raise if background noise causes false triggers. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Whisper Stream Params",
		meta = (ClampMin = "0.0", ClampMax = "1.0"))
	float VADThreshold = 0.02f;

	/** Seconds of audio below VADThreshold before an active speech segment is considered ended. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Whisper Stream Params",
		meta = (ClampMin = "0.1"))
	float VADHoldTimeSec = 0.8f;

	/** Pre-roll: seconds of audio before VAD onset to include at the start of a segment,
	 *  ensuring consonant attack sounds are captured. */
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
		meta = (ClampMin = "0.0", ClampMax = "5.0", EditCondition = "!bUseVAD"))
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
