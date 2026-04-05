// Copyright 2025-current Getnamo.

#pragma once

#include "CoreMinimal.h"
#include "LlamaMediaCaptureTypes.generated.h"

class UActorComponent;
class UTexture2D;

// ─── Enums ───────────────────────────────────────────────────────────────────

/** Voice-activity-detection strategy for audio capture. */
UENUM(BlueprintType)
enum class ELlamaVADMode : uint8
{
	/** No VAD — segments are dispatched on a fixed timer or manually. */
	Disabled		UMETA(DisplayName = "Disabled"),

	/** Simple energy / RMS threshold VAD (low latency, low CPU). */
	EnergyBased		UMETA(DisplayName = "Energy Based"),

	/** Silero neural-network VAD (higher accuracy, requires background thread). */
	Silero			UMETA(DisplayName = "Silero")
};

/** Source type for video frame capture. */
UENUM(BlueprintType)
enum class EVideoCaptureSource : uint8
{
	/** Physical webcam device. */
	Webcam			UMETA(DisplayName = "Webcam"),

	/** In-engine Scene Capture Component 2D. */
	SceneCapture	UMETA(DisplayName = "Scene Capture")
};

// ─── Structs ─────────────────────────────────────────────────────────────────

/** A chunk of captured audio ready for downstream processing (STT, etc.). */
USTRUCT(BlueprintType)
struct LLAMACORE_API FLlamaAudioSegment
{
	GENERATED_USTRUCT_BODY()

	/** Raw PCM samples — 16 kHz, mono, float32. */
	UPROPERTY(BlueprintReadOnly, Category = "Llama|Audio")
	TArray<float> PCMSamples;

	/** Duration of this segment in seconds. */
	UPROPERTY(BlueprintReadOnly, Category = "Llama|Audio")
	float DurationSeconds = 0.0f;

	/** True when the segment boundaries were determined by VAD onset/offset,
	 *  false when dispatched by a fixed timer or manual trigger. */
	UPROPERTY(BlueprintReadOnly, Category = "Llama|Audio")
	bool bIsVADTriggered = false;
};

// ─── Delegates ───────────────────────────────────────────────────────────────

/** Fired when an audio segment is ready for consumption (e.g. STT). */
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnLlamaAudioSegmentReady, const FLlamaAudioSegment&, Segment);

/** Fired when the VAD state transitions (speech detected / silence). */
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnLlamaVADStateChanged, bool, bIsSpeechDetected);

/** Fired when a new video frame is available. */
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnLlamaVideoFrameReady, UTexture2D*, Frame);

// ─── Pure C++ Interfaces ─────────────────────────────────────────────────────

/**
 * Downstream consumer of captured audio segments.
 * Implementations must be thread-safe — OnAudioSegment is called from a
 * background capture thread.
 */
class LLAMACORE_API ILlamaAudioConsumer
{
public:
	/** Called on the capture component's BG thread.
	 *  Implementer must be thread-safe — typically just enqueue to own BG queue. */
	virtual void OnAudioSegment(const FLlamaAudioSegment& Segment) = 0;
	virtual ~ILlamaAudioConsumer() = default;

	// --- Component ↔ Consumer registry (enables Blueprint wiring) ---

	/** Register a UActorComponent as owning this consumer. Call in Activate(). */
	static void RegisterComponent(UActorComponent* Component, ILlamaAudioConsumer* Consumer);

	/** Unregister a component. Call in Deactivate() or destructor. */
	static void UnregisterComponent(UActorComponent* Component);

	/** Look up the consumer for a given component. Returns nullptr if not registered. */
	static ILlamaAudioConsumer* FindConsumerForComponent(UActorComponent* Component);

private:
	static TMap<UActorComponent*, ILlamaAudioConsumer*>& GetComponentRegistry();
};

/** Result of a single VAD evaluation step. */
struct FVADDecision
{
	enum class EResult : uint8 { Continue, DispatchSegment, SpeechOnset };

	EResult Result = EResult::Continue;
	int64 SegmentStartSample = 0;
	int64 SegmentEndSample = 0;
};

/**
 * Pluggable audio processor (VAD strategy, gain, etc.).
 * ProcessAudioChunk may be called from a background thread depending on
 * RequiresBackgroundThread().
 */
class LLAMACORE_API ILlamaAudioProcessor
{
public:
	virtual FVADDecision ProcessAudioChunk(const float* Samples, int32 Count,
											int64 AbsoluteWritePos, float DeltaSeconds) = 0;
	virtual void Reset() = 0;

	/** If true, ProcessAudioChunk must be called on a background thread (e.g. Silero neural VAD). */
	virtual bool RequiresBackgroundThread() const = 0;
	virtual ~ILlamaAudioProcessor() = default;
};

// ─── Processor Registry ──────────────────────────────────────────────────────

/** Static factory registry for ILlamaAudioProcessor implementations. */
class LLAMACORE_API FLlamaAudioProcessorRegistry
{
public:
	using FFactoryFunc = TFunction<ILlamaAudioProcessor*(const FString& /*ConfigPath*/)>;

	static void Register(const FString& Name, FFactoryFunc Factory);
	static ILlamaAudioProcessor* Create(const FString& Name, const FString& ConfigPath = TEXT(""));
	static bool IsRegistered(const FString& Name);

private:
	static TMap<FString, FFactoryFunc>& GetRegistry();
};
