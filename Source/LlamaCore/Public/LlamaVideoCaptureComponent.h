// Copyright 2025-current Getnamo.

#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "LlamaMediaCaptureTypes.h"

#include "LlamaVideoCaptureComponent.generated.h"

struct FMediaCaptureDeviceInfo;
class UMediaPlayer;
class UMediaSource;
class UMediaTexture;
class USceneCaptureComponent2D;
class UTextureRenderTarget2D;
class UTexture2D;

/** Blueprint-friendly wrapper for a discovered video capture device. */
USTRUCT(BlueprintType)
struct LLAMACORE_API FLlamaVideoDevice
{
	GENERATED_USTRUCT_BODY()

	/** Human-readable display name (e.g. "Logitech C920"). */
	UPROPERTY(BlueprintReadOnly, EditAnywhere, Category = "Video Capture")
	FString DisplayName;

	/** Media URL to pass to the player (e.g. "vidcap://..."). */
	UPROPERTY(BlueprintReadOnly, EditAnywhere, Category = "Video Capture")
	FString URL;

	/** Platform-specific debug info. */
	UPROPERTY(BlueprintReadOnly, EditAnywhere, Category = "Video Capture")
	FString Info;
};

/**
 * Video capture component with on-demand frame snapshot.
 * Supports two source modes:
 *   - Webcam: Real hardware camera via Unreal's Media Framework (UMediaPlayer).
 *   - SceneCapture: In-game rendered scene via USceneCaptureComponent2D.
 *
 * Call SnapshotFrame() to capture the current frame as a UTexture2D.
 * Feed the result to ULlamaComponent::InsertTemplateImagePrompt() for vision inference.
 */
UCLASS(Category = "LLM", BlueprintType, meta = (BlueprintSpawnableComponent))
class LLAMACORE_API ULlamaVideoCaptureComponent : public UActorComponent
{
	GENERATED_BODY()

public:
	ULlamaVideoCaptureComponent(const FObjectInitializer& ObjectInitializer);

	virtual void BeginPlay() override;
	virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;

	// -----------------------------------------------------------------------
	// Configuration
	// -----------------------------------------------------------------------

	/** Capture source: real webcam hardware or in-game scene. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Video Capture")
	EVideoCaptureSource CaptureSource = EVideoCaptureSource::Webcam;

	/** Resolution width for the capture. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Video Capture",
		meta = (ClampMin = "64", ClampMax = "3840"))
	int32 CaptureWidth = 512;

	/** Resolution height for the capture. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Video Capture",
		meta = (ClampMin = "64", ClampMax = "2160"))
	int32 CaptureHeight = 512;

	/** Index into the enumerated device list. 0 = first/default device.
	 *  Call EnumerateVideoDevices() to discover available devices. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Video Capture",
		meta = (ClampMin = "0", EditCondition = "CaptureSource == EVideoCaptureSource::Webcam"))
	int32 SelectedDeviceIndex = 0;

	/** Override webcam URL. When non-empty, this takes priority over SelectedDeviceIndex.
	 *  Leave empty to use the device selected by index. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Video Capture",
		meta = (EditCondition = "CaptureSource == EVideoCaptureSource::Webcam"))
	FString WebcamURLOverride;

	/** Whether to start capture automatically on BeginPlay. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Video Capture")
	bool bAutoStartCapture = false;

	// -----------------------------------------------------------------------
	// Delegates
	// -----------------------------------------------------------------------

	/** Fires when the webcam media has opened and playback has started.
	 *  Snapshot is safe to call after this fires. */
	UPROPERTY(BlueprintAssignable, Category = "Video Capture")
	FOnLlamaWebcamReady OnWebcamReady;

	// -----------------------------------------------------------------------
	// Blueprint API
	// -----------------------------------------------------------------------

	/** Enumerate available video capture devices on this platform.
	 *  Results are cached until the next call. */
	UFUNCTION(BlueprintCallable, Category = "Video Capture")
	TArray<FLlamaVideoDevice> EnumerateVideoDevices();

	/** Get the currently selected device info (after EnumerateVideoDevices has been called).
	 *  Returns an empty struct if no devices are available or index is out of range. */
	UFUNCTION(BlueprintPure, Category = "Video Capture")
	FLlamaVideoDevice GetSelectedDevice() const;

	/** Returns true if the webcam has fully opened and is streaming. */
	UFUNCTION(BlueprintPure, Category = "Video Capture")
	bool IsWebcamReady() const { return bMediaReady; }

	UFUNCTION(BlueprintCallable, Category = "Video Capture")
	void StartCapture();

	UFUNCTION(BlueprintCallable, Category = "Video Capture")
	void StopCapture();

	UFUNCTION(BlueprintPure, Category = "Video Capture")
	bool IsCaptureActive() const;

	/** Capture the current frame and return it as a transient UTexture2D (BGRA).
	 *  Performs a GPU->CPU readback -- avoid calling every frame.
	 *  Returns nullptr if capture is not active or readback fails. */
	UFUNCTION(BlueprintCallable, Category = "Video Capture")
	UTexture2D* SnapshotFrame();

	/** Get the live media texture (Webcam mode). Can be used in materials or UI for preview.
	 *  Returns nullptr in SceneCapture mode. */
	UFUNCTION(BlueprintPure, Category = "Video Capture")
	UMediaTexture* GetMediaTexture() const;

	/** Get the render target (SceneCapture mode). Returns nullptr in Webcam mode. */
	UFUNCTION(BlueprintPure, Category = "Video Capture")
	UTextureRenderTarget2D* GetRenderTarget() const;

	/** Set the scene capture component to use (SceneCapture mode).
	 *  If not set, the component will try to find one on the owning actor. */
	UFUNCTION(BlueprintCallable, Category = "Video Capture")
	void SetSceneCaptureComponent(USceneCaptureComponent2D* InSceneCapture);

private:
	// Webcam mode
	UPROPERTY()
	UMediaPlayer* MediaPlayer = nullptr;

	UPROPERTY()
	UMediaSource* MediaSource = nullptr;

	UPROPERTY()
	UMediaTexture* CachedMediaTexture = nullptr;

	// Scene capture mode
	UPROPERTY()
	USceneCaptureComponent2D* SceneCaptureComponent = nullptr;

	UPROPERTY()
	UTextureRenderTarget2D* RenderTarget = nullptr;

	bool bIsCapturing = false;
	bool bMediaReady = false;

	UFUNCTION()
	void HandleMediaOpened(FString OpenedUrl);

	UFUNCTION()
	void HandleMediaOpenFailed(FString FailedUrl);

	/** Cached device list from last enumeration. */
	TArray<FLlamaVideoDevice> CachedVideoDevices;

	/** Resolves the webcam URL: uses override if set, otherwise looks up by device index. */
	FString ResolveWebcamURL();

	void SetupWebcamCapture();
	void SetupSceneCapture();
	void CleanupCapture();

	UTexture2D* SnapshotFromMediaTexture();
	UTexture2D* SnapshotFromRenderTarget();
};
