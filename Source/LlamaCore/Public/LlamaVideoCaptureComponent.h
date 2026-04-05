// Copyright 2025-current Getnamo.

#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "LlamaMediaCaptureTypes.h"

#include "LlamaVideoCaptureComponent.generated.h"

class UMediaPlayer;
class UMediaSource;
class UMediaTexture;
class USceneCaptureComponent2D;
class UTextureRenderTarget2D;
class UTexture2D;

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

	/** Webcam device URL. Platform-dependent. Common examples:
	 *  Windows: "cam://0" or device name. Leave empty for default device. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Video Capture",
		meta = (EditCondition = "CaptureSource == EVideoCaptureSource::Webcam"))
	FString WebcamURL;

	/** Whether to start capture automatically on BeginPlay. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Video Capture")
	bool bAutoStartCapture = false;

	// -----------------------------------------------------------------------
	// Blueprint API
	// -----------------------------------------------------------------------

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

	void SetupWebcamCapture();
	void SetupSceneCapture();
	void CleanupCapture();

	UTexture2D* SnapshotFromMediaTexture();
	UTexture2D* SnapshotFromRenderTarget();
};
