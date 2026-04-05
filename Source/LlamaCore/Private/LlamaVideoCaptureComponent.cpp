// Copyright 2025-current Getnamo.

#include "LlamaVideoCaptureComponent.h"
#include "LlamaUtility.h"

#include "MediaPlayer.h"
#include "MediaTexture.h"
#include "MediaSource.h"
#include "Engine/TextureRenderTarget2D.h"
#include "Engine/Texture2D.h"
#include "Engine/Canvas.h"
#include "Components/SceneCaptureComponent2D.h"
#include "Kismet/KismetRenderingLibrary.h"
#include "TextureResource.h"
#include "RenderUtils.h"

ULlamaVideoCaptureComponent::ULlamaVideoCaptureComponent(const FObjectInitializer& ObjectInitializer)
	: Super(ObjectInitializer)
{
	PrimaryComponentTick.bCanEverTick = false;
}

void ULlamaVideoCaptureComponent::BeginPlay()
{
	Super::BeginPlay();

	if (bAutoStartCapture)
	{
		StartCapture();
	}
}

void ULlamaVideoCaptureComponent::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
	StopCapture();
	Super::EndPlay(EndPlayReason);
}

void ULlamaVideoCaptureComponent::StartCapture()
{
	if (bIsCapturing)
	{
		return;
	}

	if (CaptureSource == EVideoCaptureSource::Webcam)
	{
		SetupWebcamCapture();
	}
	else
	{
		SetupSceneCapture();
	}
}

void ULlamaVideoCaptureComponent::StopCapture()
{
	if (!bIsCapturing)
	{
		return;
	}

	CleanupCapture();
	bIsCapturing = false;
}

bool ULlamaVideoCaptureComponent::IsCaptureActive() const
{
	return bIsCapturing;
}

void ULlamaVideoCaptureComponent::SetupWebcamCapture()
{
	// Create MediaPlayer
	MediaPlayer = NewObject<UMediaPlayer>(this);
	MediaPlayer->SetLooping(true);

	// Create MediaTexture
	CachedMediaTexture = NewObject<UMediaTexture>(this);
	CachedMediaTexture->AutoClear = true;
	CachedMediaTexture->SetMediaPlayer(MediaPlayer);
	CachedMediaTexture->UpdateResource();

	// Open the webcam URL
	FString URL = WebcamURL;
	if (URL.IsEmpty())
	{
		// Platform default webcam
		// The actual URL format depends on the platform and installed media framework plugins
		URL = TEXT("vidcap://0");
	}

	if (MediaPlayer->OpenUrl(URL))
	{
		bIsCapturing = true;
		UE_LOG(LlamaLog, Log, TEXT("ULlamaVideoCaptureComponent: Opened webcam: %s"), *URL);
	}
	else
	{
		UE_LOG(LlamaLog, Warning, TEXT("ULlamaVideoCaptureComponent: Failed to open webcam: %s"), *URL);
		CleanupCapture();
	}
}

void ULlamaVideoCaptureComponent::SetupSceneCapture()
{
	// Try to find an existing SceneCaptureComponent2D on the actor
	if (!SceneCaptureComponent && GetOwner())
	{
		SceneCaptureComponent = GetOwner()->FindComponentByClass<USceneCaptureComponent2D>();
	}

	if (!SceneCaptureComponent)
	{
		UE_LOG(LlamaLog, Warning, TEXT("ULlamaVideoCaptureComponent: No SceneCaptureComponent2D found. Use SetSceneCaptureComponent() to assign one."));
		return;
	}

	// Create render target
	RenderTarget = NewObject<UTextureRenderTarget2D>(this);
	RenderTarget->InitAutoFormat(CaptureWidth, CaptureHeight);
	RenderTarget->UpdateResourceImmediate(true);

	SceneCaptureComponent->TextureTarget = RenderTarget;
	bIsCapturing = true;

	UE_LOG(LlamaLog, Log, TEXT("ULlamaVideoCaptureComponent: Scene capture started (%dx%d)"), CaptureWidth, CaptureHeight);
}

void ULlamaVideoCaptureComponent::CleanupCapture()
{
	if (MediaPlayer)
	{
		MediaPlayer->Close();
		MediaPlayer = nullptr;
	}
	CachedMediaTexture = nullptr;
	MediaSource = nullptr;

	if (SceneCaptureComponent && RenderTarget)
	{
		if (SceneCaptureComponent->TextureTarget == RenderTarget)
		{
			SceneCaptureComponent->TextureTarget = nullptr;
		}
	}
	RenderTarget = nullptr;
}

UTexture2D* ULlamaVideoCaptureComponent::SnapshotFrame()
{
	if (!bIsCapturing)
	{
		return nullptr;
	}

	if (CaptureSource == EVideoCaptureSource::Webcam)
	{
		return SnapshotFromMediaTexture();
	}
	else
	{
		return SnapshotFromRenderTarget();
	}
}

UTexture2D* ULlamaVideoCaptureComponent::SnapshotFromMediaTexture()
{
	if (!CachedMediaTexture || !MediaPlayer || !MediaPlayer->IsPlaying())
	{
		return nullptr;
	}

	// Media textures do not support direct ReadPixels.
	// Draw the media texture into a temporary render target, then read back from that.
	UTextureRenderTarget2D* TempRT = NewObject<UTextureRenderTarget2D>();
	TempRT->InitAutoFormat(CaptureWidth, CaptureHeight);
	TempRT->UpdateResourceImmediate(true);

	// Draw media texture to render target via Canvas
	UCanvas* Canvas = nullptr;
	FVector2D CanvasSize;
	FDrawToRenderTargetContext Context;
	UKismetRenderingLibrary::BeginDrawCanvasToRenderTarget(this, TempRT, Canvas, CanvasSize, Context);
	if (Canvas)
	{
		Canvas->K2_DrawTexture(CachedMediaTexture, FVector2D::ZeroVector, CanvasSize, FVector2D::ZeroVector, FVector2D::UnitVector);
	}
	UKismetRenderingLibrary::EndDrawCanvasToRenderTarget(this, Context);

	// Read back pixels from the temporary render target
	TArray<FColor> Pixels;
	FTextureRenderTargetResource* RTResource = TempRT->GameThread_GetRenderTargetResource();
	if (!RTResource || !RTResource->ReadPixels(Pixels))
	{
		return nullptr;
	}

	int32 Width = TempRT->SizeX;
	int32 Height = TempRT->SizeY;

	if (Pixels.Num() != Width * Height)
	{
		return nullptr;
	}

	// Create output texture
	UTexture2D* OutTexture = UTexture2D::CreateTransient(Width, Height, PF_B8G8R8A8);
	if (!OutTexture)
	{
		return nullptr;
	}

	void* MipData = OutTexture->GetPlatformData()->Mips[0].BulkData.Lock(LOCK_READ_WRITE);
	FMemory::Memcpy(MipData, Pixels.GetData(), Pixels.Num() * sizeof(FColor));
	OutTexture->GetPlatformData()->Mips[0].BulkData.Unlock();
	OutTexture->UpdateResource();

	return OutTexture;
}

UTexture2D* ULlamaVideoCaptureComponent::SnapshotFromRenderTarget()
{
	if (!RenderTarget || !RenderTarget->GameThread_GetRenderTargetResource())
	{
		return nullptr;
	}

	// Trigger an immediate capture
	if (SceneCaptureComponent)
	{
		SceneCaptureComponent->CaptureScene();
	}

	TArray<FColor> Pixels;
	if (!RenderTarget->GameThread_GetRenderTargetResource()->ReadPixels(Pixels))
	{
		return nullptr;
	}

	int32 Width = RenderTarget->SizeX;
	int32 Height = RenderTarget->SizeY;

	if (Pixels.Num() != Width * Height)
	{
		return nullptr;
	}

	UTexture2D* OutTexture = UTexture2D::CreateTransient(Width, Height, PF_B8G8R8A8);
	if (!OutTexture)
	{
		return nullptr;
	}

	void* MipData = OutTexture->GetPlatformData()->Mips[0].BulkData.Lock(LOCK_READ_WRITE);
	FMemory::Memcpy(MipData, Pixels.GetData(), Pixels.Num() * sizeof(FColor));
	OutTexture->GetPlatformData()->Mips[0].BulkData.Unlock();
	OutTexture->UpdateResource();

	return OutTexture;
}

UMediaTexture* ULlamaVideoCaptureComponent::GetMediaTexture() const
{
	return CachedMediaTexture;
}

UTextureRenderTarget2D* ULlamaVideoCaptureComponent::GetRenderTarget() const
{
	return RenderTarget;
}

void ULlamaVideoCaptureComponent::SetSceneCaptureComponent(USceneCaptureComponent2D* InSceneCapture)
{
	SceneCaptureComponent = InSceneCapture;
}
