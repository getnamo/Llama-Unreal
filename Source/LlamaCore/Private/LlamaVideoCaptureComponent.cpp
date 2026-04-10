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
#include "MediaCaptureSupport.h"
#include "IMediaCaptureSupport.h"

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

TArray<FLlamaVideoDevice> ULlamaVideoCaptureComponent::EnumerateVideoDevices()
{
	CachedVideoDevices.Reset();

	TArray<FMediaCaptureDeviceInfo> MediaDevices;
	MediaCaptureSupport::EnumerateVideoCaptureDevices(MediaDevices);

	for (const FMediaCaptureDeviceInfo& Device : MediaDevices)
	{
		FLlamaVideoDevice Entry;
		Entry.DisplayName = Device.DisplayName.ToString();
		Entry.URL = Device.Url;
		Entry.Info = Device.Info;
		CachedVideoDevices.Add(MoveTemp(Entry));
	}

	UE_LOG(LlamaLog, Log, TEXT("ULlamaVideoCaptureComponent: Found %d video capture device(s):"), CachedVideoDevices.Num());
	for (int32 i = 0; i < CachedVideoDevices.Num(); ++i)
	{
		UE_LOG(LlamaLog, Log, TEXT("  [%d] %s (%s)"), i, *CachedVideoDevices[i].DisplayName, *CachedVideoDevices[i].URL);
	}

	return CachedVideoDevices;
}

FLlamaVideoDevice ULlamaVideoCaptureComponent::GetSelectedDevice() const
{
	if (CachedVideoDevices.IsValidIndex(SelectedDeviceIndex))
	{
		return CachedVideoDevices[SelectedDeviceIndex];
	}
	return FLlamaVideoDevice();
}

FString ULlamaVideoCaptureComponent::ResolveWebcamURL()
{
	// Explicit override takes priority
	if (!WebcamURLOverride.IsEmpty())
	{
		return WebcamURLOverride;
	}

	// Enumerate if we haven't yet
	if (CachedVideoDevices.Num() == 0)
	{
		EnumerateVideoDevices();
	}

	// Use device at selected index
	if (CachedVideoDevices.IsValidIndex(SelectedDeviceIndex))
	{
		return CachedVideoDevices[SelectedDeviceIndex].URL;
	}

	// Fallback to platform default
	UE_LOG(LlamaLog, Warning, TEXT("ULlamaVideoCaptureComponent: No device at index %d, falling back to vidcap://0"), SelectedDeviceIndex);
	return TEXT("vidcap://0");
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
	// Create MediaPlayer — no looping for live capture streams
	MediaPlayer = NewObject<UMediaPlayer>(this);
	MediaPlayer->PlayOnOpen = true;

	// Create MediaTexture
	CachedMediaTexture = NewObject<UMediaTexture>(this);
	CachedMediaTexture->AutoClear = true;
	CachedMediaTexture->NewStyleOutput = true;
	CachedMediaTexture->SetMediaPlayer(MediaPlayer);
	CachedMediaTexture->UpdateResource();

	// Resolve the webcam URL from device index or override
	FString URL = ResolveWebcamURL();

	// Bind to OnMediaOpened to know when the stream is actually ready
	MediaPlayer->OnMediaOpened.AddDynamic(this, &ULlamaVideoCaptureComponent::HandleMediaOpened);
	MediaPlayer->OnMediaOpenFailed.AddDynamic(this, &ULlamaVideoCaptureComponent::HandleMediaOpenFailed);

	if (MediaPlayer->OpenUrl(URL))
	{
		bIsCapturing = true;
		UE_LOG(LlamaLog, Log, TEXT("ULlamaVideoCaptureComponent: Opening webcam: %s (waiting for media ready...)"), *URL);
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
		MediaPlayer->OnMediaOpened.RemoveAll(this);
		MediaPlayer->OnMediaOpenFailed.RemoveAll(this);
		MediaPlayer->Close();
		MediaPlayer = nullptr;
	}
	bMediaReady = false;
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

void ULlamaVideoCaptureComponent::HandleMediaOpened(FString OpenedUrl)
{
	if (!MediaPlayer)
	{
		return;
	}

	const int32 NumVideoTracks = MediaPlayer->GetNumTracks(EMediaPlayerTrack::Video);
	const int32 SelectedVideoTrack = MediaPlayer->GetSelectedTrack(EMediaPlayerTrack::Video);

	UE_LOG(LlamaLog, Log, TEXT("ULlamaVideoCaptureComponent: Webcam media opened: %s"), *OpenedUrl);
	UE_LOG(LlamaLog, Log, TEXT("  Video tracks: %d, Selected: %d, IsPlaying: %s, IsReady: %s"),
		NumVideoTracks, SelectedVideoTrack,
		MediaPlayer->IsPlaying() ? TEXT("true") : TEXT("false"),
		MediaPlayer->IsReady() ? TEXT("true") : TEXT("false"));

	// Ensure a video track is selected
	if (NumVideoTracks > 0 && SelectedVideoTrack == INDEX_NONE)
	{
		UE_LOG(LlamaLog, Log, TEXT("  No video track selected — selecting track 0"));
		MediaPlayer->SelectTrack(EMediaPlayerTrack::Video, 0);
	}

	// Log video dimensions if available
	if (NumVideoTracks > 0)
	{
		const FIntPoint Dims = MediaPlayer->GetVideoTrackDimensions(
			FMath::Max(SelectedVideoTrack, 0), INDEX_NONE);
		UE_LOG(LlamaLog, Log, TEXT("  Video dimensions: %dx%d"), Dims.X, Dims.Y);
	}

	// Force playback start — capture devices need explicit Play()
	MediaPlayer->Play();

	UE_LOG(LlamaLog, Log, TEXT("  Play() called. IsPlaying: %s"),
		MediaPlayer->IsPlaying() ? TEXT("true") : TEXT("false"));

	bMediaReady = true;
	OnWebcamReady.Broadcast();
}

void ULlamaVideoCaptureComponent::HandleMediaOpenFailed(FString FailedUrl)
{
	UE_LOG(LlamaLog, Warning, TEXT("ULlamaVideoCaptureComponent: Failed to open webcam media: %s"), *FailedUrl);
	bMediaReady = false;
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
	if (!CachedMediaTexture || !MediaPlayer)
	{
		return nullptr;
	}

	if (!bMediaReady)
	{
		UE_LOG(LlamaLog, Warning, TEXT("ULlamaVideoCaptureComponent: Webcam media not ready yet. Wait for capture to fully open."));
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
