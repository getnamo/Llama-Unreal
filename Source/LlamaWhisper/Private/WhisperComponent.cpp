// Copyright 2025-current Getnamo.

#include "WhisperComponent.h"
#include "WhisperNative.h"

UWhisperComponent::UWhisperComponent(const FObjectInitializer& ObjectInitializer)
	: Super(ObjectInitializer)
{
	PrimaryComponentTick.bCanEverTick = true;

	//bAutoActivate = true;

	WhisperNative = new FWhisperNative();

	// Wire up native callbacks to component delegates (all fired on the game thread)

	WhisperNative->OnTranscriptionResult = [this](const FString& Text, bool bIsFinal)
	{
		ModelState.LastTranscription = Text;
		OnTranscriptionResult.Broadcast(Text, bIsFinal);
	};

	WhisperNative->OnModelLoaded = [this](const FString& ModelPath)
	{
		ModelState.bModelLoaded = true;
		OnModelLoaded.Broadcast(ModelPath);
	};

	WhisperNative->OnError = [this](const FString& ErrorMessage)
	{
		OnError.Broadcast(ErrorMessage);
	};

	WhisperNative->OnVADStateChanged = [this](bool bIsSpeech)
	{
		ModelState.bVADSpeechDetected = bIsSpeech;
		OnVADStateChanged.Broadcast(bIsSpeech);
	};

	WhisperNative->OnTranscribingStateChanged = [this](bool bTranscribing)
	{
		ModelState.bIsTranscribing = bTranscribing;
	};
}

UWhisperComponent::~UWhisperComponent()
{
	delete WhisperNative;
	WhisperNative = nullptr;
}

void UWhisperComponent::BeginPlay()
{
	Super::BeginPlay();
	Activate(true);
}

void UWhisperComponent::Activate(bool bReset)
{
	Super::Activate(bReset);

	WhisperNative->SetModelParams(ModelParams);
	WhisperNative->SetStreamParams(StreamParams);

	if (ModelParams.bAutoLoadModelOnStartup)
	{
		LoadModel(bReset);
	}
}

void UWhisperComponent::Deactivate()
{
	if (WhisperNative->IsMicrophoneCaptureActive())
	{
		WhisperNative->StopMicrophoneCapture();
	}

	WhisperNative->UnloadModel();
	ModelState.bModelLoaded     = false;
	ModelState.bMicrophoneActive = false;

	Super::Deactivate();
}

void UWhisperComponent::TickComponent(float DeltaTime, ELevelTick TickType,
                                       FActorComponentTickFunction* ThisTickFunction)
{
	Super::TickComponent(DeltaTime, TickType, ThisTickFunction);
	WhisperNative->OnGameThreadTick(DeltaTime);
}

// ---------------------------------------------------------------------------
// Blueprint API
// ---------------------------------------------------------------------------

void UWhisperComponent::LoadModel(bool bForceReload)
{
	WhisperNative->SetModelParams(ModelParams);

	WhisperNative->LoadModel(bForceReload,
		[this](const FString& ModelPath, int32 StatusCode)
		{
			if (StatusCode == 0)
			{
				ModelState.bModelLoaded = true;
			}
		});
}

void UWhisperComponent::UnloadModel()
{
	WhisperNative->UnloadModel([this](int32)
	{
		ModelState.bModelLoaded = false;
	});
}

bool UWhisperComponent::IsModelLoaded() const
{
	return ModelState.bModelLoaded;
}

void UWhisperComponent::TranscribeAudioData(TArray<float> PCMSamples, int32 SampleRate)
{
	WhisperNative->SetModelParams(ModelParams);
	WhisperNative->TranscribeAudioData(MoveTemp(PCMSamples), SampleRate);
}

void UWhisperComponent::TranscribeWaveFile(const FString& FilePath)
{
	WhisperNative->SetModelParams(ModelParams);
	WhisperNative->TranscribeWaveFile(FilePath);
}

void UWhisperComponent::StartMicrophoneCapture()
{
	WhisperNative->SetModelParams(ModelParams);
	WhisperNative->SetStreamParams(StreamParams);
	WhisperNative->StartMicrophoneCapture();
	ModelState.bMicrophoneActive = true;
}

void UWhisperComponent::StopMicrophoneCapture()
{
	WhisperNative->StopMicrophoneCapture();
	ModelState.bMicrophoneActive = false;
}

bool UWhisperComponent::IsMicrophoneCaptureActive() const
{
	return ModelState.bMicrophoneActive;
}
