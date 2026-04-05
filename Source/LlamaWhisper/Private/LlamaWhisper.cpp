// Copyright 2025-current Getnamo.

#include "LlamaWhisper.h"
#include "LlamaMediaCaptureTypes.h"
#include "AudioProcessors/LlamaSileroVAD.h"
#include "LlamaUtility.h"

#define LOCTEXT_NAMESPACE "FLlamaWhisperModule"

void FLlamaWhisperModule::StartupModule()
{
	FLlamaAudioProcessorRegistry::Register(TEXT("Silero"), [](const FString& ConfigPath) -> ILlamaAudioProcessor*
	{
		FLlamaSileroVAD* VAD = new FLlamaSileroVAD();
		if (!ConfigPath.IsEmpty())
		{
			if (!VAD->LoadModel(ConfigPath))
			{
				UE_LOG(LlamaLog, Warning, TEXT("Silero VAD model failed to load from: %s"), *ConfigPath);
			}
		}
		return VAD;
	});
}

void FLlamaWhisperModule::ShutdownModule()
{
}

#undef LOCTEXT_NAMESPACE

IMPLEMENT_MODULE(FLlamaWhisperModule, LlamaWhisper)
