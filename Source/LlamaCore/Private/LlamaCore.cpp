// Copyright 2023 Mika Pi, Modifications 2025-current Getnamo

#include "LlamaCore.h"

#define LOCTEXT_NAMESPACE "FLlamaCoreModule"

void FLlamaCoreModule::StartupModule()
{
	/*GgmlDllHandle = FPlatformProcess::GetDllHandle(TEXT("ggml.dll"));
	if (GgmlDllHandle == nullptr)
	{
		UE_LOG(LogTemp, Error, TEXT("Failed to load ggml.dll"));
	}

	LlamaDllHandle = FPlatformProcess::GetDllHandle(TEXT("llama.dll"));
	if (LlamaDllHandle == nullptr)
	{
		UE_LOG(LogTemp, Error, TEXT("Failed to load llama.dll"));
	}*/	
	

	//llama_backend_init();
	IModuleInterface::StartupModule();
}

void FLlamaCoreModule::ShutdownModule()
{
	IModuleInterface::ShutdownModule();

	/*if (LlamaDllHandle != nullptr)
	{
		FPlatformProcess::FreeDllHandle(LlamaDllHandle);
		LlamaDllHandle = nullptr;
	}
	if (GgmlDllHandle != nullptr)
	{
		FPlatformProcess::FreeDllHandle(GgmlDllHandle);
		GgmlDllHandle = nullptr;
	}*/

	//llama_backend_free();
}

#undef LOCTEXT_NAMESPACE

IMPLEMENT_MODULE(FLlamaCoreModule, LlamaCore)
DEFINE_LOG_CATEGORY(LlamaLog);