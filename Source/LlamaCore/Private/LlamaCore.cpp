// Copyright 2025-current Getnamo.

#include "LlamaCore.h"

#define LOCTEXT_NAMESPACE "FLlamaCoreModule"

void FLlamaCoreModule::StartupModule()
{
	IModuleInterface::StartupModule();
}

void FLlamaCoreModule::ShutdownModule()
{
	IModuleInterface::ShutdownModule();
}

#undef LOCTEXT_NAMESPACE

IMPLEMENT_MODULE(FLlamaCoreModule, LlamaCore)