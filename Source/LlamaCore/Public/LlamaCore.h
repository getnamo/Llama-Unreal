// Copyright 2023 Mika Pi, Modifications 2025-current Getnamo

#pragma once

#include <CoreMinimal.h>
#include <Modules/ModuleManager.h>

class FLlamaCoreModule final : public IModuleInterface
{
public:
  virtual void StartupModule() override;
  virtual void ShutdownModule() override;

private:
	//void* LlamaDllHandle = nullptr;
	//void* GgmlDllHandle = nullptr;
};