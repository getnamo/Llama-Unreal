// 2023 (c) Mika Pi, Modifications Getnamo

#pragma once

#include <CoreMinimal.h>
#include <Modules/ModuleManager.h>

class FLlamaCoreModule final : public IModuleInterface
{
public:
  virtual void StartupModule() override;
  virtual void ShutdownModule() override;
};
