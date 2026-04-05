// Copyright 2025-current Getnamo.

#include "LlamaMediaCaptureTypes.h"
#include "LlamaUtility.h"

TMap<FString, FLlamaAudioProcessorRegistry::FFactoryFunc>& FLlamaAudioProcessorRegistry::GetRegistry()
{
    static TMap<FString, FFactoryFunc> Registry;
    return Registry;
}

void FLlamaAudioProcessorRegistry::Register(const FString& Name, FFactoryFunc Factory)
{
    GetRegistry().Add(Name, MoveTemp(Factory));
    UE_LOG(LlamaLog, Log, TEXT("FLlamaAudioProcessorRegistry: Registered '%s'"), *Name);
}

ILlamaAudioProcessor* FLlamaAudioProcessorRegistry::Create(const FString& Name, const FString& ConfigPath)
{
    if (FFactoryFunc* Found = GetRegistry().Find(Name))
    {
        return (*Found)(ConfigPath);
    }
    UE_LOG(LlamaLog, Warning, TEXT("FLlamaAudioProcessorRegistry: '%s' not registered"), *Name);
    return nullptr;
}

bool FLlamaAudioProcessorRegistry::IsRegistered(const FString& Name)
{
    return GetRegistry().Contains(Name);
}

// ─── ILlamaAudioConsumer Component Registry ─────────────────────────────────

TMap<UActorComponent*, ILlamaAudioConsumer*>& ILlamaAudioConsumer::GetComponentRegistry()
{
    static TMap<UActorComponent*, ILlamaAudioConsumer*> Registry;
    return Registry;
}

void ILlamaAudioConsumer::RegisterComponent(UActorComponent* Component, ILlamaAudioConsumer* Consumer)
{
    if (Component && Consumer)
    {
        GetComponentRegistry().Add(Component, Consumer);
    }
}

void ILlamaAudioConsumer::UnregisterComponent(UActorComponent* Component)
{
    if (Component)
    {
        GetComponentRegistry().Remove(Component);
    }
}

ILlamaAudioConsumer* ILlamaAudioConsumer::FindConsumerForComponent(UActorComponent* Component)
{
    if (ILlamaAudioConsumer** Found = GetComponentRegistry().Find(Component))
    {
        return *Found;
    }
    return nullptr;
}
