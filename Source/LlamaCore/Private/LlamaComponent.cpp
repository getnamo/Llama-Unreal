// Copyright 2025-current Getnamo.

#include "LlamaComponent.h"
#include "LlamaNative.h"

ULlamaComponent::ULlamaComponent(const FObjectInitializer &ObjectInitializer)
    : UActorComponent(ObjectInitializer)
{
    LlamaNative = new FLlamaNative();

    PrimaryComponentTick.bCanEverTick = true;
    PrimaryComponentTick.bStartWithTickEnabled = true;

    //All sentence ending formatting.
    ModelParams.Advanced.PartialsSeparators.Add(TEXT("."));
    ModelParams.Advanced.PartialsSeparators.Add(TEXT("?"));
    ModelParams.Advanced.PartialsSeparators.Add(TEXT("!"));
}

ULlamaComponent::~ULlamaComponent()
{
	if (LlamaNative)
	{
		delete LlamaNative;
		LlamaNative = nullptr;
	}
}

void ULlamaComponent::Activate(bool bReset)
{
    Super::Activate(bReset);
    LoadModel();
}

void ULlamaComponent::Deactivate()
{
    Super::Deactivate();
}

void ULlamaComponent::TickComponent(float DeltaTime,
                                    ELevelTick TickType,
                                    FActorComponentTickFunction* ThisTickFunction)
{
    Super::TickComponent(DeltaTime, TickType, ThisTickFunction);
}

void ULlamaComponent::InsertTemplatedPrompt(const FString& Prompt)
{
    //todo: add support for adjusting the role...

    LlamaNative->InsertPrompt(Prompt);
}

void ULlamaComponent::LoadModel()
{
    LlamaNative->OnResponseGenerated = [this](const FString& Response)
    {
        OnResponseGenerated.Broadcast(Response);
    };

    LlamaNative->OnTokenGenerated = [this](const FString& Token)
    {
        OnTokenGenerated.Broadcast(Token);
    };

    LlamaNative->OnModelLoaded = [this](const FString& ModelPath)
    {
        //Todo: we need model name from path...
        OnModelLoaded.Broadcast(ModelPath);
    };

    LlamaNative->SetModelParams(ModelParams);
    LlamaNative->LoadModel();
}

void ULlamaComponent::UnloadModel()
{
    LlamaNative->UnloadModel();
}

void ULlamaComponent::ResetContextHistory()
{
    //todo:implement
}

void ULlamaComponent::InsertRawPrompt(const FString& Text)
{
}

void ULlamaComponent::UserImpersonateText(const FString& Text, EChatTemplateRole Role, bool bIsEos)
{
    FString CombinedText = Text;

    /**
    //Check last role, ensure we give ourselves an assistant role if we haven't yet.
    if (ModelState.LastRole != Role)
    {
        CombinedText = GetRolePrefix(Role) + Text;

        //Modify the role
        ModelState.LastRole = Role;
    }

    //If this was the last text in the stream, auto-wrap suffix
    if (bIsEos)
    {
        CombinedText += ModelParams.ChatTemplate.CommonSuffix + ModelParams.ChatTemplate.Delimiter;
    }

    TokenCallbackInternal(CombinedText, ModelState.ContextLength + CombinedText.Len());

    */
}

FString ULlamaComponent::WrapPromptForRole(const FString& Text, EChatTemplateRole Role, const FString& Template)
{
    return FString();
}

void ULlamaComponent::StopGeneration()
{
    LlamaNative->StopGeneration();
}

void ULlamaComponent::ResumeGeneration()
{
    LlamaNative->ResumeGeneration();
}

FString ULlamaComponent::RawContextHistory()
{
    return LlamaNative->RawContextHistory();
}

FStructuredChatHistory ULlamaComponent::GetStructuredHistory()
{
    FStructuredChatHistory Chat;
    LlamaNative->GetStructuredChatHistory(Chat);

    return Chat;
}
