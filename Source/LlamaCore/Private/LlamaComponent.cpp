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

    //Forward tick to llama so it can process the game thread callbacks
    LlamaNative->OnTick(DeltaTime);
}

void ULlamaComponent::InsertTemplatedPrompt(const FString& Prompt, EChatTemplateRole Role, bool bAddAssistantBOS, bool bGenerateReply)
{
    FLlamaChatPrompt ChatPrompt;
    ChatPrompt.Prompt = Prompt;
    ChatPrompt.Role = Role;
    ChatPrompt.bAddAssistantBOS = bAddAssistantBOS;
    ChatPrompt.bGenerateReply = bGenerateReply;

    LlamaNative->InsertTemplatedPrompt(ChatPrompt, [this, ChatPrompt](const FString& Response)
    {
        if (!ChatPrompt.bGenerateReply)
        {
            //OnPromptProcessed.Broadcast()
        }
        else
        {
            OnResponseGenerated.Broadcast(Response);
            OnEndOfStream.Broadcast(true, ModelState.LastTokenGenerationSpeed);
        }
    });
}

void ULlamaComponent::LoadModel()
{
    LlamaNative->OnModelStateChanged = [this](const FLLMModelState& UpdatedModelState)
    {
        ModelState = UpdatedModelState;
    };

    LlamaNative->OnTokenGenerated = [this](const FString& Token)
    {
        OnTokenGenerated.Broadcast(Token);
    };
    
    LlamaNative->OnPartialGenerated = [this](const FString& Partial)
    {
        OnPartialGenerated.Broadcast(Partial);
    };

    //move to function callbacks?
    LlamaNative->OnPromptProcessed = [this](int32 TokensProcessed, EChatTemplateRole Role, float Speed)
    {
        OnPromptProcessed.Broadcast(TokensProcessed, Role, Speed);
    };

    LlamaNative->SetModelParams(ModelParams);
    LlamaNative->LoadModel([this](const FString& ModelPath, int32 StatusCode)
    {
        if (ModelParams.bAutoInsertSystemPromptOnLoad)
        {
            InsertTemplatedPrompt(ModelParams.SystemPrompt, EChatTemplateRole::System, false, false);
        }

        //Todo: we need model name from path...
        OnModelLoaded.Broadcast(ModelPath);
    });
}

void ULlamaComponent::UnloadModel()
{
    LlamaNative->UnloadModel([this](int32 StatusCode)
    {
        if (StatusCode != 0)
        {
            UE_LOG(LlamaLog, Warning, TEXT("UnloadModel return error code: %d"), StatusCode);
        }
    });
}

void ULlamaComponent::ResetContextHistory(bool bKeepSystemPrompt)
{
    LlamaNative->ResetContextHistory(bKeepSystemPrompt);
}

void ULlamaComponent::RemoveLastAssistantReply()
{
    LlamaNative->RemoveLastReply();
}

void ULlamaComponent::RemoveLastUserInput()
{
    LlamaNative->RemoveLastUserInput();
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
    return LlamaNative->WrapPromptForRole(Text, Role, Template);
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
    FString History;
    LlamaNative->RawContextHistory(History);
    return History;
}

FStructuredChatHistory ULlamaComponent::GetStructuredChatHistory()
{
    FStructuredChatHistory Chat;
    LlamaNative->GetStructuredChatHistory(Chat);
    return Chat;
}
