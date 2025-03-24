// Copyright 2025-current Getnamo.

#include "LlamaSubsystem.h"
#include "HAL/PlatformTime.h"
#include "Tickable.h"
#include "LlamaNative.h"

void ULlamaSubsystem::Initialize(FSubsystemCollectionBase& Collection)
{
	Super::Initialize(Collection);
    LlamaNative = new FLlamaNative();

    //Hookup native callbacks
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
    LlamaNative->OnPromptProcessed = [this](int32 TokensProcessed, EChatTemplateRole Role, float Speed)
    {
        OnPromptProcessed.Broadcast(TokensProcessed, Role, Speed);
    };
    LlamaNative->OnError = [this](const FString& ErrorMessage)
    {
        OnError.Broadcast(ErrorMessage);
    };

    //All sentence ending formatting.
    ModelParams.Advanced.PartialsSeparators.Add(TEXT("."));
    ModelParams.Advanced.PartialsSeparators.Add(TEXT("?"));
    ModelParams.Advanced.PartialsSeparators.Add(TEXT("!"));
}

void ULlamaSubsystem::Deinitialize()
{
	if (LlamaNative)
	{
		delete LlamaNative;
		LlamaNative = nullptr;
	}

    Super::Deinitialize();
}

void ULlamaSubsystem::InsertTemplatedPrompt(const FString& Text, EChatTemplateRole Role, bool bAddAssistantBOS, bool bGenerateReply)
{
    FLlamaChatPrompt ChatPrompt;
    ChatPrompt.Prompt = Text;
    ChatPrompt.Role = Role;
    ChatPrompt.bAddAssistantBOS = bAddAssistantBOS;
    ChatPrompt.bGenerateReply = bGenerateReply;
    InsertTemplatedPromptStruct(ChatPrompt);
}

void ULlamaSubsystem::InsertTemplatedPromptStruct(const FLlamaChatPrompt& ChatPrompt)
{
    LlamaNative->InsertTemplatedPrompt(ChatPrompt, [this, ChatPrompt](const FString& Response)
    {
        if (ChatPrompt.bGenerateReply)
        {
            OnResponseGenerated.Broadcast(Response);
            OnEndOfStream.Broadcast(true, ModelState.LastTokenGenerationSpeed);
        }
    });
}

void ULlamaSubsystem::InsertRawPrompt(const FString& Text, bool bGenerateReply)
{
    LlamaNative->InsertRawPrompt(Text, bGenerateReply, [this, bGenerateReply](const FString& Response)
    {
        if (bGenerateReply)
        {
            OnResponseGenerated.Broadcast(Response);
            OnEndOfStream.Broadcast(true, ModelState.LastTokenGenerationSpeed);
        }
    });
}

void ULlamaSubsystem::LoadModel()
{
    //Sync gt params
    LlamaNative->SetModelParams(ModelParams);

    //If ticker isn't active right now, start it. This will stay active until
    if (!LlamaNative->IsNativeTickerActive())
    {
        LlamaNative->AddTicker();
    }

    LlamaNative->LoadModel([this](const FString& ModelPath, int32 StatusCode)
    {
        if (ModelParams.bAutoInsertSystemPromptOnLoad)
        {
            InsertTemplatedPrompt(ModelParams.SystemPrompt, EChatTemplateRole::System, false, false);
        }

        OnModelLoaded.Broadcast(ModelPath);
    });


}

void ULlamaSubsystem::UnloadModel()
{
    LlamaNative->UnloadModel([this](int32 StatusCode)
    {
        if (StatusCode != 0)
        {
            UE_LOG(LlamaLog, Warning, TEXT("UnloadModel return error code: %d"), StatusCode);
        }
    });
}

void ULlamaSubsystem::ResetContextHistory(bool bKeepSystemPrompt)
{
    LlamaNative->ResetContextHistory(bKeepSystemPrompt);
}

void ULlamaSubsystem::RemoveLastAssistantReply()
{
    LlamaNative->RemoveLastReply();
}

void ULlamaSubsystem::RemoveLastUserInput()
{
    LlamaNative->RemoveLastUserInput();
}

void ULlamaSubsystem::StopGeneration()
{
    LlamaNative->StopGeneration();
}

void ULlamaSubsystem::ResumeGeneration()
{
    LlamaNative->ResumeGeneration();
}

FString ULlamaSubsystem::RawContextHistory()
{
    return ModelState.ContextHistory;
}

FStructuredChatHistory ULlamaSubsystem::GetStructuredChatHistory()
{
    return ModelState.ChatHistory;
}