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

void ULlamaComponent::InsertPrompt(const FString& Prompt)
{
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

void ULlamaComponent::UserImpersonateText(const FString& Text, EChatTemplateRole Role, bool bIsEos)
{
    FString CombinedText = Text;

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
}

FString ULlamaComponent::WrapPromptForRole(const FString& Content, EChatTemplateRole Role, bool AppendModelRolePrefix)
{
    FString FinalInputText = TEXT("");
    if (Role == EChatTemplateRole::User)
    {
        FinalInputText = ModelParams.ChatTemplate.User + ModelParams.ChatTemplate.Delimiter + Content + ModelParams.ChatTemplate.CommonSuffix + ModelParams.ChatTemplate.Delimiter;
    }
    else if (Role == EChatTemplateRole::Assistant)
    {
        FinalInputText = ModelParams.ChatTemplate.Assistant + ModelParams.ChatTemplate.Delimiter + Content + ModelParams.ChatTemplate.CommonSuffix + ModelParams.ChatTemplate.Delimiter;
    }
    else if (Role == EChatTemplateRole::System)
    {
        FinalInputText = ModelParams.ChatTemplate.System + ModelParams.ChatTemplate.Delimiter + Content + ModelParams.ChatTemplate.CommonSuffix + ModelParams.ChatTemplate.Delimiter;
    }
    else
    {
        return Content;
    }

    if (AppendModelRolePrefix) 
    {
        //Preset role reply
        FinalInputText += GetRolePrefix(EChatTemplateRole::Assistant);
    }

    return FinalInputText;
}

FString ULlamaComponent::GetRolePrefix(EChatTemplateRole Role)
{
    FString Prefix = TEXT("");

    if (Role != EChatTemplateRole::Unknown)
    {
        if (Role == EChatTemplateRole::Assistant)
        {
            Prefix += ModelParams.ChatTemplate.Assistant + ModelParams.ChatTemplate.Delimiter;
        }
        else if (Role == EChatTemplateRole::User)
        {
            Prefix += ModelParams.ChatTemplate.User + ModelParams.ChatTemplate.Delimiter;
        }
        else if (Role == EChatTemplateRole::System)
        {
            Prefix += ModelParams.ChatTemplate.System + ModelParams.ChatTemplate.Delimiter;
        }
    }
    return Prefix;
}

void ULlamaComponent::InsertPromptTemplated(const FString& Content, EChatTemplateRole Role)
{
    //done by default, TBD: remove
}

void ULlamaComponent::StopGeneration()
{
    LlamaNative->StopGeneration();
}

void ULlamaComponent::ResumeGeneration()
{
    //Not yet supported
}

FString ULlamaComponent::GetTemplateStrippedPrompt()
{
    FString CleanPrompt;
    
    CleanPrompt = ModelState.PromptHistory.Replace(*ModelParams.ChatTemplate.User, TEXT(""));
    CleanPrompt = CleanPrompt.Replace(*ModelParams.ChatTemplate.Assistant, TEXT(""));
    CleanPrompt = CleanPrompt.Replace(*ModelParams.ChatTemplate.System, TEXT(""));
    CleanPrompt = CleanPrompt.Replace(*ModelParams.ChatTemplate.CommonSuffix, TEXT(""));

    return CleanPrompt;
}

FStructuredChatMessage ULlamaComponent::FirstChatMessageInHistory(const FString& History, FString& Remainder)
{
    FStructuredChatMessage Message;
    Message.Role = EChatTemplateRole::Unknown;

    int32 StartIndex = INDEX_NONE;
    FString StartRole = TEXT("");
    int32 StartSystem = History.Find(ModelParams.ChatTemplate.System, ESearchCase::CaseSensitive, ESearchDir::FromStart, -1);
    int32 StartAssistant = History.Find(ModelParams.ChatTemplate.Assistant, ESearchCase::CaseSensitive, ESearchDir::FromStart, -1);
    int32 StartUser = History.Find(ModelParams.ChatTemplate.User, ESearchCase::CaseSensitive, ESearchDir::FromStart, -1);

    //Early exit
    if (StartSystem == INDEX_NONE &&
        StartAssistant == INDEX_NONE &&
        StartUser == INDEX_NONE)
    {
        //Failed end find
        Remainder = TEXT("");
        return Message;
    }

    //so they aren't the lowest (-1)
    if (StartSystem == INDEX_NONE)
    {
        StartSystem = INT32_MAX;
    }
    if (StartAssistant == INDEX_NONE)
    {
        StartAssistant = INT32_MAX;
    }
    if (StartUser == INDEX_NONE)
    {
        StartUser = INT32_MAX;
    }
    
    if (StartSystem <= StartAssistant &&
        StartSystem <= StartUser)
    {
        StartIndex = StartSystem;
        StartRole = ModelParams.ChatTemplate.System;
        Message.Role = EChatTemplateRole::System;
    }

    else if (
        StartUser <= StartAssistant &&
        StartUser <= StartSystem)
    {
        StartIndex = StartUser;
        StartRole = ModelParams.ChatTemplate.User;
        Message.Role = EChatTemplateRole::User;
    }

    else if (
        StartAssistant <= StartUser &&
        StartAssistant <= StartSystem)
    {
        StartIndex = StartAssistant;
        StartRole = ModelParams.ChatTemplate.Assistant;
        Message.Role = EChatTemplateRole::Assistant;
    }

    //Look for system role first
    if (StartIndex != INDEX_NONE)
    {
        const FString& CommonSuffix = ModelParams.ChatTemplate.CommonSuffix;

        StartIndex = StartIndex + StartRole.Len();

        int32 EndIndex = History.Find(CommonSuffix, ESearchCase::CaseSensitive, ESearchDir::FromStart, StartIndex);

        if (EndIndex != INDEX_NONE)
        {
            int32 Count = EndIndex - StartIndex;
            Message.Content = History.Mid(StartIndex, Count).TrimStartAndEnd();

            EndIndex = EndIndex + CommonSuffix.Len();

            Remainder = History.RightChop(EndIndex);
        }
        else
        {
            //No ending, assume all content belongs to this bit
            Message.Content = History.RightChop(StartIndex).TrimStartAndEnd();
            Remainder = TEXT("");
        }
    }
    return Message;
}

FStructuredChatHistory ULlamaComponent::GetStructuredHistory()
{
    FString WorkingHistory = ModelState.PromptHistory;
    FStructuredChatHistory Chat;


    while (!WorkingHistory.IsEmpty())
    {
        FStructuredChatMessage Message = FirstChatMessageInHistory(WorkingHistory, WorkingHistory);

        //Only add proper role messages
        if (Message.Role != EChatTemplateRole::Unknown)
        {
            Chat.History.Add(Message);
        }
    }
    return Chat;
}

EChatTemplateRole ULlamaComponent::LastRoleFromStructuredHistory()
{
    if (ModelState.ChatHistory.History.Num() > 0)
    {
        return ModelState.ChatHistory.History.Last().Role;
    }
    else
    {
        return EChatTemplateRole::Unknown;
    }
}

