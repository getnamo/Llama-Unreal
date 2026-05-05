// Copyright 2025-current Getnamo.

#include "LlamaSubsystem.h"
#include "HAL/PlatformTime.h"
#include "Tickable.h"
#include "LlamaNative.h"
#include "LlamaUtility.h"
#include "Embedding/VectorDatabase.h"

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
    LlamaNative->OnResponseGenerated = [this](const FString& Response)
    {
        OnResponseGenerated.Broadcast(Response);
        OnEndOfStream.Broadcast(true, ModelState.LastTokenGenerationSpeed);
    };
    LlamaNative->OnError = [this](const FString& ErrorMessage, int32 ErrorCode)
    {
        OnError.Broadcast(ErrorMessage, ErrorCode);
    };

    //All sentence ending formatting.
    ModelParams.Advanced.Output.PartialsSeparators.Add(TEXT("."));
    ModelParams.Advanced.Output.PartialsSeparators.Add(TEXT("?"));
    ModelParams.Advanced.Output.PartialsSeparators.Add(TEXT("!"));
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
    LlamaNative->InsertTemplatedPrompt(ChatPrompt);/*, [this, ChatPrompt](const FString& Response)
    {
        if (ChatPrompt.bGenerateReply)
        {
            OnResponseGenerated.Broadcast(Response);
            OnEndOfStream.Broadcast(true, ModelState.LastTokenGenerationSpeed);
        }
    });*/
}

void ULlamaSubsystem::InsertRawPrompt(const FString& Text, bool bGenerateReply)
{
    LlamaNative->InsertRawPrompt(Text, bGenerateReply);/*, [this, bGenerateReply](const FString& Response)
    {
        if (bGenerateReply)
        {
            OnResponseGenerated.Broadcast(Response);
            OnEndOfStream.Broadcast(true, ModelState.LastTokenGenerationSpeed);
        }
    })*/;
}

void ULlamaSubsystem::LoadModel(bool bForceReload)
{
    //Sync gt params
    LlamaNative->SetModelParams(ModelParams);

    //If ticker isn't active right now, start it. This will stay active until subsystem gets destroyed.
    if (!LlamaNative->IsNativeTickerActive())
    {
        LlamaNative->AddTicker();
    }

    LlamaNative->LoadModel(bForceReload, [this](const FString& ModelPath, int32 StatusCode)
    {
        //We errored, the emit will happen before we reach here so just exit
        if (StatusCode != 0)
        {
            return;
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
            FString ErrorMessage = FString::Printf(TEXT("UnloadModel return error code: %d"), StatusCode);
            UE_LOG(LlamaLog, Warning, TEXT("%s"), *ErrorMessage);
            OnError.Broadcast(ErrorMessage, StatusCode);
        }
    });
}

bool ULlamaSubsystem::IsModelLoaded()
{
    return ModelState.bModelIsLoaded;
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

void ULlamaSubsystem::GeneratePromptEmbeddingsForText(const FString& Text)
{
    if (!LlamaNative) return;
    if (!ModelParams.Advanced.bEmbeddingMode)
    {
        UE_LOG(LlamaLog, Warning, TEXT("Model is not in embedding mode, cannot generate embeddings."));
        return;
    }
    LlamaNative->GetPromptEmbeddings(Text, [this](const TArray<float>& Embeddings, const FString& SourceText)
    {
        OnEmbeddings.Broadcast(Embeddings, SourceText);
    });
}

void ULlamaSubsystem::GeneratePromptEmbeddingsForTexts(const TArray<FString>& Texts)
{
    if (!LlamaNative) return;
    if (!ModelParams.Advanced.bEmbeddingMode)
    {
        UE_LOG(LlamaLog, Warning, TEXT("Model is not in embedding mode, cannot generate embeddings."));
        return;
    }
    LlamaNative->GetPromptEmbeddingsBatch(Texts,
        [this](const TArray<float>& Embeddings, const FString& SourceText)
        {
            OnEmbeddings.Broadcast(Embeddings, SourceText);
        },
        [this](const TArray<TArray<float>>& /*All*/, const TArray<FString>& AllSourceTexts)
        {
            OnAllEmbeddingsGenerated.Broadcast(AllSourceTexts);
        });
}

int32 ULlamaSubsystem::GetEmbeddingDimension() const
{
    return LlamaNative ? LlamaNative->GetEmbeddingDimension() : 0;
}

float ULlamaSubsystem::TestVectorSearch()
{
    FVectorDatabase VectorDb;
    VectorDb.Params.Dimensions = 16;
    VectorDb.Params.MaxElements = 200;
    const float Recall = VectorDb.BasicsTest();
    return Recall;
}

FString ULlamaSubsystem::RawContextHistory()
{
    return ModelState.ContextHistory;
}

FStructuredChatHistory ULlamaSubsystem::GetStructuredChatHistory()
{
    return ModelState.ChatHistory;
}