// Copyright 2025-current Getnamo.

#include "LlamaComponent.h"
#include "LlamaUtility.h"
#include "LlamaNative.h"
#include "Embedding/VectorDatabase.h"

ULlamaComponent::ULlamaComponent(const FObjectInitializer &ObjectInitializer)
    : UActorComponent(ObjectInitializer)
{
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

    LlamaNative->OnResponseGenerated = [this](const FString& Response)
    {
        OnResponseGenerated.Broadcast(Response);
        OnEndOfStream.Broadcast(true, ModelState.LastTokenGenerationSpeed);
    };

    LlamaNative->OnPartialGenerated = [this](const FString& Partial)
    {
        OnPartialGenerated.Broadcast(Partial);
    };
    LlamaNative->OnPromptProcessed = [this](int32 TokensProcessed, EChatTemplateRole Role, float Speed)
    {
        OnPromptProcessed.Broadcast(TokensProcessed, Role, Speed);
    };
    LlamaNative->OnError = [this](const FString& ErrorMessage, int32 ErrorCode)
    {
        OnError.Broadcast(ErrorMessage, ErrorCode);
    };

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
    if (VectorDb)
    {
        delete VectorDb;
        VectorDb = nullptr;
    }
}

void ULlamaComponent::Activate(bool bReset)
{
    Super::Activate(bReset);

    if (ModelParams.bAutoLoadModelOnStartup)
    {
        LoadModel(true);
    }
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
    LlamaNative->OnGameThreadTick(DeltaTime);
}

void ULlamaComponent::InsertTemplatedPrompt(const FString& Text, EChatTemplateRole Role, bool bAddAssistantBOS, bool bGenerateReply)
{
    FLlamaChatPrompt ChatPrompt;
    ChatPrompt.Prompt = Text;
    ChatPrompt.Role = Role;
    ChatPrompt.bAddAssistantBOS = bAddAssistantBOS;
    ChatPrompt.bGenerateReply = bGenerateReply;
    InsertTemplatedPromptStruct(ChatPrompt);
}

void ULlamaComponent::InsertTemplatedPromptStruct(const FLlamaChatPrompt& ChatPrompt)
{
    LlamaNative->InsertTemplatedPrompt(ChatPrompt);/*, [this, ChatPrompt](const FString& Response));
     {
        if (ChatPrompt.bGenerateReply)
        {
            OnResponseGenerated.Broadcast(Response);
            OnEndOfStream.Broadcast(true, ModelState.LastTokenGenerationSpeed);
        }
    });*/
}

void ULlamaComponent::InsertRawPrompt(const FString& Text, bool bGenerateReply)
{
    LlamaNative->InsertRawPrompt(Text, bGenerateReply); /*, [this, bGenerateReply](const FString& Response)
    {
        if (bGenerateReply)
        {
            OnResponseGenerated.Broadcast(Response);
            OnEndOfStream.Broadcast(true, ModelState.LastTokenGenerationSpeed);
        }
    });*/
}

void ULlamaComponent::LoadModel(bool bForceReload)
{
    LlamaNative->SetModelParams(ModelParams);
    LlamaNative->LoadModel(bForceReload, [this](const FString& ModelPath, int32 StatusCode)
    {
        //We errored, the emit will happen before we reach here so just exit
        if (StatusCode !=0)
        {
            return;
        }

        OnModelLoaded.Broadcast(ModelPath);
    });
}

void ULlamaComponent::UnloadModel()
{
    LlamaNative->UnloadModel([this](int32 StatusCode)
    {
        //this pretty much should never get called, just in case: emit.
        if (StatusCode != 0)
        {
            FString ErrorMessage = FString::Printf(TEXT("UnloadModel returned error code: %d"), StatusCode);
            UE_LOG(LlamaLog, Warning, TEXT("%s"), *ErrorMessage);
            OnError.Broadcast(ErrorMessage, StatusCode);
        }
    });
}

bool ULlamaComponent::IsModelLoaded()
{
    return ModelState.bModelIsLoaded;
}

void ULlamaComponent::ResetContextHistory(bool bKeepSystemPrompt)
{
    LlamaNative->ResetContextHistory(bKeepSystemPrompt);
}

void ULlamaComponent::RemoveLastAssistantReply()
{
    if (ModelParams.bRemoteMode)
    {
        //modify state only
        int32 Count = ModelState.ChatHistory.History.Num();
        if (Count >0)
        {
            ModelState.ChatHistory.History.RemoveAt(Count - 1);
        }
    }
    else
    {
        LlamaNative->RemoveLastReply();
    }
}

void ULlamaComponent::RemoveLastUserInput()
{
    if (ModelParams.bRemoteMode)
    {
        //modify state only
        int32 Count = ModelState.ChatHistory.History.Num();
        if (Count > 1)
        {
            ModelState.ChatHistory.History.RemoveAt(Count - 1);
            ModelState.ChatHistory.History.RemoveAt(Count - 2);
        }
    }
    else
    {
        LlamaNative->RemoveLastUserInput();
    }
}


void ULlamaComponent::ImpersonateTemplatedPrompt(const FLlamaChatPrompt& ChatPrompt)
{
    LlamaNative->SetModelParams(ModelParams);

    LlamaNative->ImpersonateTemplatedPrompt(ChatPrompt);
}

void ULlamaComponent::ImpersonateTemplatedToken(const FString& Token, EChatTemplateRole Role, bool bEoS)
{
    LlamaNative->ImpersonateTemplatedToken(Token, Role, bEoS);
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
    return ModelState.ContextHistory;
}

FStructuredChatHistory ULlamaComponent::GetStructuredChatHistory()
{
    return ModelState.ChatHistory;
}

void ULlamaComponent::GeneratePromptEmbeddingsForText(const FString& Text)
{
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

void ULlamaComponent::TestVectorDB()
{
    //roughly 10 sample sentences from https://randomwordgenerator.com/sentence.php
    TArray<FString> Sentences = 
    {
        TEXT("They desperately needed another drummer since the current one only knew how to play bongos."),
        TEXT("Poison ivy grew through the fence they said was impenetrable."),
        TEXT("Most shark attacks occur about 10 feet from the beach since that's where the people are."),
        TEXT("Mothers spend months of their lives waiting on their children."),
        TEXT("He realized there had been several deaths on this road, but his concern rose when he saw the exact number."),
        TEXT("Just go ahead and press that button."),
        TEXT("The worst thing about being at the top of the career ladder is that there's a long way to fall."),
        TEXT("The fish listened intently to what the frogs had to say."),
        TEXT("My dentist tells me that chewing bricks is very bad for your teeth."),
        TEXT("She wondered what his eyes were saying beneath his mirrored sunglasses."),
    };

    if (!VectorDb)
    {
        VectorDb = new FVectorDatabase();
        VectorDb->InitializeDB();
    }

    UE_LOG(LogTemp, Log, TEXT("VectorDB Pre"));
    //VectorDb->BasicsTest();
    //UE_LOG(LogTemp, Log, TEXT("VectorDB Post"));

    TempN = 0;
    int32 Total = Sentences.Num();
    FString QueryTest = Sentences[2];   //should be "Most shark attacks..."

    //used for query later, guaranteed due to piping to be processed first
    LlamaNative->GetPromptEmbeddings(QueryTest, [this](const TArray<float>& Embeddings, const FString& SourceText)
    {
        TempEmbeddings = Embeddings;
    });

    for (FString& Sentence : Sentences)
    {
        UE_LOG(LogTemp, Log, TEXT("Queuing embed for <%s>"), *Sentence);
        LlamaNative->GetPromptEmbeddings(Sentence, [this, Sentence, Total, QueryTest](const TArray<float>& Embeddings, const FString& SourceText)
        {
            UE_LOG(LogTemp, Log, TEXT("Got embed for <%s>"), *Sentence);
            TArray<float> SafeEmbeddings = Embeddings;
            VectorDb->AddVectorEmbeddingStringPair(SafeEmbeddings, SourceText);
            TempN++;

            //Last one?
            if (TempN == Total)
            {
                //Try searching
                UE_LOG(LogTemp, Log, TEXT("Embedded all sentences, doing a search..."));
                
                //we need to run Query test through embed..
                FString Nearest = VectorDb->FindNearestString(TempEmbeddings);

                UE_LOG(LogTemp, Log, TEXT("Nearest to <%s> is <%s>"), *QueryTest, *Nearest);
            }
        });   
    }
}
