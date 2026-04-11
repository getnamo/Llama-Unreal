// Copyright 2025-current Getnamo.

#include "LlamaComponent.h"
#include "LlamaUtility.h"
#include "LlamaNative.h"
#include "LlamaAudioCaptureComponent.h"
#include "Engine/Texture2D.h"
#include "TextureResource.h"

ULlamaComponent::ULlamaComponent(const FObjectInitializer &ObjectInitializer)
    : UActorComponent(ObjectInitializer)
{
    LlamaNative = new FLlamaNative();

    // Register in consumer registry so Blueprint can wire us via AddConsumerComponent.
    // Done in constructor because LlamaNative pointer is stable for the component's lifetime.
    ILlamaAudioConsumer::RegisterComponent(this, LlamaNative);

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
    LlamaNative->OnMarkdownPartialGenerated = [this](const FString& Partial, EMarkdownStreamState State)
    {
        OnMarkdownPartialGenerated.Broadcast(Partial, State);
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
    ModelParams.Advanced.Output.PartialsSeparators.Add(TEXT("."));
    ModelParams.Advanced.Output.PartialsSeparators.Add(TEXT("?"));
    ModelParams.Advanced.Output.PartialsSeparators.Add(TEXT("!"));
}

ULlamaComponent::~ULlamaComponent()
{
	ILlamaAudioConsumer::UnregisterComponent(this);

	if (LlamaNative)
	{
		delete LlamaNative;
		LlamaNative = nullptr;
	}
}

void ULlamaComponent::Activate(bool bReset)
{
    Super::Activate(bReset);

    if (ModelParams.bAutoLoadModelOnStartup)
    {
        LoadModel(true);
    }

    //Always sync template in case audio source attaches later
    LlamaNative->AudioPromptTemplate = AudioPromptTemplate;

    if (AudioSource)
    {
        AudioSource->AddConsumer(LlamaNative);
    }
}

void ULlamaComponent::Deactivate()
{
    if (AudioSource)
    {
        AudioSource->RemoveConsumer(LlamaNative);
    }

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

void ULlamaComponent::RemoveLastNTokens(int32 TokenCount)
{
    LlamaNative->RemoveLastNTokens(TokenCount);
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

void ULlamaComponent::InsertTemplateImagePrompt(UTexture2D* Image, const FString& Text, EChatTemplateRole Role, bool bAddAssistantBOS, bool bGenerateReply)
{
    if (!Image || !Image->GetPlatformData() || Image->GetPlatformData()->Mips.Num() == 0)
    {
        OnError.Broadcast(TEXT("Invalid or null texture passed to InsertTemplateImagePrompt"), 52);
        return;
    }
    if (!LlamaNative->IsMultimodalLoaded())
    {
        OnError.Broadcast(TEXT("Multimodal projector not loaded. Set MmprojPath in ModelParams before calling LoadModel."), 50);
        return;
    }
    if (!LlamaNative->SupportsVision())
    {
        OnError.Broadcast(TEXT("Vision not supported by loaded multimodal model"), 55);
        return;
    }

    // Read pixels from texture on game thread
    FTexturePlatformData* PlatformData = Image->GetPlatformData();
    const int32 Width = PlatformData->SizeX;
    const int32 Height = PlatformData->SizeY;

    const void* RawData = PlatformData->Mips[0].BulkData.LockReadOnly();

    TArray<uint8> RGBData;
    RGBData.SetNum(Width * Height * 3);

    // Convert BGRA -> RGB
    const uint8* Src = static_cast<const uint8*>(RawData);
    for (int32 i = 0; i < Width * Height; i++)
    {
        RGBData[i * 3 + 0] = Src[i * 4 + 2]; // R
        RGBData[i * 3 + 1] = Src[i * 4 + 1]; // G
        RGBData[i * 3 + 2] = Src[i * 4 + 0]; // B
    }
    PlatformData->Mips[0].BulkData.Unlock();

    // Build multimodal prompt
    FLlamaMultimodalPrompt Prompt;
    Prompt.Prompt = Text.Contains(TEXT("<__media__>")) ? Text : FString::Printf(TEXT("<__media__>\n%s"), *Text);
    Prompt.Role = Role;
    Prompt.bAddAssistantBOS = bAddAssistantBOS;
    Prompt.bGenerateReply = bGenerateReply;

    FLlamaMediaEntry Entry;
    Entry.MediaType = ELlamaMediaType::Image;
    Entry.ImageRGBData = MoveTemp(RGBData);
    Entry.ImageWidth = Width;
    Entry.ImageHeight = Height;
    Prompt.MediaEntries.Add(MoveTemp(Entry));

    LlamaNative->InsertMultimodalPrompt(Prompt);
}

void ULlamaComponent::InsertTemplateImagePromptFromFile(const FString& ImagePath, const FString& Text, EChatTemplateRole Role, bool bAddAssistantBOS, bool bGenerateReply)
{
    if (!LlamaNative->IsMultimodalLoaded())
    {
        OnError.Broadcast(TEXT("Multimodal projector not loaded. Set MmprojPath in ModelParams before calling LoadModel."), 50);
        return;
    }
    if (!LlamaNative->SupportsVision())
    {
        OnError.Broadcast(TEXT("Vision not supported by loaded multimodal model"), 55);
        return;
    }

    FLlamaMultimodalPrompt Prompt;
    Prompt.Prompt = Text.Contains(TEXT("<__media__>")) ? Text : FString::Printf(TEXT("<__media__>\n%s"), *Text);
    Prompt.Role = Role;
    Prompt.bAddAssistantBOS = bAddAssistantBOS;
    Prompt.bGenerateReply = bGenerateReply;

    FLlamaMediaEntry Entry;
    Entry.MediaType = ELlamaMediaType::Image;
    Entry.FilePath = ImagePath;
    Prompt.MediaEntries.Add(MoveTemp(Entry));

    LlamaNative->InsertMultimodalPrompt(Prompt);
}

void ULlamaComponent::InsertTemplateAudioPrompt(const TArray<float>& PCMAudio, const FString& Text, EChatTemplateRole Role, bool bAddAssistantBOS, bool bGenerateReply)
{
    if (!LlamaNative->IsMultimodalLoaded())
    {
        OnError.Broadcast(TEXT("Multimodal projector not loaded. Set MmprojPath in ModelParams before calling LoadModel."), 50);
        return;
    }
    if (!LlamaNative->SupportsAudio())
    {
        OnError.Broadcast(TEXT("Audio not supported by loaded multimodal model"), 56);
        return;
    }

    FLlamaMultimodalPrompt Prompt;
    Prompt.Prompt = Text.Contains(TEXT("<__media__>")) ? Text : FString::Printf(TEXT("<__media__>\n%s"), *Text);
    Prompt.Role = Role;
    Prompt.bAddAssistantBOS = bAddAssistantBOS;
    Prompt.bGenerateReply = bGenerateReply;

    FLlamaMediaEntry Entry;
    Entry.MediaType = ELlamaMediaType::Audio;
    Entry.AudioPCMData = PCMAudio;
    Prompt.MediaEntries.Add(MoveTemp(Entry));

    LlamaNative->InsertMultimodalPrompt(Prompt);
}

void ULlamaComponent::InsertMultimodalPrompt(const FLlamaMultimodalPrompt& Prompt)
{
    if (!LlamaNative->IsMultimodalLoaded())
    {
        OnError.Broadcast(TEXT("Multimodal projector not loaded. Set MmprojPath in ModelParams before calling LoadModel."), 50);
        return;
    }

    LlamaNative->InsertMultimodalPrompt(Prompt);
}

bool ULlamaComponent::IsMultimodalLoaded() const
{
    return LlamaNative->IsMultimodalLoaded();
}

bool ULlamaComponent::SupportsVision() const
{
    return LlamaNative->SupportsVision();
}

bool ULlamaComponent::SupportsAudio() const
{
    return LlamaNative->SupportsAudio();
}

int32 ULlamaComponent::GetAudioSampleRate() const
{
    return LlamaNative->GetAudioSampleRate();
}

void ULlamaComponent::SetAudioPromptTemplate(const FString& NewTemplate)
{
    AudioPromptTemplate = NewTemplate;
    LlamaNative->AudioPromptTemplate = NewTemplate;
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
