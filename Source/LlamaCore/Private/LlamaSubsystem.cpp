// Copyright 2025-current Getnamo.

#include "LlamaSubsystem.h"

#include "LlamaDualBackend.h"
#include "LlamaNative.h"
#include "LlamaUtility.h"
#include "Embedding/VectorDatabase.h"

void ULlamaSubsystem::Initialize(FSubsystemCollectionBase& Collection)
{
    Super::Initialize(Collection);

    // Sentence-ending separators so OnPartialGenerated fires per-sentence during streaming.
    // Note: matcher uses Sep[0], so only single-character entries are effective.
    auto& Seps = ModelParams.Advanced.Output.PartialsSeparators;
    Seps.Add(TEXT("."));
    Seps.Add(TEXT("?"));
    Seps.Add(TEXT("!"));
    Seps.Add(TEXT("\n"));     // paragraph / line break
    Seps.Add(TEXT("…")); // horizontal ellipsis
    Seps.Add(TEXT("。")); // CJK full stop
    Seps.Add(TEXT("？")); // fullwidth question mark
    Seps.Add(TEXT("！")); // fullwidth exclamation mark
    Seps.Add(TEXT("।")); // Devanagari danda
    Seps.Add(TEXT("؟")); // Arabic question mark

    Backend = new FLlamaDualBackend();
    Backend->ModelParams = ModelParams;
    Backend->Initialize();

    // Subsystems have no TickComponent — let FLlamaNative own its own ticker so native
    // callbacks drain on the game thread without an outer pumping it.
    if (Backend->GetLlamaNative())
    {
        Backend->GetLlamaNative()->AddTicker();
    }

    WireBackendCallbacks();
}

void ULlamaSubsystem::Deinitialize()
{
    if (Backend)
    {
        if (Backend->GetLlamaNative())
        {
            Backend->GetLlamaNative()->RemoveTicker();
        }
        delete Backend;
        Backend = nullptr;
    }
    Super::Deinitialize();
}

void ULlamaSubsystem::WireBackendCallbacks()
{
    if (!Backend) return;

    Backend->OnTokenGenerated = [this](const FString& Token)
    {
        OnTokenGenerated.Broadcast(Token);
    };
    Backend->OnPartialGenerated = [this](const FString& Partial)
    {
        OnPartialGenerated.Broadcast(Partial);
    };
    Backend->OnMarkdownPartialGenerated = [this](const FString& Partial, EMarkdownStreamState State)
    {
        OnMarkdownPartialGenerated.Broadcast(Partial, State);
    };
    Backend->OnResponseGenerated = [this](const FString& Response)
    {
        OnResponseGenerated.Broadcast(Response);
    };
    Backend->OnPromptProcessed = [this](int32 Tokens, EChatTemplateRole Role, float Speed)
    {
        OnPromptProcessed.Broadcast(Tokens, Role, Speed);
    };
    Backend->OnEndOfStream = [this](bool bStopSeq, float Tps)
    {
        OnEndOfStream.Broadcast(bStopSeq, Tps);
    };
    Backend->OnContextReset = [this]()
    {
        OnContextReset.Broadcast();
    };
    Backend->OnModelLoaded = [this](const FString& ModelName)
    {
        OnModelLoaded.Broadcast(ModelName);
    };
    Backend->OnError = [this](const FString& Err, int32 Code)
    {
        OnError.Broadcast(Err, Code);
    };
    Backend->OnEmbeddings = [this](const TArray<float>& Embeddings, const FString& Source)
    {
        OnEmbeddings.Broadcast(Embeddings, Source);
    };
    Backend->OnAllEmbeddingsGenerated = [this](const TArray<FString>& Sources)
    {
        OnAllEmbeddingsGenerated.Broadcast(Sources);
    };
    Backend->OnModelStateChanged = [this](const FLLMModelState& Updated)
    {
        ModelState = Updated;
    };
}

// ── Loading ─────────────────────────────────────────────────────────────────

void ULlamaSubsystem::LoadModel(bool bForceReload)
{
    if (!Backend) return;
    Backend->ModelParams = ModelParams;
    Backend->Endpoint = Endpoint;
    Backend->bUseRemote = bUseRemote;
    Backend->LoadModel(bForceReload);
}

void ULlamaSubsystem::UnloadModel()
{
    if (Backend) Backend->UnloadModel();
}

bool ULlamaSubsystem::IsModelLoaded() const
{
    return Backend && Backend->IsModelLoaded();
}

void ULlamaSubsystem::SetUseRemote(bool bNewUseRemote)
{
    if (!Backend) { bUseRemote = bNewUseRemote; return; }
    Backend->ModelParams = ModelParams;
    Backend->Endpoint = Endpoint;
    Backend->SetUseRemote(bNewUseRemote);
    bUseRemote = Backend->bUseRemote;
}

// ── Chat ────────────────────────────────────────────────────────────────────

void ULlamaSubsystem::ResetContextHistory(bool bKeepSystemPrompt)
{
    if (Backend) Backend->ResetContextHistory(bKeepSystemPrompt);
}

void ULlamaSubsystem::RebuildContextFromHistory(const FStructuredChatHistory& History)
{
    if (Backend) Backend->RebuildContextFromHistory(History);
}

void ULlamaSubsystem::RemoveLastAssistantReply()
{
    if (Backend) Backend->RemoveLastReply();
}

void ULlamaSubsystem::RemoveLastUserInput()
{
    if (Backend) Backend->RemoveLastUserInput();
}

void ULlamaSubsystem::RemoveLastNTokens(int32 TokenCount)
{
    if (Backend) Backend->RemoveLastNTokens(TokenCount);
}

void ULlamaSubsystem::InsertTemplatedPrompt(const FString& Text, EChatTemplateRole Role,
                                            bool bAddAssistantBOS, bool bGenerateReply)
{
    FLlamaChatPrompt Prompt;
    Prompt.Prompt = Text;
    Prompt.Role = Role;
    Prompt.bAddAssistantBOS = bAddAssistantBOS;
    Prompt.bGenerateReply = bGenerateReply;
    InsertTemplatedPromptStruct(Prompt);
}

void ULlamaSubsystem::InsertTemplatedPromptStruct(const FLlamaChatPrompt& ChatPrompt)
{
    if (!Backend) return;
    Backend->ModelParams = ModelParams;
    Backend->Endpoint = Endpoint;
    Backend->InsertTemplatedPrompt(ChatPrompt);
}

void ULlamaSubsystem::InsertRawPrompt(const FString& Text, bool bGenerateReply)
{
    if (!Backend) return;
    Backend->ModelParams = ModelParams;
    Backend->Endpoint = Endpoint;
    Backend->InsertRawPrompt(Text, bGenerateReply);
}

void ULlamaSubsystem::ImpersonateTemplatedPrompt(const FLlamaChatPrompt& ChatPrompt)
{
    if (!Backend) return;
    Backend->ModelParams = ModelParams;
    Backend->ImpersonateTemplatedPrompt(ChatPrompt);
}

void ULlamaSubsystem::ImpersonateTemplatedToken(const FString& Token, EChatTemplateRole Role, bool bEoS)
{
    if (Backend) Backend->ImpersonateTemplatedToken(Token, Role, bEoS);
}

FString ULlamaSubsystem::WrapPromptForRole(const FString& Text, EChatTemplateRole Role, const FString& Template)
{
    return Backend ? Backend->WrapPromptForRole(Text, Role, Template) : Text;
}

void ULlamaSubsystem::StopGeneration()
{
    if (Backend) Backend->StopGeneration();
}

void ULlamaSubsystem::ResumeGeneration()
{
    if (Backend) Backend->ResumeGeneration();
}

FString ULlamaSubsystem::RawContextHistory()
{
    return ModelState.ContextHistory;
}

FStructuredChatHistory ULlamaSubsystem::GetStructuredChatHistory()
{
    return ModelState.ChatHistory;
}

// ── Multimodal ──────────────────────────────────────────────────────────────

void ULlamaSubsystem::InsertTemplateImagePrompt(UTexture2D* Image, const FString& Text,
                                                EChatTemplateRole Role, bool bAddAssistantBOS, bool bGenerateReply)
{
    if (!Backend) return;
    Backend->ModelParams = ModelParams;
    Backend->Endpoint = Endpoint;
    Backend->InsertTemplateImagePromptFromTexture(Image, Text, Role, bAddAssistantBOS, bGenerateReply);
}

void ULlamaSubsystem::InsertTemplateImagePromptFromFile(const FString& ImagePath, const FString& Text,
                                                        EChatTemplateRole Role, bool bAddAssistantBOS, bool bGenerateReply)
{
    if (!Backend) return;
    Backend->ModelParams = ModelParams;
    Backend->Endpoint = Endpoint;
    Backend->InsertTemplateImagePromptFromFile(ImagePath, Text, Role, bAddAssistantBOS, bGenerateReply);
}

void ULlamaSubsystem::InsertTemplateAudioPrompt(const TArray<float>& PCMAudio, const FString& Text,
                                                EChatTemplateRole Role, bool bAddAssistantBOS, bool bGenerateReply)
{
    if (!Backend) return;
    Backend->ModelParams = ModelParams;
    Backend->Endpoint = Endpoint;
    Backend->InsertTemplateAudioPrompt(PCMAudio, Text, Role, bAddAssistantBOS, bGenerateReply);
}

void ULlamaSubsystem::InsertMultimodalPrompt(const FLlamaMultimodalPrompt& Prompt)
{
    if (!Backend) return;
    Backend->ModelParams = ModelParams;
    Backend->Endpoint = Endpoint;
    Backend->InsertMultimodalPrompt(Prompt);
}

bool ULlamaSubsystem::IsMultimodalLoaded() const { return Backend && Backend->IsMultimodalLoaded(); }
bool ULlamaSubsystem::SupportsVision() const     { return Backend && Backend->SupportsVision(); }
bool ULlamaSubsystem::SupportsAudio() const      { return Backend && Backend->SupportsAudio(); }
int32 ULlamaSubsystem::GetAudioSampleRate() const { return Backend ? Backend->GetAudioSampleRate() : 16000; }

// ── Embedding ───────────────────────────────────────────────────────────────

void ULlamaSubsystem::GeneratePromptEmbeddingsForText(const FString& Text)
{
    if (!Backend) return;
    Backend->ModelParams = ModelParams;
    Backend->GeneratePromptEmbeddingsForText(Text);
}

void ULlamaSubsystem::GeneratePromptEmbeddingsForTexts(const TArray<FString>& Texts)
{
    if (!Backend) return;
    Backend->ModelParams = ModelParams;
    Backend->GeneratePromptEmbeddingsForTexts(Texts);
}

int32 ULlamaSubsystem::GetEmbeddingDimension() const
{
    return Backend ? Backend->GetEmbeddingDimension() : 0;
}

void ULlamaSubsystem::EmbedTextsAsync(const TArray<FString>& Texts,
    TFunction<void(const TArray<TArray<float>>&, const TArray<FString>&)> OnDone)
{
    if (!Backend)
    {
        if (OnDone) OnDone(TArray<TArray<float>>(), TArray<FString>());
        return;
    }
    Backend->ModelParams = ModelParams;
    Backend->EmbedTextsAsync(Texts, MoveTemp(OnDone));
}

// ── Diagnostics ─────────────────────────────────────────────────────────────

float ULlamaSubsystem::TestVectorSearch()
{
    FVectorDatabase VectorDb;
    VectorDb.Params.Dimensions = 16;
    VectorDb.Params.MaxElements = 200;
    return VectorDb.BasicsTest();
}
