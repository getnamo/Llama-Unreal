// Copyright 2025-current Getnamo.

#include "LlamaComponent.h"

#include "LlamaDualBackend.h"
#include "LlamaUtility.h"
#include "LlamaAudioCaptureComponent.h"

ULlamaComponent::ULlamaComponent(const FObjectInitializer& ObjectInitializer)
    : UActorComponent(ObjectInitializer)
{
    // Sentence-ending separators so OnPartialGenerated fires per-sentence during streaming
    // (every per-token sync of ModelParams to the backend carries these forward).
    // Note: matcher uses Sep[0], so only single-character entries are effective.
    auto& Seps = ModelParams.Advanced.Output.PartialsSeparators;
    Seps.Add(TEXT("."));
    Seps.Add(TEXT("?"));
    Seps.Add(TEXT("!"));
    Seps.Add(TEXT("\n"));     // paragraph / line break (lists, headings, multi-line output)
    Seps.Add(TEXT("…")); // … horizontal ellipsis
    Seps.Add(TEXT("。")); // 。 CJK full stop
    Seps.Add(TEXT("？")); // ？ fullwidth question mark
    Seps.Add(TEXT("！")); // ！ fullwidth exclamation mark
    Seps.Add(TEXT("।")); // ।  Devanagari danda
    Seps.Add(TEXT("؟")); // ؟ Arabic question mark

    Backend = new FLlamaDualBackend();
    SyncBackendConfig();
    Backend->Initialize();
    WireBackendCallbacks();

    // Backend implements ILlamaAudioConsumer so it can route audio segments to the active
    // backend (local FLlamaNative or remote WAV-encode + chat-completion path).
    ILlamaAudioConsumer::RegisterComponent(this, Backend);

    PrimaryComponentTick.bCanEverTick = true;
    PrimaryComponentTick.bStartWithTickEnabled = true;
}

ULlamaComponent::~ULlamaComponent()
{
    ILlamaAudioConsumer::UnregisterComponent(this);
    if (Backend)
    {
        delete Backend;
        Backend = nullptr;
    }
}

void ULlamaComponent::WireBackendCallbacks()
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

void ULlamaComponent::SyncBackendConfig()
{
    if (!Backend) return;
    Backend->ModelParams = ModelParams;
    Backend->Endpoint = Endpoint;
    Backend->bUseRemote = bUseRemote;
    Backend->AudioPromptTemplate = AudioPromptTemplate;
    Backend->bUseIncrementalKVSyncOnToggle = bUseIncrementalKVSyncOnToggle;
    Backend->bPreloadLocalWhenRemote = bPreloadLocalWhenRemote;
}

void ULlamaComponent::Activate(bool bReset)
{
    Super::Activate(bReset);

    SyncBackendConfig();

    if (Backend && AudioSource)
    {
        AudioSource->AddConsumer(Backend);
    }

    if (ModelParams.bAutoLoadModelOnStartup)
    {
        LoadModel(true);
    }
}

void ULlamaComponent::Deactivate()
{
    if (Backend && AudioSource)
    {
        AudioSource->RemoveConsumer(Backend);
    }
    Super::Deactivate();
}

void ULlamaComponent::TickComponent(float DeltaTime, ELevelTick TickType,
                                    FActorComponentTickFunction* ThisTickFunction)
{
    Super::TickComponent(DeltaTime, TickType, ThisTickFunction);
    if (Backend)
    {
        Backend->OnGameThreadTick(DeltaTime);
    }
}

// ── Loading ─────────────────────────────────────────────────────────────────

void ULlamaComponent::LoadModel(bool bForceReload)
{
    if (!Backend) return;
    SyncBackendConfig();
    Backend->LoadModel(bForceReload);
}

void ULlamaComponent::UnloadModel()
{
    if (Backend) Backend->UnloadModel();
}

bool ULlamaComponent::IsModelLoaded() const
{
    return Backend && Backend->IsModelLoaded();
}

void ULlamaComponent::SetUseRemote(bool bNewUseRemote)
{
    if (!Backend) { bUseRemote = bNewUseRemote; return; }
    SyncBackendConfig();
    Backend->SetUseRemote(bNewUseRemote);
    bUseRemote = Backend->bUseRemote;
}

// ── Chat ────────────────────────────────────────────────────────────────────

void ULlamaComponent::ResetContextHistory(bool bKeepSystemPrompt)
{
    if (Backend) Backend->ResetContextHistory(bKeepSystemPrompt);
}

void ULlamaComponent::RebuildContextFromHistory(const FStructuredChatHistory& History)
{
    if (Backend) Backend->RebuildContextFromHistory(History);
}

void ULlamaComponent::RemoveLastAssistantReply()
{
    if (Backend) Backend->RemoveLastReply();
}

void ULlamaComponent::RemoveLastUserInput()
{
    if (Backend) Backend->RemoveLastUserInput();
}

void ULlamaComponent::RemoveLastNTokens(int32 TokenCount)
{
    if (Backend) Backend->RemoveLastNTokens(TokenCount);
}

void ULlamaComponent::InsertTemplatedPrompt(const FString& Text, EChatTemplateRole Role,
                                            bool bAddAssistantBOS, bool bGenerateReply,
                                            const FString& AssistantPrefill)
{
    FLlamaChatPrompt Prompt;
    Prompt.Prompt = Text;
    Prompt.Role = Role;
    Prompt.bAddAssistantBOS = bAddAssistantBOS;
    Prompt.bGenerateReply = bGenerateReply;
    Prompt.AssistantPrefill = AssistantPrefill;
    InsertTemplatedPromptStruct(Prompt);
}

void ULlamaComponent::InsertTemplatedPromptStruct(const FLlamaChatPrompt& ChatPrompt)
{
    if (!Backend) return;
    SyncBackendConfig();
    Backend->InsertTemplatedPrompt(ChatPrompt);
}

void ULlamaComponent::InsertRawPrompt(const FString& Text, bool bGenerateReply)
{
    if (!Backend) return;
    SyncBackendConfig();
    Backend->InsertRawPrompt(Text, bGenerateReply);
}

void ULlamaComponent::ImpersonateTemplatedPrompt(const FLlamaChatPrompt& ChatPrompt)
{
    if (!Backend) return;
    SyncBackendConfig();
    Backend->ImpersonateTemplatedPrompt(ChatPrompt);
}

void ULlamaComponent::ImpersonateTemplatedToken(const FString& Token, EChatTemplateRole Role, bool bEoS)
{
    if (Backend) Backend->ImpersonateTemplatedToken(Token, Role, bEoS);
}

FString ULlamaComponent::WrapPromptForRole(const FString& Text, EChatTemplateRole Role, const FString& Template)
{
    return Backend ? Backend->WrapPromptForRole(Text, Role, Template) : Text;
}

void ULlamaComponent::StopGeneration()
{
    if (Backend) Backend->StopGeneration();
}

void ULlamaComponent::ResumeGeneration()
{
    if (Backend) Backend->ResumeGeneration();
}

FString ULlamaComponent::RawContextHistory()
{
    return ModelState.ContextHistory;
}

FStructuredChatHistory ULlamaComponent::GetStructuredChatHistory()
{
    return ModelState.ChatHistory;
}

// ── Multimodal ──────────────────────────────────────────────────────────────

void ULlamaComponent::InsertTemplateImagePrompt(UTexture2D* Image, const FString& Text,
                                                EChatTemplateRole Role, bool bAddAssistantBOS, bool bGenerateReply)
{
    if (!Backend) return;
    SyncBackendConfig();
    Backend->InsertTemplateImagePromptFromTexture(Image, Text, Role, bAddAssistantBOS, bGenerateReply);
}

void ULlamaComponent::InsertTemplateImagePromptFromFile(const FString& ImagePath, const FString& Text,
                                                        EChatTemplateRole Role, bool bAddAssistantBOS, bool bGenerateReply)
{
    if (!Backend) return;
    SyncBackendConfig();
    Backend->InsertTemplateImagePromptFromFile(ImagePath, Text, Role, bAddAssistantBOS, bGenerateReply);
}

void ULlamaComponent::InsertTemplateAudioPrompt(const TArray<float>& PCMAudio, const FString& Text,
                                                EChatTemplateRole Role, bool bAddAssistantBOS, bool bGenerateReply)
{
    if (!Backend) return;
    SyncBackendConfig();
    Backend->InsertTemplateAudioPrompt(PCMAudio, Text, Role, bAddAssistantBOS, bGenerateReply);
}

void ULlamaComponent::InsertMultimodalPrompt(const FLlamaMultimodalPrompt& Prompt)
{
    if (!Backend) return;
    SyncBackendConfig();
    Backend->InsertMultimodalPrompt(Prompt);
}

bool ULlamaComponent::IsMultimodalLoaded() const { return Backend && Backend->IsMultimodalLoaded(); }
bool ULlamaComponent::SupportsVision() const     { return Backend && Backend->SupportsVision(); }
bool ULlamaComponent::SupportsAudio() const      { return Backend && Backend->SupportsAudio(); }
int32 ULlamaComponent::GetAudioSampleRate() const { return Backend ? Backend->GetAudioSampleRate() : 16000; }

void ULlamaComponent::SetAudioPromptTemplate(const FString& NewTemplate)
{
    AudioPromptTemplate = NewTemplate;
    if (Backend) Backend->AudioPromptTemplate = NewTemplate;
}

// ── Embedding ───────────────────────────────────────────────────────────────

void ULlamaComponent::GeneratePromptEmbeddingsForText(const FString& Text)
{
    if (!Backend) return;
    SyncBackendConfig();
    Backend->GeneratePromptEmbeddingsForText(Text);
}

void ULlamaComponent::GeneratePromptEmbeddingsForTexts(const TArray<FString>& Texts)
{
    if (!Backend) return;
    SyncBackendConfig();
    Backend->GeneratePromptEmbeddingsForTexts(Texts);
}

int32 ULlamaComponent::GetEmbeddingDimension() const
{
    return Backend ? Backend->GetEmbeddingDimension() : 0;
}

void ULlamaComponent::EmbedTextsAsync(const TArray<FString>& Texts,
    TFunction<void(const TArray<TArray<float>>&, const TArray<FString>&)> OnDone)
{
    if (!Backend)
    {
        if (OnDone) OnDone(TArray<TArray<float>>(), TArray<FString>());
        return;
    }
    SyncBackendConfig();
    Backend->EmbedTextsAsync(Texts, MoveTemp(OnDone));
}
