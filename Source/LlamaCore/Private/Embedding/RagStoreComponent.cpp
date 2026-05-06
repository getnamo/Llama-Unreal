// Copyright 2025-current Getnamo.

#include "Embedding/RagStoreComponent.h"
#include "LlamaComponent.h"
#include "LlamaUtility.h"
#include "GameFramework/Actor.h"

URagStoreComponent::URagStoreComponent()
{
    PrimaryComponentTick.bCanEverTick = false;

    // Mirror URagStore's defaults so the component's UPROPERTY editor view shows the
    // same recommended models out of the box (otherwise FLLMModelParams' default
    // ./model.gguf would shadow them when the component syncs config to the store).
    EmbeddingModelParams.PathToModel              = TEXT("./nomic-embed-text-v1.5.Q4_K_M.gguf");
    EmbeddingModelParams.MaxContextLength         = 2048;
    EmbeddingModelParams.GPULayers                = 99;
    EmbeddingModelParams.MaxBatchLength           = 2048;
    EmbeddingModelParams.bAutoLoadModelOnStartup  = false;
    EmbeddingModelParams.bAutoInsertSystemPromptOnLoad = false;
    EmbeddingModelParams.SystemPrompt             = TEXT("");
    EmbeddingModelParams.Advanced.bEmbeddingMode  = true;

    AnswerModelParams.PathToModel                 = TEXT("./google_gemma-3-4b-it-Q4_K_L.gguf");
    AnswerModelParams.MaxContextLength            = 8192;
    AnswerModelParams.GPULayers                   = 99;
    AnswerModelParams.MaxBatchLength              = 1024;
    AnswerModelParams.bAutoLoadModelOnStartup     = false;
    AnswerModelParams.bAutoInsertSystemPromptOnLoad = true;
    AnswerModelParams.SystemPrompt                = TEXT("You are a helpful AI assistant.");
    AnswerModelParams.Advanced.Thinking.bEnableThinking            = false;
    AnswerModelParams.Advanced.Thinking.bStripThinkingFromResponse = true;
}

void URagStoreComponent::BeginPlay()
{
    Super::BeginPlay();
    EnsureStore();

    // Back-compat / discovery shortcut: if neither an internal embedder model nor an
    // explicit ExternalEmbedder is configured, look for a sibling ULlamaComponent on
    // the same actor that's already in embedding mode.
    if (EmbeddingModelParams.PathToModel.IsEmpty() && ExternalEmbedder == nullptr)
    {
        ExternalEmbedder = AutoDiscoverEmbedder();
    }

    SyncStoreConfig();

    if (bAutoInitializeOnBeginPlay)
    {
        // One-call chain: serialized model loads + auto-Initialize + OnRagPipelineReady.
        // Replaces the old polling-via-TickComponent pattern, which could call Initialize()
        // before the embedder finished loading and concurrently kick off both backend
        // loads (causing crashes during ggml/Vulkan device init).
        Store->LoadAndInitialize();
    }
}

void URagStoreComponent::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
    if (Store) { Store->OnIngestComplete.RemoveAll(this); }
    Super::EndPlay(EndPlayReason);
}

void URagStoreComponent::TickComponent(float DeltaTime, ELevelTick TickType,
                                       FActorComponentTickFunction* ThisTickFunction)
{
    Super::TickComponent(DeltaTime, TickType, ThisTickFunction);
    // Tick disabled by default — the auto-init flow uses delegates now.
}

void URagStoreComponent::EnsureStore()
{
    if (!Store)
    {
        Store = NewObject<URagStore>(this);
        HookStoreDelegates();
    }
}

void URagStoreComponent::HookStoreDelegates()
{
    if (!Store) { return; }
    Store->OnIngestComplete.AddDynamic(this,         &URagStoreComponent::HandleStoreIngestComplete);
    Store->OnRagPipelineReady.AddDynamic(this,       &URagStoreComponent::HandleStoreRagPipelineReady);

    Store->OnAskRetrievedChunks.AddDynamic(this,     &URagStoreComponent::HandleStoreAskRetrieved);
    Store->OnAskTokenGenerated.AddDynamic(this,      &URagStoreComponent::HandleStoreAskToken);
    Store->OnAskPartialGenerated.AddDynamic(this,    &URagStoreComponent::HandleStoreAskPartial);
    Store->OnAskMarkdownPartialGenerated.AddDynamic(this, &URagStoreComponent::HandleStoreAskMarkdownPartial);
    Store->OnAskResponseGenerated.AddDynamic(this,   &URagStoreComponent::HandleStoreAskResponse);
    Store->OnAskEndOfStream.AddDynamic(this,         &URagStoreComponent::HandleStoreAskEndOfStream);
    Store->OnAskError.AddDynamic(this,               &URagStoreComponent::HandleStoreAskError);
}

void URagStoreComponent::SyncStoreConfig()
{
    if (!Store) { return; }
    Store->EmbeddingModelParams      = EmbeddingModelParams;
    Store->ExternalEmbedder          = ExternalEmbedder;
    Store->AnswerModelParams         = AnswerModelParams;
    Store->AnswerEngine              = AnswerEngine;
    Store->SummarizingPromptTemplate = SummarizingPromptTemplate;
    Store->VectorParams              = VectorParams;
    Store->ChunkerParams             = ChunkerParams;
    Store->RetrievalDefaults         = RetrievalDefaults;
    Store->bSyncVectorDimToEmbedder  = bSyncVectorDimToEmbedder;
    Store->bBroadcastChunksOnAsk     = bBroadcastChunksOnAsk;
}

ULlamaComponent* URagStoreComponent::AutoDiscoverEmbedder()
{
    AActor* Owner = GetOwner();
    if (!Owner) { return nullptr; }

    TArray<ULlamaComponent*> Comps;
    Owner->GetComponents<ULlamaComponent>(Comps);
    for (ULlamaComponent* C : Comps)
    {
        if (C && C->ModelParams.Advanced.bEmbeddingMode) { return C; }
    }
    return Comps.Num() > 0 ? Comps[0] : nullptr;
}

void URagStoreComponent::LoadModels()
{
    EnsureStore();
    SyncStoreConfig();
    Store->LoadModels();
}

void URagStoreComponent::LoadAndInitialize()
{
    EnsureStore();
    SyncStoreConfig();
    Store->LoadAndInitialize();
}

bool URagStoreComponent::Initialize()
{
    EnsureStore();
    SyncStoreConfig();
    Store->Initialize();
    return Store->IsInitialized();
}

bool URagStoreComponent::IsInitialized() const     { return Store && Store->IsInitialized(); }
int32 URagStoreComponent::NumChunks() const        { return Store ? Store->NumChunks() : 0; }
bool URagStoreComponent::IsEmbedderReady() const   { return Store && Store->IsEmbedderReady(); }
bool URagStoreComponent::IsAnswerEngineReady() const { return Store && Store->IsAnswerEngineReady(); }

void URagStoreComponent::Reset() { if (Store) { Store->Reset(); } }

// ── Ingest ───────────────────────────────────────────────────────────────────

void URagStoreComponent::IngestText(const FString& Text, const FString& Source)
{
    EnsureStore(); SyncStoreConfig();
    Store->IngestText(Text, Source);
}

bool URagStoreComponent::IngestFile(const FString& FilePath)
{
    EnsureStore(); SyncStoreConfig();
    return Store->IngestFile(FilePath);
}

void URagStoreComponent::IngestDocuments(const TArray<FString>& Texts, const TArray<FString>& Sources)
{
    EnsureStore(); SyncStoreConfig();
    Store->IngestDocuments(Texts, Sources);
}

int32 URagStoreComponent::IngestDirectory(const FString& FolderPath, const FString& ExtensionsCsv, bool bRecursive)
{
    EnsureStore(); SyncStoreConfig();
    return Store->IngestDirectory(FolderPath, ExtensionsCsv, bRecursive);
}

// ── Retrieval ────────────────────────────────────────────────────────────────

void URagStoreComponent::RetrieveAsync(const FString& Query, FRagRetrievalParams Params)
{
    EnsureStore(); SyncStoreConfig();

    TWeakObjectPtr<URagStoreComponent> WeakThis(this);
    const FString QueryCopy = Query;
    Store->RetrieveAsync(Query, Params,
        [WeakThis, QueryCopy](const TArray<FLlamaChunk>& Chunks)
        {
            URagStoreComponent* Self = WeakThis.Get();
            if (!Self) { return; }
            Self->OnRetrievalComplete.Broadcast(Chunks, QueryCopy);
        });
}

void URagStoreComponent::RetrieveAsyncDefault(const FString& Query)
{
    RetrieveAsync(Query, RetrievalDefaults);
}

// ── Ask ──────────────────────────────────────────────────────────────────────

void URagStoreComponent::Ask(const FString& Query, FRagRetrievalParams Params)
{
    EnsureStore(); SyncStoreConfig();
    Store->Ask(Query, Params);
}

void URagStoreComponent::AskDefault(const FString& Query)
{
    EnsureStore(); SyncStoreConfig();
    Store->AskDefault(Query);
}

// ── Utilities / persistence ─────────────────────────────────────────────────

FString URagStoreComponent::FormatChunksAsContext(const TArray<FLlamaChunk>& InChunks, const FString& HeaderTemplate) const
{
    return Store ? Store->FormatChunksAsContext(InChunks, HeaderTemplate) : FString();
}

bool URagStoreComponent::SaveToFile(const FString& FilePath)
{
    return Store && Store->SaveToFile(FilePath);
}

bool URagStoreComponent::LoadFromFile(const FString& FilePath)
{
    EnsureStore();
    const bool bOk = Store->LoadFromFile(FilePath);
    if (bOk) { VectorParams = Store->VectorParams; }
    return bOk;
}

// ── Relay handlers ──────────────────────────────────────────────────────────

void URagStoreComponent::HandleStoreIngestComplete(int32 ChunksAdded)
{
    OnIngestComplete.Broadcast(ChunksAdded);
}

void URagStoreComponent::HandleStoreRagPipelineReady()
{
    // Re-sync VectorParams in case the inner store auto-pulled Dimensions from the
    // loaded embedder during the chain.
    if (Store) { VectorParams = Store->VectorParams; }
    OnRagPipelineReady.Broadcast();
}

void URagStoreComponent::HandleStoreAskRetrieved(const TArray<FLlamaChunk>& RetrievedChunks)
{
    OnAskRetrievedChunks.Broadcast(RetrievedChunks);
}
void URagStoreComponent::HandleStoreAskToken(const FString& Token)
{
    OnAskTokenGenerated.Broadcast(Token);
}
void URagStoreComponent::HandleStoreAskPartial(const FString& Partial)
{
    OnAskPartialGenerated.Broadcast(Partial);
}
void URagStoreComponent::HandleStoreAskMarkdownPartial(const FString& Partial, EMarkdownStreamState State)
{
    OnAskMarkdownPartialGenerated.Broadcast(Partial, State);
}
void URagStoreComponent::HandleStoreAskResponse(const FString& Response)
{
    OnAskResponseGenerated.Broadcast(Response);
}
void URagStoreComponent::HandleStoreAskEndOfStream(bool bStopSeq, float Tps)
{
    OnAskEndOfStream.Broadcast(bStopSeq, Tps);
}
void URagStoreComponent::HandleStoreAskError(const FString& Err, int32 Code)
{
    OnAskError.Broadcast(Err, Code);
}
