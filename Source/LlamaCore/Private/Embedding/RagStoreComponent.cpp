// Copyright 2025-current Getnamo.

#include "Embedding/RagStoreComponent.h"
#include "LlamaComponent.h"
#include "LlamaUtility.h"
#include "GameFramework/Actor.h"

URagStoreComponent::URagStoreComponent()
{
    PrimaryComponentTick.bCanEverTick = true;
    PrimaryComponentTick.bStartWithTickEnabled = true;
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
        Store->LoadModels();
        bPendingAutoInitialize = true; // resolved by TickComponent once embedder is ready
        TryAutoInitialize();
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
    if (bPendingAutoInitialize) { TryAutoInitialize(); }
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
}

void URagStoreComponent::TryAutoInitialize()
{
    if (!bPendingAutoInitialize || !Store) { return; }

    // Wait until either an embedder is ready OR the user has manually populated
    // VectorParams.Dimensions (in which case we can Initialize without an embedder
    // — useful for LoadFromFile flows).
    const bool bCanInit = Store->IsEmbedderReady() || Store->VectorParams.Dimensions > 0;
    if (!bCanInit) { return; }

    // Re-sync VectorParams from Store in case an internal embedder has populated
    // Dimensions during its OnModelLoaded.
    VectorParams = Store->VectorParams;
    Store->Initialize();
    bPendingAutoInitialize = false;
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
