// Copyright 2025-current Getnamo.

#include "Embedding/RagStoreComponent.h"
#include "LlamaComponent.h"
#include "LlamaUtility.h"
#include "GameFramework/Actor.h"

URagStoreComponent::URagStoreComponent()
{
    PrimaryComponentTick.bCanEverTick = false;
}

void URagStoreComponent::BeginPlay()
{
    Super::BeginPlay();
    EnsureStore();

    if (!Embedder)
    {
        Embedder = ResolveEmbedder();
    }

    if (bAutoInitializeOnBeginPlay)
    {
        // If embedder dim is known now, init immediately; otherwise the user can call Initialize()
        // manually after their model loads (typical: bind to OnModelLoaded -> Initialize).
        if (Embedder)
        {
            const int32 Dim = Embedder->GetEmbeddingDimension();
            if (Dim > 0)
            {
                if (VectorParams.Dimensions <= 0) { VectorParams.Dimensions = Dim; }
                Initialize();
            }
        }
        else if (VectorParams.Dimensions > 0)
        {
            // Allow init without an embedder (e.g. for LoadFromFile-only consumers).
            Initialize();
        }
    }
}

void URagStoreComponent::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
    if (Store)
    {
        Store->OnIngestComplete.RemoveAll(this);
    }
    Super::EndPlay(EndPlayReason);
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
    // Re-broadcast ingest completion so consumers can subscribe at the component level.
    Store->OnIngestComplete.AddDynamic(this, &URagStoreComponent::HandleStoreIngestComplete);
}

void URagStoreComponent::HandleStoreIngestComplete(int32 ChunksAdded)
{
    OnIngestComplete.Broadcast(ChunksAdded);
}

ULlamaComponent* URagStoreComponent::ResolveEmbedder()
{
    AActor* Owner = GetOwner();
    if (!Owner) { return nullptr; }

    TArray<ULlamaComponent*> Comps;
    Owner->GetComponents<ULlamaComponent>(Comps);
    for (ULlamaComponent* C : Comps)
    {
        if (C && C->ModelParams.Advanced.bEmbeddingMode)
        {
            return C;
        }
    }
    // Fall back to the first ULlamaComponent — caller may set bEmbeddingMode after construction.
    return Comps.Num() > 0 ? Comps[0] : nullptr;
}

bool URagStoreComponent::Initialize()
{
    EnsureStore();

    // Pull dim from the embedder if not explicitly configured.
    if (VectorParams.Dimensions <= 0 && Embedder)
    {
        const int32 Dim = Embedder->GetEmbeddingDimension();
        if (Dim > 0) { VectorParams.Dimensions = Dim; }
    }

    if (VectorParams.Dimensions <= 0)
    {
        UE_LOG(LlamaLog, Warning, TEXT("URagStoreComponent::Initialize: VectorParams.Dimensions not set and embedder dim unknown"));
        return false;
    }

    Store->Embedder       = Embedder;
    Store->VectorParams   = VectorParams;
    Store->ChunkerParams  = ChunkerParams;
    Store->RetrievalDefaults = RetrievalDefaults;
    Store->Initialize();
    return Store->IsInitialized();
}

bool URagStoreComponent::IsInitialized() const
{
    return Store && Store->IsInitialized();
}

int32 URagStoreComponent::NumChunks() const
{
    return Store ? Store->NumChunks() : 0;
}

void URagStoreComponent::Reset()
{
    if (Store) { Store->Reset(); }
}

void URagStoreComponent::IngestText(const FString& Text, const FString& Source)
{
    EnsureStore();
    Store->Embedder = Embedder; // keep in sync if user reassigned
    Store->IngestText(Text, Source);
}

bool URagStoreComponent::IngestFile(const FString& FilePath)
{
    EnsureStore();
    Store->Embedder = Embedder;
    return Store->IngestFile(FilePath);
}

void URagStoreComponent::IngestDocuments(const TArray<FString>& Texts, const TArray<FString>& Sources)
{
    EnsureStore();
    Store->Embedder = Embedder;
    Store->IngestDocuments(Texts, Sources);
}

int32 URagStoreComponent::IngestDirectory(const FString& FolderPath, const FString& ExtensionsCsv, bool bRecursive)
{
    EnsureStore();
    Store->Embedder = Embedder;
    return Store->IngestDirectory(FolderPath, ExtensionsCsv, bRecursive);
}

void URagStoreComponent::RetrieveAsync(const FString& Query, FRagRetrievalParams Params)
{
    EnsureStore();
    Store->Embedder = Embedder;

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
