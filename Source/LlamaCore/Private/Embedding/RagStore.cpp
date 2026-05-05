// Copyright 2025-current Getnamo.

#include "Embedding/RagStore.h"
#include "Embedding/BM25Index.h"
#include "Embedding/HybridRetriever.h"

#include "LlamaComponent.h"
#include "LlamaUtility.h"

#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "HAL/FileManager.h"
#include "Serialization/MemoryReader.h"
#include "Serialization/MemoryWriter.h"

namespace
{
    constexpr uint32 RAG_MAGIC = 0x52414730; // 'RAG0'
    constexpr uint32 RAG_VERSION = 1;

    /** Chunks are addressed by 1-based id (matching FVectorDatabase auto-id scheme). */
    static int64 ChunkIndexToId(int32 Index) { return static_cast<int64>(Index) + 1; }
    static int32 ChunkIdToIndex(int64 Id)    { return static_cast<int32>(Id) - 1; }
}

URagStore::URagStore()
{
    Vector = MakeUnique<FVectorDatabase>();
    Bm25 = MakeUnique<FBM25Index>();
}

URagStore::~URagStore() = default;

void URagStore::Initialize()
{
    Vector->Params = VectorParams;
    Vector->InitializeDB();
    Bm25->Reset();
    Chunks.Empty();
    bInitialized = true;
}

void URagStore::Reset()
{
    if (Vector) { Vector->Reset(); }
    if (Bm25)   { Bm25->Reset();   }
    Chunks.Empty();
    bInitialized = false;
}

void URagStore::IngestText(const FString& Text, const FString& Source)
{
    if (!bInitialized)
    {
        UE_LOG(LlamaLog, Warning, TEXT("URagStore::IngestText called before Initialize"));
        return;
    }
    if (!Embedder)
    {
        UE_LOG(LlamaLog, Warning, TEXT("URagStore::IngestText: Embedder is null"));
        return;
    }

    TArray<FLlamaChunk> NewChunks;
    FLlamaCorpusChunker::ChunkText(Text, Source, ChunkerParams, NewChunks);
    if (NewChunks.Num() == 0)
    {
        OnIngestComplete.Broadcast(0);
        return;
    }

    TArray<FString> Texts;
    Texts.Reserve(NewChunks.Num());
    for (const FLlamaChunk& C : NewChunks) { Texts.Add(C.Text); }

    TWeakObjectPtr<URagStore> WeakThis(this);
    Embedder->EmbedTextsAsync(Texts,
        [WeakThis, NewChunks](const TArray<TArray<float>>& All, const TArray<FString>& /*Sources*/) mutable
        {
            URagStore* Self = WeakThis.Get();
            if (!Self) { return; }
            Self->IngestChunksWithEmbeddings(NewChunks, All);
        });
}

bool URagStore::IngestFile(const FString& FilePath)
{
    FString Body;
    if (!FFileHelper::LoadFileToString(Body, *FilePath))
    {
        UE_LOG(LlamaLog, Warning, TEXT("URagStore::IngestFile could not read %s"), *FilePath);
        return false;
    }
    IngestText(Body, FPaths::GetCleanFilename(FilePath));
    return true;
}

void URagStore::IngestChunksWithEmbeddings(const TArray<FLlamaChunk>& NewChunks,
                                           const TArray<TArray<float>>& Embeddings)
{
    if (NewChunks.Num() != Embeddings.Num())
    {
        UE_LOG(LlamaLog, Warning, TEXT("URagStore: embedding/chunk count mismatch (%d vs %d)"),
            Embeddings.Num(), NewChunks.Num());
        return;
    }

    int32 Added = 0;
    for (int32 i = 0; i < NewChunks.Num(); ++i)
    {
        if (Embeddings[i].Num() != Vector->Params.Dimensions)
        {
            UE_LOG(LlamaLog, Warning,
                TEXT("URagStore: chunk %d embedding dim %d != VectorParams.Dimensions %d, skipping"),
                i, Embeddings[i].Num(), Vector->Params.Dimensions);
            continue;
        }

        const int32 Index = Chunks.Add(NewChunks[i]);
        const int64 Id = ChunkIndexToId(Index);

        Vector->AddVectorEmbeddingIdPair(Embeddings[i], Id);
        Bm25->AddDocument(Id, NewChunks[i].Text);
        ++Added;
    }

    if (Added > 0) { Bm25->Finalize(); }
    OnIngestComplete.Broadcast(Added);
}

void URagStore::Retrieve(const TArray<float>& QueryEmbedding, const FString& QueryText,
                         const FRagRetrievalParams& Params, TArray<FLlamaChunk>& OutChunks)
{
    OutChunks.Reset();
    if (!bInitialized || Chunks.Num() == 0) { return; }

    TArray<int64> Ids;
    TArray<float> Scores;

    switch (Params.Mode)
    {
    case ERagRetrievalMode::Vector:
        if (QueryEmbedding.Num() > 0)
        {
            Vector->FindNearestNIds(Ids, Scores, QueryEmbedding, Params.TopK);
        }
        break;

    case ERagRetrievalMode::BM25:
        Bm25->Query(QueryText, Params.TopK, Ids, Scores);
        break;

    case ERagRetrievalMode::Hybrid:
    default:
    {
        FHybridRetriever R;
        R.Vector = Vector.Get();
        R.Bm25   = Bm25.Get();
        R.Query(QueryEmbedding, QueryText,
                Params.TopK, Params.CandidatesPerSide, Params.RRFConstant,
                Ids, Scores);
        break;
    }
    }

    OutChunks.Reserve(Ids.Num());
    for (int64 Id : Ids)
    {
        const int32 Idx = ChunkIdToIndex(Id);
        if (Chunks.IsValidIndex(Idx))
        {
            OutChunks.Add(Chunks[Idx]);
        }
    }
}

void URagStore::RetrieveAsync(const FString& QueryText, const FRagRetrievalParams& Params,
                              TFunction<void(const TArray<FLlamaChunk>&)> OnDone)
{
    if (!bInitialized || !Embedder)
    {
        if (OnDone) { OnDone(TArray<FLlamaChunk>()); }
        return;
    }

    // BM25-only path: no embedding needed
    if (Params.Mode == ERagRetrievalMode::BM25)
    {
        TArray<FLlamaChunk> Out;
        Retrieve(TArray<float>(), QueryText, Params, Out);
        if (OnDone) { OnDone(Out); }
        return;
    }

    TArray<FString> Single = { QueryText };
    TWeakObjectPtr<URagStore> WeakThis(this);
    const FRagRetrievalParams ParamsCopy = Params;

    Embedder->EmbedTextsAsync(Single,
        [WeakThis, ParamsCopy, QueryText, OnDone = MoveTemp(OnDone)]
        (const TArray<TArray<float>>& All, const TArray<FString>& /*Sources*/) mutable
        {
            URagStore* Self = WeakThis.Get();
            if (!Self) { if (OnDone) OnDone(TArray<FLlamaChunk>()); return; }

            const TArray<float> Empty;
            const TArray<float>& Q = (All.Num() > 0) ? All[0] : Empty;
            TArray<FLlamaChunk> Out;
            Self->Retrieve(Q, QueryText, ParamsCopy, Out);
            if (OnDone) { OnDone(Out); }
        });
}

FString URagStore::FormatChunksAsContext(const TArray<FLlamaChunk>& InChunks, const FString& HeaderTemplate) const
{
    FString Out;
    Out.Reserve(2048);
    if (!HeaderTemplate.IsEmpty())
    {
        Out += HeaderTemplate;
        Out += TEXT("\n\n");
    }
    for (int32 i = 0; i < InChunks.Num(); ++i)
    {
        const FLlamaChunk& C = InChunks[i];
        Out += FString::Printf(TEXT("[%d] (%s) %s\n\n"),
            i + 1,
            C.Source.IsEmpty() ? TEXT("anon") : *C.Source,
            *C.Text);
    }
    return Out;
}

bool URagStore::SaveToFile(const FString& FilePath)
{
    if (!bInitialized)
    {
        UE_LOG(LlamaLog, Warning, TEXT("URagStore::SaveToFile before Initialize"));
        return false;
    }

    const FString DirOnly = FPaths::GetPath(FilePath);
    if (!DirOnly.IsEmpty())
    {
        IFileManager::Get().MakeDirectory(*DirOnly, /*Tree*/ true);
    }

    // Save vector DB to a sibling path, then bundle.
    const FString VdbPath = FilePath + TEXT(".vdb.tmp");
    if (!Vector->Save(VdbPath))
    {
        UE_LOG(LlamaLog, Warning, TEXT("URagStore::SaveToFile vector save failed"));
        return false;
    }
    TArray<uint8> VdbBytes;
    if (!FFileHelper::LoadFileToArray(VdbBytes, *VdbPath))
    {
        IFileManager::Get().Delete(*VdbPath, false, true, true);
        return false;
    }
    IFileManager::Get().Delete(*VdbPath, false, true, true);

    TArray<uint8> Buffer;
    FMemoryWriter Writer(Buffer, /*persistent*/ true);

    uint32 Magic = RAG_MAGIC;
    uint32 Version = RAG_VERSION;
    Writer << Magic;
    Writer << Version;

    // Chunk metadata
    int32 NChunks = Chunks.Num();
    Writer << NChunks;
    for (FLlamaChunk& C : Chunks)
    {
        Writer << C.Text;
        Writer << C.StartChar;
        Writer << C.EndChar;
        Writer << C.Source;
    }

    // BM25 index
    Bm25->Save(Writer);

    // Embedded VDB blob
    int64 VdbSize = VdbBytes.Num();
    Writer << VdbSize;
    Writer.Serialize(VdbBytes.GetData(), VdbBytes.Num());

    return FFileHelper::SaveArrayToFile(Buffer, *FilePath);
}

bool URagStore::LoadFromFile(const FString& FilePath)
{
    TArray<uint8> Buffer;
    if (!FFileHelper::LoadFileToArray(Buffer, *FilePath)) { return false; }

    FMemoryReader Reader(Buffer, /*persistent*/ true);
    uint32 Magic = 0, Version = 0;
    Reader << Magic;
    Reader << Version;
    if (Magic != RAG_MAGIC || Version != RAG_VERSION)
    {
        UE_LOG(LlamaLog, Warning, TEXT("URagStore::LoadFromFile bad magic/version"));
        return false;
    }

    Reset();

    int32 NChunks = 0;
    Reader << NChunks;
    Chunks.Reserve(NChunks);
    for (int32 i = 0; i < NChunks; ++i)
    {
        FLlamaChunk C;
        Reader << C.Text;
        Reader << C.StartChar;
        Reader << C.EndChar;
        Reader << C.Source;
        Chunks.Add(MoveTemp(C));
    }

    if (!Bm25->Load(Reader)) { return false; }

    int64 VdbSize = 0;
    Reader << VdbSize;
    if (VdbSize <= 0 || VdbSize > static_cast<int64>(Buffer.Num())) { return false; }

    TArray<uint8> VdbBytes;
    VdbBytes.SetNumUninitialized(static_cast<int32>(VdbSize));
    Reader.Serialize(VdbBytes.GetData(), VdbBytes.Num());

    const FString VdbPath = FilePath + TEXT(".vdb.tmp");
    if (!FFileHelper::SaveArrayToFile(VdbBytes, *VdbPath)) { return false; }
    const bool bOk = Vector->Load(VdbPath);
    IFileManager::Get().Delete(*VdbPath, false, true, true);

    if (!bOk) { return false; }
    VectorParams = Vector->Params;
    bInitialized = true;
    return true;
}
