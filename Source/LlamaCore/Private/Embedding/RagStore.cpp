// Copyright 2025-current Getnamo.

#include "Embedding/RagStore.h"
#include "Embedding/BM25Index.h"
#include "Embedding/HybridRetriever.h"

#include "LlamaComponent.h"
#include "LlamaDualBackend.h"
#include "LlamaNative.h"
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

    // Sensible defaults preconfigured for the canonical recommended models.
    // Users override any field via the property panel; PathToModel uses the './...' form
    // which FLlamaPaths resolves under <Project>/Saved/Models/.

    // Embedder: nomic-embed-text-v1.5.Q4_K_M (~85 MB, 768-dim, English-strong).
    EmbeddingModelParams.PathToModel              = TEXT("./nomic-embed-text-v1.5.Q4_K_M.gguf");
    EmbeddingModelParams.MaxContextLength         = 2048;
    EmbeddingModelParams.GPULayers                = 99;
    EmbeddingModelParams.MaxBatchLength           = 2048;
    EmbeddingModelParams.bAutoLoadModelOnStartup  = false;
    EmbeddingModelParams.bAutoInsertSystemPromptOnLoad = false;
    EmbeddingModelParams.SystemPrompt             = TEXT("");
    EmbeddingModelParams.Advanced.bEmbeddingMode  = true;

    // Answerer: gemma-3-4b-it Q4_K_L (~2.5 GB, multilingual chat).
    AnswerModelParams.PathToModel                 = TEXT("./google_gemma-3-4b-it-Q4_K_L.gguf");
    AnswerModelParams.MaxContextLength            = 8192;
    AnswerModelParams.GPULayers                   = 99;
    AnswerModelParams.MaxBatchLength              = 1024;
    AnswerModelParams.bAutoLoadModelOnStartup     = false;
    AnswerModelParams.bAutoInsertSystemPromptOnLoad = true;
    AnswerModelParams.SystemPrompt                = TEXT("You are a helpful AI assistant.");
}

URagStore::~URagStore() = default;

void URagStore::BeginDestroy()
{
    if (InternalEmbedder)
    {
        if (InternalEmbedder->GetLlamaNative())
        {
            InternalEmbedder->GetLlamaNative()->RemoveTicker();
        }
        InternalEmbedder->Shutdown();
        InternalEmbedder.Reset();
    }
    if (InternalAnswerer)
    {
        if (InternalAnswerer->GetLlamaNative())
        {
            InternalAnswerer->GetLlamaNative()->RemoveTicker();
        }
        InternalAnswerer->Shutdown();
        InternalAnswerer.Reset();
    }
    Super::BeginDestroy();
}

void URagStore::LoadModels()
{
    // Sequential: kick off embedder first; its OnModelLoaded chains the answerer.
    // This avoids concurrent ggml_backend_load_all() / Vulkan device init races that
    // could crash when both backends initialize at once.
    if (!EmbeddingModelParams.PathToModel.IsEmpty() && !InternalEmbedder)
    {
        LoadEmbedderInternal();
    }
    else if (!AnswerModelParams.PathToModel.IsEmpty() && !InternalAnswerer)
    {
        // No embedder configured — load the answerer directly.
        LoadAnswererInternal();
    }
    // If neither, nothing to do; user is presumably wiring an external embedder.
}

void URagStore::LoadEmbedderInternal()
{
    // Force the embedding-mode flag — that's the whole point of this backend.
    if (!EmbeddingModelParams.Advanced.bEmbeddingMode)
    {
        UE_LOG(LlamaLog, Log, TEXT("URagStore: forcing EmbeddingModelParams.Advanced.bEmbeddingMode = true"));
        EmbeddingModelParams.Advanced.bEmbeddingMode = true;
    }

    InternalEmbedder = MakeUnique<FLlamaDualBackend>();
    InternalEmbedder->Initialize();
    if (FLlamaNative* Native = InternalEmbedder->GetLlamaNative())
    {
        Native->AddTicker();
    }
    InternalEmbedder->ModelParams = EmbeddingModelParams;
    InternalEmbedder->bUseRemote = false;

    InternalEmbedder->OnModelLoaded = [this](const FString& /*ModelName*/)
    {
        bInternalEmbedderReady = true;
        if (VectorParams.Dimensions <= 0 && InternalEmbedder)
        {
            const int32 Dim = InternalEmbedder->GetEmbeddingDimension();
            if (Dim > 0) { VectorParams.Dimensions = Dim; }
        }

        // Chain: now load the answerer if configured (serialized to avoid backend races).
        if (!AnswerModelParams.PathToModel.IsEmpty() && !InternalAnswerer)
        {
            LoadAnswererInternal();
        }
        else
        {
            // No answerer to load — the chain is complete after this.
            OnAllInternalLoadsComplete();
        }
    };
    InternalEmbedder->OnError = [this](const FString& Err, int32 /*Code*/)
    {
        UE_LOG(LlamaLog, Warning, TEXT("URagStore embedder error: %s"), *Err);
    };

    InternalEmbedder->LoadModel(/*bForceReload=*/false);
}

void URagStore::LoadAnswererInternal()
{
    InternalAnswerer = MakeUnique<FLlamaDualBackend>();
    InternalAnswerer->Initialize();
    if (FLlamaNative* Native = InternalAnswerer->GetLlamaNative())
    {
        Native->AddTicker();
    }
    InternalAnswerer->ModelParams = AnswerModelParams;
    InternalAnswerer->bUseRemote = false;

    InternalAnswerer->OnModelLoaded = [this](const FString& /*ModelName*/)
    {
        bInternalAnswererReady = true;
        OnAllInternalLoadsComplete();
    };

    // Wire streaming callbacks gated on bAskInFlight so unrelated direct chat
    // through the same backend doesn't bleed into OnAsk* delegates.
    TWeakObjectPtr<URagStore> WeakThis(this);
    InternalAnswerer->OnTokenGenerated = [WeakThis](const FString& Token)
    {
        URagStore* Self = WeakThis.Get();
        if (Self && Self->bAskInFlight) { Self->OnAskTokenGenerated.Broadcast(Token); }
    };
    InternalAnswerer->OnPartialGenerated = [WeakThis](const FString& Partial)
    {
        URagStore* Self = WeakThis.Get();
        if (Self && Self->bAskInFlight) { Self->OnAskPartialGenerated.Broadcast(Partial); }
    };
    InternalAnswerer->OnMarkdownPartialGenerated = [WeakThis](const FString& Partial, EMarkdownStreamState State)
    {
        URagStore* Self = WeakThis.Get();
        if (Self && Self->bAskInFlight) { Self->OnAskMarkdownPartialGenerated.Broadcast(Partial, State); }
    };
    InternalAnswerer->OnResponseGenerated = [WeakThis](const FString& Response)
    {
        URagStore* Self = WeakThis.Get();
        if (Self && Self->bAskInFlight) { Self->OnAskResponseGenerated.Broadcast(Response); }
    };
    InternalAnswerer->OnEndOfStream = [WeakThis](bool bStopSeq, float Tps)
    {
        URagStore* Self = WeakThis.Get();
        if (!Self) { return; }
        if (Self->bAskInFlight)
        {
            Self->OnAskEndOfStream.Broadcast(bStopSeq, Tps);
            Self->bAskInFlight = false;
        }
    };
    InternalAnswerer->OnError = [WeakThis](const FString& Err, int32 Code)
    {
        URagStore* Self = WeakThis.Get();
        if (!Self) { return; }
        UE_LOG(LlamaLog, Warning, TEXT("URagStore answerer error: %s"), *Err);
        if (Self->bAskInFlight)
        {
            Self->OnAskError.Broadcast(Err, Code);
            Self->bAskInFlight = false;
        }
    };

    InternalAnswerer->LoadModel(/*bForceReload=*/false);
}

void URagStore::OnAllInternalLoadsComplete()
{
    // If LoadAndInitialize() drove this chain, run Initialize() now and signal ready.
    if (bAutoInitPending)
    {
        bAutoInitPending = false;
        Initialize();
        OnRagPipelineReady.Broadcast();
    }
}

void URagStore::LoadAndInitialize()
{
    // Already fully ready? Just signal.
    if (IsInitialized())
    {
        OnRagPipelineReady.Broadcast();
        return;
    }

    // No internal models AND no external paths — can still init if the user has set
    // VectorParams.Dimensions or is loading from a saved file.
    const bool bHasInternalEmbedder = !EmbeddingModelParams.PathToModel.IsEmpty();
    const bool bHasInternalAnswerer = !AnswerModelParams.PathToModel.IsEmpty();
    if (!bHasInternalEmbedder && !bHasInternalAnswerer)
    {
        // Nothing to load; init now if we can, then signal.
        Initialize();
        OnRagPipelineReady.Broadcast();
        return;
    }

    bAutoInitPending = true;
    LoadModels();
}

bool URagStore::IsEmbedderReady() const
{
    if (ExternalEmbedder && ExternalEmbedder->ModelParams.Advanced.bEmbeddingMode &&
        ExternalEmbedder->IsModelLoaded())
    {
        return true;
    }
    return bInternalEmbedderReady;
}

bool URagStore::IsAnswerEngineReady() const
{
    if (bInternalAnswererReady) { return true; }
    if (AnswerEngine && AnswerEngine->IsModelLoaded()) { return true; }
    return false;
}

void URagStore::Initialize()
{
    // Resolve the embedder's actual output dimension (if any backend is ready) and prefer
    // it over a user-supplied VectorParams.Dimensions. This avoids subtle mismatch bugs
    // when a user changes embedding model without updating Dimensions (the default 384
    // matches bge-small but not nomic-768 / e5-1024 etc).
    int32 EmbedderDim = 0;
    if (bInternalEmbedderReady && InternalEmbedder)
    {
        EmbedderDim = InternalEmbedder->GetEmbeddingDimension();
    }
    else if (ExternalEmbedder && ExternalEmbedder->IsModelLoaded())
    {
        EmbedderDim = ExternalEmbedder->GetEmbeddingDimension();
    }

    if (EmbedderDim > 0)
    {
        if (bSyncVectorDimToEmbedder)
        {
            // Silent overwrite — this is the default ergonomic path.
            VectorParams.Dimensions = EmbedderDim;
        }
        else if (VectorParams.Dimensions > 0 && VectorParams.Dimensions != EmbedderDim)
        {
            // Manual mode but mismatch — warn, don't override.
            UE_LOG(LlamaLog, Warning,
                TEXT("URagStore::Initialize: VectorParams.Dimensions=%d disagrees with the loaded embedder's output dim=%d. ")
                TEXT("Sync skipped because bSyncVectorDimToEmbedder=false; expect ingest mismatch errors unless you intend this."),
                VectorParams.Dimensions, EmbedderDim);
        }
        else if (VectorParams.Dimensions <= 0)
        {
            // Manual mode but unset — fall through to the embedder dim anyway since 0 is invalid.
            VectorParams.Dimensions = EmbedderDim;
        }
    }
    else if (VectorParams.Dimensions <= 0)
    {
        UE_LOG(LlamaLog, Warning,
            TEXT("URagStore::Initialize: VectorParams.Dimensions is 0 and no embedder is ready to auto-pull from. ")
            TEXT("Set Dimensions manually or call LoadModels() first and wait for the embedder to load."));
    }

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

void URagStore::EmbedTextsViaActiveEmbedder(const TArray<FString>& Texts,
    TFunction<void(const TArray<TArray<float>>&, const TArray<FString>&)> OnDone)
{
    // External embedder wins if configured AND in embedding mode AND loaded.
    if (ExternalEmbedder &&
        ExternalEmbedder->ModelParams.Advanced.bEmbeddingMode &&
        ExternalEmbedder->IsModelLoaded())
    {
        ExternalEmbedder->EmbedTextsAsync(Texts, MoveTemp(OnDone));
        return;
    }
    if (bInternalEmbedderReady && InternalEmbedder)
    {
        InternalEmbedder->EmbedTextsAsync(Texts, MoveTemp(OnDone));
        return;
    }
    UE_LOG(LlamaLog, Warning, TEXT("URagStore: no embedder ready (configure EmbeddingModelParams.PathToModel + LoadModels(), or set ExternalEmbedder)"));
    if (OnDone) { OnDone(TArray<TArray<float>>(), TArray<FString>()); }
}

void URagStore::IngestText(const FString& Text, const FString& Source)
{
    if (!bInitialized)
    {
        UE_LOG(LlamaLog, Warning, TEXT("URagStore::IngestText called before Initialize"));
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
    EmbedTextsViaActiveEmbedder(Texts,
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

void URagStore::IngestDocuments(const TArray<FString>& Texts, const TArray<FString>& Sources)
{
    if (!bInitialized)
    {
        UE_LOG(LlamaLog, Warning, TEXT("URagStore::IngestDocuments called before Initialize"));
        return;
    }
    if (Texts.Num() != Sources.Num())
    {
        UE_LOG(LlamaLog, Warning, TEXT("URagStore::IngestDocuments: Texts/Sources length mismatch (%d vs %d)"),
            Texts.Num(), Sources.Num());
        return;
    }

    TArray<FLlamaChunk> AllChunks;
    AllChunks.Reserve(Texts.Num() * 4);
    for (int32 i = 0; i < Texts.Num(); ++i)
    {
        TArray<FLlamaChunk> Local;
        FLlamaCorpusChunker::ChunkText(Texts[i], Sources[i], ChunkerParams, Local);
        AllChunks.Append(MoveTemp(Local));
    }

    if (AllChunks.Num() == 0)
    {
        OnIngestComplete.Broadcast(0);
        return;
    }

    TArray<FString> ChunkTexts;
    ChunkTexts.Reserve(AllChunks.Num());
    for (const FLlamaChunk& C : AllChunks) { ChunkTexts.Add(C.Text); }

    TWeakObjectPtr<URagStore> WeakThis(this);
    EmbedTextsViaActiveEmbedder(ChunkTexts,
        [WeakThis, AllChunks](const TArray<TArray<float>>& All, const TArray<FString>& /*Sources*/) mutable
        {
            URagStore* Self = WeakThis.Get();
            if (!Self) { return; }
            Self->IngestChunksWithEmbeddings(AllChunks, All);
        });
}

int32 URagStore::IngestDirectory(const FString& FolderPath, const FString& ExtensionsCsv, bool bRecursive)
{
    if (!bInitialized)
    {
        UE_LOG(LlamaLog, Warning, TEXT("URagStore::IngestDirectory called before Initialize"));
        return 0;
    }

    // ConvertRelativePathToFull resolves both absolute paths (no-op) and relative paths
    // (relative to CWD, typically the engine binary dir). Pass already-anchored paths in.
    const FString FullPath = FPaths::ConvertRelativePathToFull(FolderPath);

    if (!IFileManager::Get().DirectoryExists(*FullPath))
    {
        UE_LOG(LlamaLog, Warning, TEXT("URagStore::IngestDirectory: not a directory: %s"), *FullPath);
        return 0;
    }

    // Parse CSV of extensions (no dots, lowercase).
    TArray<FString> Extensions;
    {
        TArray<FString> Raw;
        ExtensionsCsv.ParseIntoArray(Raw, TEXT(","), /*CullEmpty*/ true);
        for (FString E : Raw)
        {
            E.TrimStartAndEndInline();
            if (E.StartsWith(TEXT("."))) { E.RightChopInline(1); }
            Extensions.Add(E.ToLower());
        }
    }

    TArray<FString> FilesFound;
    if (Extensions.Num() == 0)
    {
        // No filter — match everything.
        if (bRecursive)
        {
            IFileManager::Get().FindFilesRecursive(FilesFound, *FullPath, TEXT("*.*"), /*Files*/ true, /*Dirs*/ false);
        }
        else
        {
            TArray<FString> Names;
            IFileManager::Get().FindFiles(Names, *(FullPath / TEXT("*.*")), /*Files*/ true, /*Dirs*/ false);
            for (const FString& N : Names) { FilesFound.Add(FullPath / N); }
        }
    }
    else
    {
        // One pass per extension; FindFilesRecursive only takes one wildcard.
        for (const FString& Ext : Extensions)
        {
            const FString Pattern = FString::Printf(TEXT("*.%s"), *Ext);
            if (bRecursive)
            {
                TArray<FString> Match;
                IFileManager::Get().FindFilesRecursive(Match, *FullPath, *Pattern, /*Files*/ true, /*Dirs*/ false);
                FilesFound.Append(MoveTemp(Match));
            }
            else
            {
                TArray<FString> Names;
                IFileManager::Get().FindFiles(Names, *(FullPath / Pattern), /*Files*/ true, /*Dirs*/ false);
                for (const FString& N : Names) { FilesFound.Add(FullPath / N); }
            }
        }
    }

    if (FilesFound.Num() == 0)
    {
        UE_LOG(LlamaLog, Log, TEXT("URagStore::IngestDirectory: no matching files in %s"), *FullPath);
        OnIngestComplete.Broadcast(0);
        return 0;
    }

    TArray<FString> Texts;
    TArray<FString> Sources;
    Texts.Reserve(FilesFound.Num());
    Sources.Reserve(FilesFound.Num());

    for (const FString& File : FilesFound)
    {
        FString Body;
        if (FFileHelper::LoadFileToString(Body, *File))
        {
            // Use the path relative to the scanned folder as the source label so duplicate
            // filenames in different subfolders stay distinguishable.
            FString RelLabel = File;
            FPaths::MakePathRelativeTo(RelLabel, *(FullPath + TEXT("/")));
            Texts.Add(MoveTemp(Body));
            Sources.Add(MoveTemp(RelLabel));
        }
        else
        {
            UE_LOG(LlamaLog, Warning, TEXT("URagStore::IngestDirectory: failed to read %s"), *File);
        }
    }

    UE_LOG(LlamaLog, Log, TEXT("URagStore::IngestDirectory: queueing %d files from %s"),
        Texts.Num(), *FullPath);

    IngestDocuments(Texts, Sources);
    return Texts.Num();
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

    if (Ids.Num() == 0) { return; }

    // Score → similarity (higher = better) per retriever, then normalize against top-1
    // similarity to produce Confidence ∈ [0, 1] with top-1 always = 1.0.
    auto RawToSimilarity = [&](float RawScore) -> float
    {
        switch (Params.Mode)
        {
        case ERagRetrievalMode::Vector:
            // L2 distance for unit-norm embeddings: ||a - b||^2 = 2 - 2*cos(a,b),
            // so distance ∈ [0, 2]. Map to similarity ∈ [0, 1]: 1 - d/2.
            return FMath::Max(0.f, 1.f - RawScore * 0.5f);
        case ERagRetrievalMode::BM25:
        case ERagRetrievalMode::Hybrid:
        default:
            return FMath::Max(0.f, RawScore);
        }
    };

    const ERagRetrievalSource SourceTag =
        (Params.Mode == ERagRetrievalMode::Vector) ? ERagRetrievalSource::Vector :
        (Params.Mode == ERagRetrievalMode::BM25)   ? ERagRetrievalSource::BM25 :
                                                     ERagRetrievalSource::Hybrid;

    // Find the top-1 similarity for normalization. Results are already sorted nearest/best-first
    // by the retrievers (FindNearestNIds reverses the priority queue, BM25/Hybrid sort descending).
    float TopSim = -1.f;
    for (int32 i = 0; i < Ids.Num(); ++i)
    {
        const float Sim = RawToSimilarity(Scores[i]);
        if (Sim > TopSim) { TopSim = Sim; }
    }
    if (TopSim <= 0.f) { TopSim = 1.f; } // pathological: all-zero scores → keep top-1, no scaling

    OutChunks.Reserve(Ids.Num());
    for (int32 i = 0; i < Ids.Num(); ++i)
    {
        const int32 Idx = ChunkIdToIndex(Ids[i]);
        if (!Chunks.IsValidIndex(Idx)) { continue; }

        const float Sim = RawToSimilarity(Scores[i]);
        const float Confidence = Sim / TopSim;

        // Pre-filter, but always keep top-1 (i==0) so callers always get *something* when
        // results exist. MinConfidence is meant to trim the noisy tail, not blank the result.
        if (i > 0 && Confidence < Params.MinConfidence) { continue; }

        FLlamaChunk Chunk = Chunks[Idx];
        Chunk.RetrievalScore  = Scores[i];
        Chunk.Confidence      = Confidence;
        Chunk.SourceRetriever = SourceTag;
        OutChunks.Add(MoveTemp(Chunk));
    }
}

void URagStore::RetrieveAsync(const FString& QueryText, const FRagRetrievalParams& Params,
                              TFunction<void(const TArray<FLlamaChunk>&)> OnDone)
{
    if (!bInitialized)
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

    EmbedTextsViaActiveEmbedder(Single,
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

// ─── Ask pipeline ────────────────────────────────────────────────────────────

void URagStore::AskDefault(const FString& Query)
{
    Ask(Query, RetrievalDefaults);
}

void URagStore::Ask(const FString& Query, FRagRetrievalParams ParamsOverride)
{
    if (!bInitialized)
    {
        const FString Err = TEXT("URagStore::Ask called before Initialize");
        UE_LOG(LlamaLog, Warning, TEXT("%s"), *Err);
        OnAskError.Broadcast(Err, 70);
        return;
    }
    if (!IsAnswerEngineReady())
    {
        const FString Err = TEXT("URagStore::Ask: no answer engine ready (configure AnswerModelParams.PathToModel + LoadModels(), or set AnswerEngine)");
        UE_LOG(LlamaLog, Warning, TEXT("%s"), *Err);
        OnAskError.Broadcast(Err, 71);
        return;
    }
    if (bAskInFlight)
    {
        const FString Err = TEXT("URagStore::Ask: an Ask is already in flight; ignore or wait for OnAskEndOfStream");
        UE_LOG(LlamaLog, Warning, TEXT("%s"), *Err);
        OnAskError.Broadcast(Err, 72);
        return;
    }

    // Inherit defaults if user passed a default-constructed Params (i.e. TopK still 5,
    // or specifically signaled via TopK <= 0).
    if (ParamsOverride.TopK <= 0) { ParamsOverride = RetrievalDefaults; }

    TWeakObjectPtr<URagStore> WeakThis(this);
    const FString QueryCopy = Query;

    RetrieveAsync(Query, ParamsOverride,
        [WeakThis, QueryCopy](const TArray<FLlamaChunk>& Chunks)
        {
            URagStore* Self = WeakThis.Get();
            if (!Self) { return; }

            // Optionally surface the chunks for citation UI / debugging. Off by default
            // because most users only want the streamed answer; opt in via
            // bBroadcastChunksOnAsk. Direct RetrieveAsync callers always get chunks.
            if (Self->bBroadcastChunksOnAsk)
            {
                Self->OnAskRetrievedChunks.Broadcast(Chunks);
            }

            const FString FormattedPrompt = Self->BuildSummarizingPrompt(QueryCopy, Chunks);
            Self->bAskInFlight = true;
            Self->SendFormattedPromptToActiveAnswerer(FormattedPrompt);
        });
}

FString URagStore::BuildSummarizingPrompt(const FString& Query, const TArray<FLlamaChunk>& InChunks) const
{
    // Substitute a sentinel for empty chunk sets — feeding a totally empty Context block
    // tends to make the model emit empty completions. The placeholder gives it something
    // concrete to read so it follows the template instructions and responds with an
    // explicit "no context available" answer.
    FString Context;
    if (InChunks.Num() == 0)
    {
        Context = TEXT("(no relevant context was retrieved for this query)");
    }
    else
    {
        Context = FormatChunksAsContext(InChunks, /*HeaderTemplate*/ FString());
    }

    FString Out = SummarizingPromptTemplate;
    Out.ReplaceInline(TEXT("{context}"), *Context, ESearchCase::CaseSensitive);
    Out.ReplaceInline(TEXT("{query}"),   *Query,   ESearchCase::CaseSensitive);
    return Out;
}

void URagStore::SendFormattedPromptToActiveAnswerer(const FString& FormattedPrompt)
{
    // Internal answer backend wins.
    if (bInternalAnswererReady && InternalAnswerer)
    {
        // RAG queries are stateless by design: each Ask is a fresh single-turn
        // exchange. Without resetting, the answerer's chat history accumulates
        // across calls and the model can latch onto prior empty/wrong completions
        // and continue producing them. Keep the system prompt; drop everything else.
        InternalAnswerer->ResetContextHistory(/*bKeepSystemPrompt=*/ true);

        FLlamaChatPrompt P;
        P.Prompt = FormattedPrompt;
        P.Role = EChatTemplateRole::User;
        P.bAddAssistantBOS = false;
        P.bGenerateReply = true;
        InternalAnswerer->InsertTemplatedPrompt(P);
        return;
    }

    // External answer engine — bind dynamic delegates lazily on first use.
    if (AnswerEngine)
    {
        // Same statelessness guarantee for the external path. We reset the
        // component's history (preserving its system prompt) so back-to-back
        // Asks don't bleed between each other or into the user's own usage of
        // the same component.
        AnswerEngine->ResetContextHistory(/*bKeepSystemPrompt=*/ true);

        // We can't dynamically bind C++ lambdas to UFUNCTION dynamic multicasts. Instead
        // we relay through the COMPONENT's existing broadcast events. The component owns
        // its own multicast — we add UFUNCTION-marked relays on URagStore that subscribe
        // to those multicasts. Once-only binding so re-Asks don't double-fire.
        //
        // For an MVP we use a simpler shortcut: subscribe the component's broadcasts to
        // forwarding methods on URagStore via AddDynamic. Implementation details handled
        // in private helpers below. This MVP path supports one external answer engine
        // per RagStore lifetime — rebinding to a different engine mid-run is unsupported.
        static const FName N_Token = FName(TEXT("RelayAnswerToken"));
        static const FName N_Partial = FName(TEXT("RelayAnswerPartial"));
        static const FName N_Markdown = FName(TEXT("RelayAnswerMarkdownPartial"));
        static const FName N_Response = FName(TEXT("RelayAnswerResponse"));
        static const FName N_EndOfStream = FName(TEXT("RelayAnswerEndOfStream"));
        static const FName N_Error = FName(TEXT("RelayAnswerError"));

        // Rebind every call (cheap; AddUniqueDynamic is a no-op if already bound).
        AnswerEngine->OnTokenGenerated.AddUniqueDynamic(this,            &URagStore::RelayAnswerToken);
        AnswerEngine->OnPartialGenerated.AddUniqueDynamic(this,          &URagStore::RelayAnswerPartial);
        AnswerEngine->OnMarkdownPartialGenerated.AddUniqueDynamic(this,  &URagStore::RelayAnswerMarkdownPartial);
        AnswerEngine->OnResponseGenerated.AddUniqueDynamic(this,         &URagStore::RelayAnswerResponse);
        AnswerEngine->OnEndOfStream.AddUniqueDynamic(this,               &URagStore::RelayAnswerEndOfStream);
        AnswerEngine->OnError.AddUniqueDynamic(this,                     &URagStore::RelayAnswerError);

        AnswerEngine->InsertTemplatedPrompt(FormattedPrompt, EChatTemplateRole::User, /*bAddAssistantBOS*/ false, /*bGenerateReply*/ true);
        return;
    }

    // Should be unreachable — IsAnswerEngineReady() pre-check should have caught this.
    OnAskError.Broadcast(TEXT("Ask: no answer engine pathway available at dispatch time"), 73);
    bAskInFlight = false;
}

// Relay handlers for the external AnswerEngine path. These are UFUNCTION so they can
// subscribe to the dynamic multicasts on ULlamaComponent. Each gates on bAskInFlight
// to avoid bleeding unrelated direct-chat output through OnAsk* delegates.

void URagStore::RelayAnswerToken(const FString& Token)
{
    if (bAskInFlight) { OnAskTokenGenerated.Broadcast(Token); }
}
void URagStore::RelayAnswerPartial(const FString& Partial)
{
    if (bAskInFlight) { OnAskPartialGenerated.Broadcast(Partial); }
}
void URagStore::RelayAnswerMarkdownPartial(const FString& Partial, EMarkdownStreamState State)
{
    if (bAskInFlight) { OnAskMarkdownPartialGenerated.Broadcast(Partial, State); }
}
void URagStore::RelayAnswerResponse(const FString& Response)
{
    if (bAskInFlight) { OnAskResponseGenerated.Broadcast(Response); }
}
void URagStore::RelayAnswerEndOfStream(bool bStopSeq, float Tps)
{
    if (bAskInFlight)
    {
        OnAskEndOfStream.Broadcast(bStopSeq, Tps);
        bAskInFlight = false;
    }
}
void URagStore::RelayAnswerError(const FString& ErrorMessage, int32 ErrorCode)
{
    UE_LOG(LlamaLog, Warning, TEXT("URagStore relay error: %s"), *ErrorMessage);
    if (bAskInFlight)
    {
        OnAskError.Broadcast(ErrorMessage, ErrorCode);
        bAskInFlight = false;
    }
}
