// Copyright 2025-current Getnamo.

#include "Embedding/VectorDatabase.h"

#include "LlamaUtility.h"
#include "Misc/Paths.h"
#include "Misc/FileHelper.h"
#include "Serialization/MemoryReader.h"
#include "Serialization/MemoryWriter.h"
#include "HAL/PlatformFileManager.h"
#include "HAL/FileManager.h"

#include "hnswlib/hnswlib.h"

#include <random>

namespace
{
    // Versioned magic header so future format changes don't silently corrupt loads.
    constexpr uint32 VDB_MAGIC = 0x56444231; // 'VDB1'
    constexpr uint32 VDB_VERSION = 1;
}

class FHNSWPrivate
{
public:
    TUniquePtr<hnswlib::L2Space> Space;
    hnswlib::HierarchicalNSW<float>* HNSW = nullptr;

    void Initialize(const FVectorDBParams& Params)
    {
        Release();
        Space = MakeUnique<hnswlib::L2Space>(static_cast<size_t>(Params.Dimensions));
        HNSW = new hnswlib::HierarchicalNSW<float>(
            Space.Get(),
            static_cast<size_t>(Params.MaxElements),
            static_cast<size_t>(Params.M),
            static_cast<size_t>(Params.EFConstruction));
        HNSW->setEf(static_cast<size_t>(Params.EFQuery));
    }

    bool LoadFromFile(const FVectorDBParams& Params, const std::string& Path)
    {
        Release();
        Space = MakeUnique<hnswlib::L2Space>(static_cast<size_t>(Params.Dimensions));
        try
        {
            HNSW = new hnswlib::HierarchicalNSW<float>(
                Space.Get(), Path, /*nmslib*/ false,
                static_cast<size_t>(Params.MaxElements),
                /*replace_deleted*/ false);
            HNSW->setEf(static_cast<size_t>(Params.EFQuery));
            return true;
        }
        catch (const std::exception& Ex)
        {
            UE_LOG(LlamaLog, Warning, TEXT("FVectorDatabase: HNSW load failed: %hs"), Ex.what());
            Release();
            return false;
        }
    }

    void Release()
    {
        if (HNSW)
        {
            delete HNSW;
            HNSW = nullptr;
        }
        Space.Reset();
    }

    ~FHNSWPrivate() { Release(); }
};

// ---- ctor / dtor / reset ----------------------------------------------------

FVectorDatabase::FVectorDatabase()
{
    Private = new FHNSWPrivate();
}

FVectorDatabase::~FVectorDatabase()
{
    delete Private;
    Private = nullptr;
}

void FVectorDatabase::InitializeDB()
{
    Private->Initialize(Params);
    {
        FScopeLock Lock(&TextLock);
        TextDatabase.Empty();
        TextDatabaseMaxId = 0;
    }
    bInitialized = true;
}

bool FVectorDatabase::IsInitialized() const
{
    return bInitialized && Private && Private->HNSW;
}

int32 FVectorDatabase::Num() const
{
    if (!IsInitialized()) { return 0; }
    return static_cast<int32>(Private->HNSW->getCurrentElementCount());
}

void FVectorDatabase::Reset()
{
    if (Private)
    {
        Private->Release();
    }
    {
        FScopeLock Lock(&TextLock);
        TextDatabase.Empty();
        TextDatabaseMaxId = 0;
    }
    bInitialized = false;
}

// ---- Add --------------------------------------------------------------------

void FVectorDatabase::AddVectorEmbeddingIdPair(const TArray<float>& Embedding, int64 UniqueId)
{
    if (!ensureMsgf(IsInitialized(),
        TEXT("FVectorDatabase::AddVectorEmbeddingIdPair called before InitializeDB()")))
    {
        return;
    }
    if (!ensureMsgf(Embedding.Num() == Params.Dimensions,
        TEXT("FVectorDatabase: embedding dim %d != Params.Dimensions %d"),
        Embedding.Num(), Params.Dimensions))
    {
        return;
    }

    Private->HNSW->addPoint(static_cast<const void*>(Embedding.GetData()),
                            static_cast<hnswlib::labeltype>(UniqueId));
}

int64 FVectorDatabase::AddVectorEmbeddingStringPair(const TArray<float>& Embedding, const FString& Text)
{
    if (!IsInitialized() || Embedding.Num() != Params.Dimensions)
    {
        return -1;
    }

    int64 UniqueId;
    {
        FScopeLock Lock(&TextLock);
        TextDatabaseMaxId++;
        UniqueId = TextDatabaseMaxId;
        TextDatabase.Add(UniqueId, Text);
    }

    AddVectorEmbeddingIdPair(Embedding, UniqueId);
    return UniqueId;
}

// ---- Query ------------------------------------------------------------------

int64 FVectorDatabase::FindNearestId(const TArray<float>& ForEmbedding)
{
    TArray<int64> Ids;
    FindNearestNIds(Ids, ForEmbedding, 1);
    return Ids.Num() > 0 ? Ids[0] : -1;
}

FString FVectorDatabase::FindNearestString(const TArray<float>& ForEmbedding)
{
    TArray<FString> Strings;
    FindNearestNStrings(Strings, ForEmbedding, 1);
    return Strings.Num() > 0 ? Strings[0] : FString();
}

void FVectorDatabase::FindNearestNIds(TArray<int64>& OutIds, const TArray<float>& ForEmbedding, int32 N)
{
    TArray<float> Distances;
    FindNearestNIds(OutIds, Distances, ForEmbedding, N);
}

void FVectorDatabase::FindNearestNIds(TArray<int64>& OutIds, TArray<float>& OutDistances,
                                      const TArray<float>& ForEmbedding, int32 N)
{
    OutIds.Reset();
    OutDistances.Reset();

    if (!IsInitialized() || N <= 0) { return; }
    if (ForEmbedding.Num() != Params.Dimensions)
    {
        UE_LOG(LlamaLog, Warning,
            TEXT("FVectorDatabase: query embedding dim %d != Params.Dimensions %d"),
            ForEmbedding.Num(), Params.Dimensions);
        return;
    }
    if (Private->HNSW->getCurrentElementCount() == 0) { return; }

    // hnswlib returns a max-heap of (distance, label); top is FARTHEST among the K.
    // Pop into temp arrays and reverse so index 0 is the nearest.
    std::priority_queue<std::pair<float, hnswlib::labeltype>> Results =
        Private->HNSW->searchKnn(static_cast<const void*>(ForEmbedding.GetData()), static_cast<size_t>(N));

    const int32 Count = static_cast<int32>(Results.size());
    OutIds.SetNumUninitialized(Count);
    OutDistances.SetNumUninitialized(Count);

    // Fill from back so popped (farthest-first) ends up at the end.
    for (int32 i = Count - 1; i >= 0; --i)
    {
        const auto& Top = Results.top();
        OutDistances[i] = Top.first;
        OutIds[i]       = static_cast<int64>(Top.second);
        Results.pop();
    }
}

void FVectorDatabase::FindNearestNStrings(TArray<FString>& OutStrings, const TArray<float>& ForEmbedding, int32 N)
{
    TArray<float> Distances;
    FindNearestNStrings(OutStrings, Distances, ForEmbedding, N);
}

void FVectorDatabase::FindNearestNStrings(TArray<FString>& OutStrings, TArray<float>& OutDistances,
                                          const TArray<float>& ForEmbedding, int32 N)
{
    OutStrings.Reset();
    OutDistances.Reset();

    TArray<int64> Ids;
    TArray<float> Distances;
    FindNearestNIds(Ids, Distances, ForEmbedding, N);

    FScopeLock Lock(&TextLock);
    for (int32 i = 0; i < Ids.Num(); ++i)
    {
        if (const FString* Hit = TextDatabase.Find(Ids[i]))
        {
            OutStrings.Add(*Hit);
            OutDistances.Add(Distances[i]);
        }
    }
}

bool FVectorDatabase::TryGetText(int64 UniqueId, FString& OutText) const
{
    FScopeLock Lock(&TextLock);
    if (const FString* Hit = TextDatabase.Find(UniqueId))
    {
        OutText = *Hit;
        return true;
    }
    return false;
}

// ---- Persistence ------------------------------------------------------------

bool FVectorDatabase::Save(const FString& FilePath) const
{
    if (!IsInitialized())
    {
        UE_LOG(LlamaLog, Warning, TEXT("FVectorDatabase::Save called before InitializeDB"));
        return false;
    }

    // Make sure the destination directory exists.
    const FString DirOnly = FPaths::GetPath(FilePath);
    if (!DirOnly.IsEmpty())
    {
        IFileManager::Get().MakeDirectory(*DirOnly, /*Tree*/ true);
    }

    // hnswlib only knows std::ofstream — write the index to a sibling temp file, then
    // append it as a binary blob into our own framed format so callers see one file.
    const FString TempIndexPath = FilePath + TEXT(".hnsw.tmp");
    {
        const std::string StdPath = FLlamaString::ToStd(TempIndexPath);
        try
        {
            Private->HNSW->saveIndex(StdPath);
        }
        catch (const std::exception& Ex)
        {
            UE_LOG(LlamaLog, Warning, TEXT("FVectorDatabase::Save HNSW write failed: %hs"), Ex.what());
            return false;
        }
    }

    TArray<uint8> HnswBytes;
    if (!FFileHelper::LoadFileToArray(HnswBytes, *TempIndexPath))
    {
        UE_LOG(LlamaLog, Warning, TEXT("FVectorDatabase::Save could not read temp HNSW file %s"), *TempIndexPath);
        IFileManager::Get().Delete(*TempIndexPath, false, true, true);
        return false;
    }
    IFileManager::Get().Delete(*TempIndexPath, false, true, true);

    TArray<uint8> Buffer;
    FMemoryWriter Writer(Buffer, /*bIsPersistent*/ true);
    uint32 Magic = VDB_MAGIC;
    uint32 Version = VDB_VERSION;
    Writer << Magic;
    Writer << Version;

    int32 Dim    = Params.Dimensions;
    int32 MaxEl  = Params.MaxElements;
    int32 M      = Params.M;
    int32 EFC    = Params.EFConstruction;
    int32 EFQ    = Params.EFQuery;
    Writer << Dim << MaxEl << M << EFC << EFQ;

    int64 MaxIdCopy;
    {
        FScopeLock Lock(&TextLock);
        MaxIdCopy = TextDatabaseMaxId;

        int32 TextCount = TextDatabase.Num();
        Writer << MaxIdCopy;
        Writer << TextCount;
        for (auto& Pair : TextDatabase)
        {
            int64 K = Pair.Key;
            FString V = Pair.Value;
            Writer << K;
            Writer << V;
        }
    }

    int64 HnswSize = static_cast<int64>(HnswBytes.Num());
    Writer << HnswSize;
    Writer.Serialize(HnswBytes.GetData(), HnswBytes.Num());

    return FFileHelper::SaveArrayToFile(Buffer, *FilePath);
}

bool FVectorDatabase::Load(const FString& FilePath)
{
    TArray<uint8> Buffer;
    if (!FFileHelper::LoadFileToArray(Buffer, *FilePath))
    {
        UE_LOG(LlamaLog, Warning, TEXT("FVectorDatabase::Load could not read %s"), *FilePath);
        return false;
    }

    FMemoryReader Reader(Buffer, /*bIsPersistent*/ true);
    uint32 Magic = 0, Version = 0;
    Reader << Magic;
    Reader << Version;
    if (Magic != VDB_MAGIC)
    {
        UE_LOG(LlamaLog, Warning, TEXT("FVectorDatabase::Load bad magic in %s"), *FilePath);
        return false;
    }
    if (Version != VDB_VERSION)
    {
        UE_LOG(LlamaLog, Warning, TEXT("FVectorDatabase::Load version mismatch %u != %u"), Version, VDB_VERSION);
        return false;
    }

    int32 Dim, MaxEl, M, EFC, EFQ;
    Reader << Dim << MaxEl << M << EFC << EFQ;

    Params.Dimensions     = Dim;
    Params.MaxElements    = MaxEl;
    Params.M              = M;
    Params.EFConstruction = EFC;
    Params.EFQuery        = EFQ;

    int64 MaxIdRead = 0;
    int32 TextCount = 0;
    Reader << MaxIdRead;
    Reader << TextCount;

    TMap<int64, FString> NewText;
    NewText.Reserve(TextCount);
    for (int32 i = 0; i < TextCount; ++i)
    {
        int64 K = 0;
        FString V;
        Reader << K;
        Reader << V;
        NewText.Add(K, MoveTemp(V));
    }

    int64 HnswSize = 0;
    Reader << HnswSize;
    if (HnswSize <= 0 || HnswSize > static_cast<int64>(Buffer.Num()))
    {
        UE_LOG(LlamaLog, Warning, TEXT("FVectorDatabase::Load suspicious HNSW size %lld"), HnswSize);
        return false;
    }

    TArray<uint8> HnswBytes;
    HnswBytes.SetNumUninitialized(static_cast<int32>(HnswSize));
    Reader.Serialize(HnswBytes.GetData(), HnswBytes.Num());

    // Round-trip via a temp file because hnswlib::loadIndex takes a path, not a stream.
    const FString TempIndexPath = FilePath + TEXT(".hnsw.tmp");
    if (!FFileHelper::SaveArrayToFile(HnswBytes, *TempIndexPath))
    {
        UE_LOG(LlamaLog, Warning, TEXT("FVectorDatabase::Load could not write temp HNSW file"));
        return false;
    }

    bool bOk = Private->LoadFromFile(Params, FLlamaString::ToStd(TempIndexPath));
    IFileManager::Get().Delete(*TempIndexPath, false, true, true);

    if (!bOk) { return false; }

    {
        FScopeLock Lock(&TextLock);
        TextDatabase = MoveTemp(NewText);
        TextDatabaseMaxId = MaxIdRead;
    }
    bInitialized = true;
    return true;
}

// ---- Diagnostics ------------------------------------------------------------

float FVectorDatabase::BasicsTest()
{
    InitializeDB();

    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<float> distrib_real;

    const int32 D = Params.Dimensions;
    const int32 N = FMath::Min(Params.MaxElements, 1000);

    TArray<float> Data;
    Data.SetNumUninitialized(D * N);
    for (int32 i = 0; i < D * N; ++i)
    {
        Data[i] = distrib_real(rng);
    }

    for (int32 i = 0; i < N; ++i)
    {
        TArray<float> Slice(Data.GetData() + i * D, D);
        AddVectorEmbeddingIdPair(Slice, i);
    }

    int32 Correct = 0;
    for (int32 i = 0; i < N; ++i)
    {
        TArray<float> Slice(Data.GetData() + i * D, D);
        if (FindNearestId(Slice) == i) { ++Correct; }
    }
    const float Recall = static_cast<float>(Correct) / static_cast<float>(N);
    UE_LOG(LlamaLog, Log, TEXT("FVectorDatabase::BasicsTest recall=%1.3f (N=%d, D=%d)"), Recall, N, D);
    return Recall;
}
