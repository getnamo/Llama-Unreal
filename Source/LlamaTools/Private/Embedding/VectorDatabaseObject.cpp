// Copyright 2025-current Getnamo.

#include "Embedding/VectorDatabaseObject.h"

UVectorDatabase::UVectorDatabase()
{
    Native = MakeUnique<FVectorDatabase>();
}

UVectorDatabase::~UVectorDatabase() = default;

void UVectorDatabase::Initialize()
{
    Native->Params = Params;
    Native->InitializeDB();
}

bool UVectorDatabase::IsInitialized() const
{
    return Native.IsValid() && Native->IsInitialized();
}

int32 UVectorDatabase::Num() const
{
    return Native.IsValid() ? Native->Num() : 0;
}

void UVectorDatabase::Reset()
{
    if (Native) { Native->Reset(); }
}

void UVectorDatabase::AddEmbeddingWithId(const TArray<float>& Embedding, int64 UniqueId)
{
    Native->AddVectorEmbeddingIdPair(Embedding, UniqueId);
}

int64 UVectorDatabase::AddEmbeddingWithText(const TArray<float>& Embedding, const FString& Text)
{
    return Native->AddVectorEmbeddingStringPair(Embedding, Text);
}

void UVectorDatabase::FindNearestIds(const TArray<float>& QueryEmbedding, int32 N, TArray<int64>& OutIds)
{
    Native->FindNearestNIds(OutIds, QueryEmbedding, N);
}

void UVectorDatabase::FindNearestIdsWithDistance(const TArray<float>& QueryEmbedding, int32 N,
                                                 TArray<int64>& OutIds, TArray<float>& OutDistances)
{
    Native->FindNearestNIds(OutIds, OutDistances, QueryEmbedding, N);
}

void UVectorDatabase::FindNearestStrings(const TArray<float>& QueryEmbedding, int32 N, TArray<FString>& OutStrings)
{
    Native->FindNearestNStrings(OutStrings, QueryEmbedding, N);
}

void UVectorDatabase::FindNearestStringsWithDistance(const TArray<float>& QueryEmbedding, int32 N,
                                                     TArray<FString>& OutStrings, TArray<float>& OutDistances)
{
    Native->FindNearestNStrings(OutStrings, OutDistances, QueryEmbedding, N);
}

bool UVectorDatabase::SaveToFile(const FString& FilePath)
{
    return Native->Save(FilePath);
}

bool UVectorDatabase::LoadFromFile(const FString& FilePath)
{
    const bool bOk = Native->Load(FilePath);
    if (bOk) { Params = Native->Params; }
    return bOk;
}
