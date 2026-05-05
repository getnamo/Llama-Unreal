// Copyright 2025-current Getnamo.

#pragma once

#include "CoreMinimal.h"
#include "UObject/Object.h"
#include "Embedding/VectorDatabase.h"
#include "VectorDatabaseObject.generated.h"

/**
 * Blueprint-callable wrapper around FVectorDatabase. Owns a single FVectorDatabase
 * instance for its lifetime. Use this from Blueprints; native callers can keep using
 * FVectorDatabase directly.
 */
UCLASS(Blueprintable, BlueprintType, ClassGroup = "LLM")
class LLAMACORE_API UVectorDatabase : public UObject
{
    GENERATED_BODY()
public:
    UVectorDatabase();
    virtual ~UVectorDatabase();

    /** Edit before calling Initialize. Dimensions must match the embedding model. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VectorDB")
    FVectorDBParams Params;

    UFUNCTION(BlueprintCallable, Category = "VectorDB")
    void Initialize();

    UFUNCTION(BlueprintPure, Category = "VectorDB")
    bool IsInitialized() const;

    UFUNCTION(BlueprintPure, Category = "VectorDB")
    int32 Num() const;

    UFUNCTION(BlueprintCallable, Category = "VectorDB")
    void Reset();

    /** Adds an embedding with a caller-supplied id. */
    UFUNCTION(BlueprintCallable, Category = "VectorDB")
    void AddEmbeddingWithId(const TArray<float>& Embedding, int64 UniqueId);

    /** Adds an embedding paired with a text snippet; returns the auto-assigned id (-1 on failure). */
    UFUNCTION(BlueprintCallable, Category = "VectorDB")
    int64 AddEmbeddingWithText(const TArray<float>& Embedding, const FString& Text);

    /** Top-N id lookup, sorted nearest-first. */
    UFUNCTION(BlueprintCallable, Category = "VectorDB")
    void FindNearestIds(const TArray<float>& QueryEmbedding, int32 N, TArray<int64>& OutIds);

    /** Top-N id+distance lookup, sorted nearest-first. */
    UFUNCTION(BlueprintCallable, Category = "VectorDB")
    void FindNearestIdsWithDistance(const TArray<float>& QueryEmbedding, int32 N,
                                    TArray<int64>& OutIds, TArray<float>& OutDistances);

    /** Top-N text lookup (skips ids without text), sorted nearest-first. */
    UFUNCTION(BlueprintCallable, Category = "VectorDB")
    void FindNearestStrings(const TArray<float>& QueryEmbedding, int32 N, TArray<FString>& OutStrings);

    /** Top-N text+distance lookup, sorted nearest-first. */
    UFUNCTION(BlueprintCallable, Category = "VectorDB")
    void FindNearestStringsWithDistance(const TArray<float>& QueryEmbedding, int32 N,
                                        TArray<FString>& OutStrings, TArray<float>& OutDistances);

    UFUNCTION(BlueprintCallable, Category = "VectorDB|Persistence")
    bool SaveToFile(const FString& FilePath);

    UFUNCTION(BlueprintCallable, Category = "VectorDB|Persistence")
    bool LoadFromFile(const FString& FilePath);

    /** Native accessor for advanced users. */
    FVectorDatabase& GetNative() { return *Native; }
    const FVectorDatabase& GetNative() const { return *Native; }

private:
    TUniquePtr<FVectorDatabase> Native;
};
