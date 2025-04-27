// Copyright 2025-current Getnamo.

#pragma once

#include "VectorDatabase.generated.h"

USTRUCT(BlueprintType)
struct FVectorDBParams
{
    GENERATED_USTRUCT_BODY();

    // Dimension of the elements, typically 1024
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VectorDB Params")
    int32 Dimensions = 16;               

    // Maximum number of elements, should be known beforehand
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VectorDB Params")
    int32 MaxElements = 1000;   

    // Tightly connected with internal dimensionality of the data
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VectorDB Params")
    int32 M = 16;                 

    // Controls index search speed/build speed tradeoff, strongly affects the memory consumption
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VectorDB Params")
    int32 EFConstruction = 200;
};



/** 
* Unreal style native wrapper for HNSW nearest neighbor search for high dimensional vectors
* !! v0.9.8 note NB: this class is a work in progress, currently not working yet !!
*/
class FVectorDatabase
{
public:



    FVectorDBParams Params;

    //Simple test to see if the basics run
    void BasicsTest();

    //Initializes from current Params
    void InitializeDB();

    //Adding Vectors
    //Add a high dimensional vector pair with a unique db id
    void AddVectorEmbeddingIdPair(const TArray<float>& Embedding, int64 UniqueId);

    //Add a high dimensional vector pair with it's text source
    //this will internally create a DB entry
    void AddVectorEmbeddingStringPair(const TArray<float>& Embedding, const FString& Text);

    //Lookup single top entry
    int64 FindNearestId(const TArray<float>& ForEmbedding);
    FString FindNearestString(const TArray<float>& ForEmbedding);

    //Lookup group entries
    void FindNearestNIds(TArray<int64>& IdResults, const TArray<float>& ForEmbedding, int32 N = 1);
    void FindNearestNStrings(TArray<FString>& StringResults, const TArray<float>& ForEmbedding, int32 N = 1);

    //todo: add version that returns n nearest

    FVectorDatabase();
    ~FVectorDatabase();

private:
    class FHNSWPrivate* Private = nullptr;

    //Stores the embedded text database. Use UniqueDBId (aka primary key) to lookup the text snippet
    TMap<int64, FString> TextDatabase;
    int64 TextDatabaseMaxId = 0;
};