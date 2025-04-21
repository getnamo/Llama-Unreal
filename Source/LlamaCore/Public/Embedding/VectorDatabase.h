// Copyright 2025-current Getnamo.

#pragma once

#include "hnswlib/hnswlib.h"
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
*/
class FVectorDatabase
{
public:

    FVectorDBParams Params;

    //Simple test to see if the basics run
    void BasicsTest();

    //Initializes from current Params
    void InitializeDB();

    FVectorDatabase();
    ~FVectorDatabase();

private:
    hnswlib::HierarchicalNSW<float>* HNSW;
};