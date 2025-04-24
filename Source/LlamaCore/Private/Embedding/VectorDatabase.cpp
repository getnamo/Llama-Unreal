// Copyright 2025-current Getnamo.

#include "Embedding/VectorDatabase.h"
#include "Misc/Paths.h"
#include "hnswlib/hnswlib.h"
#include "LlamaUtility.h"

class FHNSWPrivate
{
public:
    hnswlib::HierarchicalNSW<float>* HNSW = nullptr;

    void ReleaseHNSWIfAllocated()
    {
        if (HNSW)
        {
            delete HNSW;
            HNSW = nullptr;
        }
    }
    ~FHNSWPrivate()
    {
        ReleaseHNSWIfAllocated();
    }
};

void FVectorDatabase::BasicsTest()
{
    //Try: https://github.com/nmslib/hnswlib/blob/master/examples/cpp/EXAMPLES.md

    InitializeDB();

    // Generate random data
    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib_real;
    float* data = new float[Params.Dimensions * Params.MaxElements];
    for (int i = 0; i < Params.Dimensions * Params.MaxElements; i++)
    {
        data[i] = distrib_real(rng);
    }

    // Add data to index
    for (int i = 0; i < Params.MaxElements; i++)
    {
        Private->HNSW->addPoint(data + i * Params.Dimensions, i);
    }

    // Query the elements for themselves and measure recall
    float correct = 0;
    for (int i = 0; i < Params.MaxElements; i++)
    {
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = Private->HNSW->searchKnn(data + i * Params.Dimensions, 1);
        hnswlib::labeltype label = result.top().second;
        if (label == i) correct++;
    }
    float recall = correct / Params.MaxElements;

    UE_LOG(LogTemp, Log, TEXT("Recall: %1.3f"), recall);

    // Serialize index
    FString SavePath = FPaths::ProjectSavedDir() / TEXT("hnsw.bin");
    std::string HNSWPath = FLlamaString::ToStd(SavePath);
    Private->HNSW->saveIndex(HNSWPath);
    delete Private->HNSW;

    // Deserialize index and check recall
    // This test appears to fail in unreal context (loading index)
    hnswlib::L2Space Space(Params.Dimensions);
    Private->HNSW = new hnswlib::HierarchicalNSW<float>(&Space, HNSWPath, false, Params.MaxElements);
    //HNSW = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);
    //HNSW->loadIndex(HNSWPath, &space);

    if (Private->HNSW->getMaxElements() > 0)
    {
        correct = 0;
        for (int i = 0; i < Private->HNSW->getMaxElements(); i++)
        {
            std::priority_queue<std::pair<float, hnswlib::labeltype>> result = Private->HNSW->searchKnn(data + i * Params.Dimensions, 1);
            hnswlib::labeltype label = result.top().second;
            if (label == i) correct++;
        }
        recall = (float)correct / Params.MaxElements;
        UE_LOG(LogTemp, Log, TEXT("Recall of deserialized index: %1.3f"), recall);
    }
    else
    {
        UE_LOG(LogTemp, Log, TEXT("Failed to load index from file correctly"));
    }
    delete[] data;
    //delete HNSW; //handled at deconstructor atm
}

void FVectorDatabase::InitializeDB()
{
    //Delete and re-initialize as needed
    Private->ReleaseHNSWIfAllocated();

    hnswlib::L2Space Space(Params.Dimensions);
    Private->HNSW = new hnswlib::HierarchicalNSW<float>(&Space, Params.MaxElements, Params.M, Params.EFConstruction);
}

FVectorDatabase::FVectorDatabase()
{
    Private = new FHNSWPrivate();

    //llamainternal
    //1. Load model
    //2. Tokenize input
    //3. llama_model_n_embd & embeddings for input allocation
    //4. batch_decode
    
    //1-4 now works within llama internal & native. 

    //5. store embeddings in index.
    //6. potentially save index

    //we should store and retrieve stored embeddings
    //What's a good api for VectorDB?
}

FVectorDatabase::~FVectorDatabase()
{
    delete Private;
    Private = nullptr;
}