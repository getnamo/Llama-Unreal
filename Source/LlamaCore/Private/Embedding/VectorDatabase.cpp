// Copyright 2025-current Getnamo.

#include "Embedding/VectorDatabase.h"
#include "Misc/Paths.h"
#include "LlamaUtility.h"
#include "hnswlib/hnswlib.h"

void FVectorDatabase::BasicsTest()
{
    //Try: https://github.com/nmslib/hnswlib/blob/master/examples/cpp/EXAMPLES.md
    int dim = 16;               // Dimension of the elements
    int max_elements = 10000;   // Maximum number of elements, should be known beforehand
    int M = 16;                 // Tightly connected with internal dimensionality of the data
    // strongly affects the memory consumption
    int ef_construction = 200;  // Controls index search speed/build speed tradeoff

    // Initing index
    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);

    // Generate random data
    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib_real;
    float* data = new float[dim * max_elements];
    for (int i = 0; i < dim * max_elements; i++) 
    {
        data[i] = distrib_real(rng);
    }

    // Add data to index
    for (int i = 0; i < max_elements; i++)
    {
        alg_hnsw->addPoint(data + i * dim, i);
    }

    // Query the elements for themselves and measure recall
    float correct = 0;
    for (int i = 0; i < max_elements; i++) 
    {
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnn(data + i * dim, 1);
        hnswlib::labeltype label = result.top().second;
        if (label == i) correct++;
    }
    float recall = correct / max_elements;

    UE_LOG(LogTemp, Log, TEXT("Recall: %1.3f"), recall);

    // Serialize index
    FString SavePath = FPaths::ProjectSavedDir() / TEXT("hnsw.bin");
    std::string hnsw_path = FLlamaString::ToStd(SavePath);
    alg_hnsw->saveIndex(hnsw_path);
    delete alg_hnsw;

    // Deserialize index and check recall
    // This test appears to fail in unreal context (loading index)
    alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, hnsw_path, false, max_elements);
    //alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);
    //alg_hnsw->loadIndex(hnsw_path, &space);

    if (alg_hnsw->getMaxElements() > 0)
    {
        correct = 0;
        for (int i = 0; i < alg_hnsw->getMaxElements(); i++)
        {
            std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnn(data + i * dim, 1);
            hnswlib::labeltype label = result.top().second;
            if (label == i) correct++;
        }
        recall = (float)correct / max_elements;
        UE_LOG(LogTemp, Log, TEXT("Recall of deserialized index: %1.3f"), recall);
    }
    else
    {
        UE_LOG(LogTemp, Log, TEXT("Failed to load index from file correctly"));
    }
    delete[] data;
    delete alg_hnsw;
}

FVectorDatabase::FVectorDatabase()
{
    //TODO: see https://github.com/ggml-org/llama.cpp/blob/master/examples/embedding/embedding.cpp

    //llamainternal
    //1. Load model
    //2. Tokenize input
    //3. llama_model_n_embd & embeddings for input allocation
    //4. batch_decode
    //5. store embeddings in index.
    //6. potentially save index
}

FVectorDatabase::~FVectorDatabase()
{

}