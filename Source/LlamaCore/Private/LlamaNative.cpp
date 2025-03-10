// Copyright 2025-current Getnamo.

#include "LlamaNative.h"
#include <string>
#include "llama.h"
#include "common/common.h"

void FLlamaNative::SetModelParams(const FLLMModelParams& Params)
{
	ModelParams = Params;
}

bool FLlamaNative::LoadModel()
{
    //Early implementation largely converted from: https://github.com/ggml-org/llama.cpp/blob/master/examples/simple-chat/simple-chat.cpp

    // only print errors
    llama_log_set([](enum ggml_log_level level, const char* text, void* /* user_data */) {
    if (level >= GGML_LOG_LEVEL_ERROR) {
        fprintf(stderr, "%s", text);
    }
    }, nullptr);

    // load dynamic backends
    ggml_backend_load_all();

    // initialize the model
    llama_model_params LlamaModelParams = llama_model_default_params();
    LlamaModelParams.n_gpu_layers = ModelParams.GPULayers;

    std::string Path = TCHAR_TO_UTF8(*ModelParams.PathToModel);
    llama_model* LlamaModel = llama_model_load_from_file(Path.c_str(), LlamaModelParams);
    if (!LlamaModel)
    {
        UE_LOG(LlamaLog, Error, TEXT("%hs: error: unable to load model\n"), __func__);
        return false;
    }

    const llama_vocab* Vocab = llama_model_get_vocab(LlamaModel);

    llama_context_params ContextParams = llama_context_default_params();
    ContextParams.n_ctx = ModelParams.MaxContextLength;
    ContextParams.n_batch = ModelParams.MaxContextLength;

    llama_context* Context = llama_init_from_model(LlamaModel, ContextParams);
    if (!Context)
    {
        UE_LOG(LlamaLog, Error, TEXT("%hs: error: failed to create the llama_context\n"), __func__);
        return false;
    }

    return true;
}

FLlamaNative::FLlamaNative()
{

}

FLlamaNative::~FLlamaNative()
{

}
