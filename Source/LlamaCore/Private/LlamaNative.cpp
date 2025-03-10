// Copyright 2025-current Getnamo.

#include "LlamaNative.h"
#include <string>
#include "llama.h"
#include "common/common.h"

class FLlamaInternalState
{
public:
    //Core State
    llama_model* LlamaModel = nullptr;
    llama_context* Context = nullptr;
    llama_sampler* Sampler = nullptr;

    bool bIsLoaded = false;

    //Messaging state
    TArray<llama_chat_message> Messages;
    TArray<char> Formatted;

    bool LoadFromParams(FLLMModelParams& InModelParams)
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
        LlamaModelParams.n_gpu_layers = InModelParams.GPULayers;

        std::string Path = TCHAR_TO_UTF8(*InModelParams.PathToModel);
        LlamaModel = llama_model_load_from_file(Path.c_str(), LlamaModelParams);
        if (!LlamaModel)
        {
            UE_LOG(LlamaLog, Error, TEXT("%hs: error: unable to load model\n"), __func__);
            return false;
        }

        llama_context_params ContextParams = llama_context_default_params();
        ContextParams.n_ctx = InModelParams.MaxContextLength;
        ContextParams.n_batch = InModelParams.MaxContextLength;

        Context = llama_init_from_model(LlamaModel, ContextParams);
        if (!Context)
        {
            UE_LOG(LlamaLog, Error, TEXT("%hs: error: failed to create the llama_context\n"), __func__);
            return false;
        }

        Sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
        llama_sampler_chain_add(Sampler, llama_sampler_init_min_p(0.05f, 1));
        llama_sampler_chain_add(Sampler, llama_sampler_init_temp(0.8f));
        llama_sampler_chain_add(Sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

        Formatted.SetNum(llama_n_ctx(Context));

        bIsLoaded = true;

        return true;
    }

    void Unload()
    {
        if (Sampler)
        {
            llama_sampler_free(Sampler);
            Sampler = nullptr;
        }
        if (Context)
        {
            llama_free(Context);
            Context = nullptr;
        }
        if (LlamaModel)
        {
            llama_model_free(LlamaModel);
            LlamaModel = nullptr;
        }
        Formatted.Empty();

        bIsLoaded = false;
    }

    ~FLlamaInternalState()
    {
        Unload();
    }
};

FLlamaNative::FLlamaNative()
{
    Internal = new FLlamaInternalState();
}

FLlamaNative::~FLlamaNative()
{
    delete Internal;
}

FString FLlamaNative::Generate(const FString& PromptFString)
{
    std::string Prompt = TCHAR_TO_UTF8(*PromptFString);
    std::string Response;

    const llama_vocab* Vocab = llama_model_get_vocab(Internal->LlamaModel);

    const bool IsFirst = llama_get_kv_cache_used_cells(Internal->Context) == 0;

    // tokenize the prompt
    const int NPromptTokens = -llama_tokenize(Vocab, Prompt.c_str(), Prompt.size(), NULL, 0, IsFirst, true);
    std::vector<llama_token> PromptTokens(NPromptTokens);
    if (llama_tokenize(Vocab, Prompt.c_str(), Prompt.size(), PromptTokens.data(), PromptTokens.size(), IsFirst, true) < 0)
    {
        GGML_ABORT("failed to tokenize the prompt\n");
    }

    // prepare a batch for the prompt
    llama_batch Batch = llama_batch_get_one(PromptTokens.data(), PromptTokens.size());
    llama_token NewTokenId;
    while (true)
    {
        // check if we have enough space in the context to evaluate this batch
        int NContext = llama_n_ctx(Internal->Context);
        int NContextUsed = llama_get_kv_cache_used_cells(Internal->Context);
        if (NContextUsed + Batch.n_tokens > NContext)
        {
            UE_LOG(LlamaLog, Error, TEXT("\033[0m\n"));
            UE_LOG(LlamaLog, Error, TEXT("context size exceeded\n"));
            break;
        }

        if (llama_decode(Internal->Context, Batch))
        {
            GGML_ABORT("failed to decode\n");
        }

        // sample the next token
        NewTokenId = llama_sampler_sample(Internal->Sampler, Internal->Context, -1);

        // is it an end of generation?
        if (llama_vocab_is_eog(Vocab, NewTokenId))
        {
            break;
        }

        // convert the token to a string, print it and add it to the response
        char Buffer[256];
        int n = llama_token_to_piece(Vocab, NewTokenId, Buffer, sizeof(Buffer), 0, true);
        if (n < 0)
        {
            GGML_ABORT("failed to convert token to piece\n");
        }
        std::string Piece(Buffer, n);
        Response += Piece;

        // prepare the next batch with the sampled token
        Batch = llama_batch_get_one(&NewTokenId, 1);
    }
    return FString(UTF8_TO_TCHAR(Response.c_str()));
}

void FLlamaNative::SetModelParams(const FLLMModelParams& Params)
{
	ModelParams = Params;
}

bool FLlamaNative::LoadModel()
{
    //Unload first if any is loaded
    UnloadModel();

    //Now load it
    bool bSuccess = Internal->LoadFromParams(ModelParams);

    //Update model state

    //llama_n_ctx(Internal->Context);
    if (bSuccess)
    {
        ModelState.ChatTemplateLlamaString = FString(llama_model_chat_template(Internal->LlamaModel, /* name */ nullptr));
    }

    return bSuccess;
}

bool FLlamaNative::UnloadModel()
{
    if (bIsModelLoaded())
    {
        Internal->Unload();
    }
    return true;
}

bool FLlamaNative::bIsModelLoaded()
{
    return Internal->bIsLoaded;
}

void FLlamaNative::InsertPrompt(const FString& Prompt)
{
    if (!bIsModelLoaded())
    {
        UE_LOG(LlamaLog, Error, TEXT("Model isn't loaded, can't run prompt."));
        return;
    }
}
