#include "LlamaInternal.h"
#include "LlamaInternal.h"
#include "LlamaUtility.h"

bool FLlamaInternal::LoadFromParams(FLLMModelParams& InModelParams)
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

    //FPlatform

    std::string Path = TCHAR_TO_UTF8(*FLlamaPaths::ParsePathIntoFullPath(InModelParams.PathToModel));
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

    Template = (char*)llama_model_chat_template(LlamaModel, /* name */ nullptr);
    PrevLen = 0;

    bIsLoaded = true;

    return true;
}

void FLlamaInternal::InsertPrompt(std::string& UserPrompt)
{
    Messages.Push({ "user", _strdup(UserPrompt.c_str()) });
    int NewLen = llama_chat_apply_template(Template, Messages.GetData(), Messages.Num(),
        true, Formatted.GetData(), Formatted.Num());


    if (NewLen > Formatted.Num())
    {
        Formatted.Reserve(NewLen);
        NewLen = llama_chat_apply_template(Template, Messages.GetData(), Messages.Num(),
            true, Formatted.GetData(), Formatted.Num());
    }
    if (NewLen < 0)
    {
        UE_LOG(LlamaLog, Warning, TEXT("failed to apply the chat template p1"));
        return;
    }

    std::string Prompt(Formatted.GetData() + PrevLen, Formatted.GetData() + NewLen);
    //FString Prompt = FString(UTF8_TO_TCHAR(PromptStd.c_str()));

    std::string Response = Generate(Prompt);
    //std::string AssistantStdString = TCHAR_TO_UTF8(*Response);

    Messages.Push({ "assistant", _strdup(Response.c_str()) });

    PrevLen = llama_chat_apply_template(Template, Messages.GetData(), Messages.Num(), false, nullptr, 0);
    if (PrevLen < 0)
    {
        UE_LOG(LlamaLog, Warning, TEXT("failed to apply the chat template p2"));
        return;
    }
}

void FLlamaInternal::Unload()
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

std::string FLlamaInternal::Generate(std::string& Prompt)
{
    std::string Response;

    const llama_vocab* Vocab = llama_model_get_vocab(LlamaModel);

    const bool IsFirst = llama_get_kv_cache_used_cells(Context) == 0;

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
        int NContext = llama_n_ctx(Context);
        int NContextUsed = llama_get_kv_cache_used_cells(Context);
        if (NContextUsed + Batch.n_tokens > NContext)
        {
            UE_LOG(LlamaLog, Error, TEXT("\033[0m\n"));
            UE_LOG(LlamaLog, Error, TEXT("context size exceeded\n"));
            break;
        }

        if (llama_decode(Context, Batch))
        {
            GGML_ABORT("failed to decode\n");
        }

        // sample the next token
        NewTokenId = llama_sampler_sample(Sampler, Context, -1);

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
    return Response;
}

FLlamaInternal::~FLlamaInternal()
{
    Unload();
}
