#include "Internal/LlamaInternal.h"
#include "LlamaDataTypes.h"
#include "LlamaUtility.h"

bool FLlamaInternal::LoadModelFromParams(const FLLMModelParams& InModelParams)
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
    ContextParams.n_batch = InModelParams.MaxBatchLength;
    ContextParams.n_threads = InModelParams.Threads;
    ContextParams.n_threads_batch = InModelParams.Threads;

    Context = llama_init_from_model(LlamaModel, ContextParams);
    if (!Context)
    {
        UE_LOG(LlamaLog, Error, TEXT("%hs: error: failed to create the llama_context\n"), __func__);
        return false;
    }

    Sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());

    //Temperature is always applied
    llama_sampler_chain_add(Sampler, llama_sampler_init_temp(InModelParams.Advanced.Temp));

    //If any of the repeat penalties are set, apply penalties to sampler
    if (InModelParams.Advanced.PenaltyLastN != 0 || 
        InModelParams.Advanced.PenaltyRepeat != 1.f ||
        InModelParams.Advanced.PenaltyFrequency != 0.f ||
        InModelParams.Advanced.PenaltyPresence != 0.f)
    {
        llama_sampler_chain_add(Sampler, llama_sampler_init_penalties(
            InModelParams.Advanced.PenaltyLastN, InModelParams.Advanced.PenaltyRepeat,
            InModelParams.Advanced.PenaltyFrequency, InModelParams.Advanced.PenaltyPresence));
    }
    
    //Optional sampling strategies - MinP should be applied by default of 0.05f
    if (InModelParams.Advanced.MinP != -1.f)
    {
        llama_sampler_chain_add(Sampler, llama_sampler_init_min_p(InModelParams.Advanced.MinP, 1));
    }
    if (InModelParams.Advanced.TopK != -1.f)
    {
        llama_sampler_chain_add(Sampler, llama_sampler_init_top_k(InModelParams.Advanced.TopK));
    }
    if (InModelParams.Advanced.TopP != -1.f)
    {
        llama_sampler_chain_add(Sampler, llama_sampler_init_top_p(InModelParams.Advanced.TopP, 1));
    }
    if (InModelParams.Advanced.TypicalP != -1.f)
    {
        llama_sampler_chain_add(Sampler, llama_sampler_init_typical(InModelParams.Advanced.TypicalP, 1));
    }
    if (InModelParams.Advanced.MirostatSeed != -1)
    {
        llama_sampler_chain_add(Sampler, llama_sampler_init_mirostat_v2(
            InModelParams.Advanced.MirostatSeed, InModelParams.Advanced.MirostatTau, InModelParams.Advanced.MirostatEta));
    }

    //Seed is either default or the one specifically passed in for deterministic results
    if (InModelParams.Seed == -1)
    {
        llama_sampler_chain_add(Sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
    }
    else
    {
        llama_sampler_chain_add(Sampler, llama_sampler_init_dist(InModelParams.Seed));
    }
    

    ContextHistory.SetNum(llama_n_ctx(Context));

    //Prioritize: custom jinja, then name, then default
    if (!InModelParams.CustomChatTemplate.Jinja.IsEmpty())
    {
        Template = FLlamaString::ToStd(InModelParams.CustomChatTemplate.Jinja);
    }
    else if (   !InModelParams.CustomChatTemplate.TemplateSource.IsEmpty() &&
                InModelParams.CustomChatTemplate.TemplateSource != TEXT("tokenizer.chat_template"))
    {
        //apply template source name
        Template = std::string(llama_model_chat_template(LlamaModel, FLlamaString::ToStd(InModelParams.CustomChatTemplate.TemplateSource).c_str()));
    }
    else
    {
        //use default template
        Template = std::string(llama_model_chat_template(LlamaModel, nullptr));
    }
    
    PrevLen = 0;

    bIsModelLoaded = true;

    return true;
}

void FLlamaInternal::UnloadModel()
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
    ContextHistory.Empty();

    bIsModelLoaded = false;
}

void FLlamaInternal::StopGeneration()
{
    bGenerationActive = false;
}

bool FLlamaInternal::IsGenerating()
{
    return bGenerationActive;
}

int32 FLlamaInternal::MaxContext()
{
    if (Context)
    {
        return llama_n_ctx(Context);
    }
    else
    {
        return 0;
    }
}

int32 FLlamaInternal::UsedContext()
{
    if (Context)
    {
        return llama_get_kv_cache_used_cells(Context);
    }
    else
    {
        return 0;
    }
}

bool FLlamaInternal::IsModelLoaded()
{
    return bIsModelLoaded;
}

std::string FLlamaInternal::InsertRawPrompt(const std::string& Prompt)
{
    if (!bIsModelLoaded)
    {
        UE_LOG(LlamaLog, Warning, TEXT("Model isn't loaded"));
        return "";
    }
    std::string Response = Generate(Prompt);

    Messages.Push({ "assistant", _strdup(Response.c_str()) });

    PrevLen = llama_chat_apply_template(Template.c_str(), Messages.GetData(), Messages.Num(), false, nullptr, 0);
    if (PrevLen < 0)
    {
        UE_LOG(LlamaLog, Warning, TEXT("failed to apply the chat template post generation."));
        return "";
    }

    return Response;
}

std::string FLlamaInternal::InsertTemplatedPrompt(const std::string& UserPrompt)
{
    if (!bIsModelLoaded)
    {
        UE_LOG(LlamaLog, Warning, TEXT("Model isn't loaded"));
        return "";
    }

    int NewLen = PrevLen;

    if (!UserPrompt.empty())
    {
        Messages.Push({ "user", _strdup(UserPrompt.c_str()) });
        NewLen = llama_chat_apply_template(Template.c_str(), Messages.GetData(), Messages.Num(),
            true, ContextHistory.GetData(), ContextHistory.Num());


        if (NewLen > ContextHistory.Num())
        {
            ContextHistory.Reserve(NewLen);
            NewLen = llama_chat_apply_template(Template.c_str(), Messages.GetData(), Messages.Num(),
                true, ContextHistory.GetData(), ContextHistory.Num());
        }
        if (NewLen < 0)
        {
            UE_LOG(LlamaLog, Warning, TEXT("failed to apply the chat template pre generation."));
            return "";
        }
    }

    std::string Prompt(ContextHistory.GetData() + PrevLen, ContextHistory.GetData() + NewLen);

    std::string Response = Generate(Prompt);

    Messages.Push({ "assistant", _strdup(Response.c_str()) });

    PrevLen = llama_chat_apply_template(Template.c_str(), Messages.GetData(), Messages.Num(), false, ContextHistory.GetData(), ContextHistory.Num());
    if (PrevLen < 0)
    {
        UE_LOG(LlamaLog, Warning, TEXT("failed to apply the chat template post generation."));
        return "";
    }

    return Response;
}

std::string FLlamaInternal::ResumeGeneration()
{
    //run an empty user prompt
    return InsertTemplatedPrompt(std::string());
}

std::string FLlamaInternal::Generate(const std::string& Prompt)
{
    bGenerationActive = true;

    std::string Response;

    const llama_vocab* Vocab = llama_model_get_vocab(LlamaModel);

    const bool IsFirst = llama_get_kv_cache_used_cells(Context) == 0;

    // tokenize the prompt
    const int NPromptTokens = -llama_tokenize(Vocab, Prompt.c_str(), Prompt.size(), NULL, 0, IsFirst, true);
    std::vector<llama_token> PromptTokens(NPromptTokens);
    if (llama_tokenize(Vocab, Prompt.c_str(), Prompt.size(), PromptTokens.data(), PromptTokens.size(), IsFirst, true) < 0)
    {
        bGenerationActive = false;
        GGML_ABORT("failed to tokenize the prompt\n");
    }

    // prepare a batch for the prompt
    llama_batch Batch = llama_batch_get_one(PromptTokens.data(), PromptTokens.size());
    llama_token NewTokenId;
    
    while (bGenerationActive) //processing can be aborted by flipping the boolean
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
            bGenerationActive = false;
            GGML_ABORT("failed to convert token to piece\n");
            break;
        }
        std::string Piece(Buffer, n);
        Response += Piece;

        if (OnTokenGenerated)
        {
            OnTokenGenerated(Piece);
        }

        // prepare the next batch with the sampled token
        Batch = llama_batch_get_one(&NewTokenId, 1);
    }

    bGenerationActive = false;

    return Response;
}

FLlamaInternal::~FLlamaInternal()
{
    OnTokenGenerated = nullptr;
    UnloadModel();
}
