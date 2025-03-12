#include "Internal/LlamaInternal.h"
#include "Internal/common.h"
#include "Internal/sampling.h"
#include "LlamaDataTypes.h"
#include "LlamaUtility.h"
#include "HardwareInfo.h"

bool FLlamaInternal::LoadModelFromParams(const FLLMModelParams& InModelParams)
{
    FString RHI = FHardwareInfo::GetHardwareDetailsString();
    FString GPU = FPlatformMisc::GetPrimaryGPUBrand();

    UE_LOG(LogTemp, Log, TEXT("Device Found: %s %s"), *GPU, *RHI);

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

    //common sampler strategy

    if (InModelParams.Advanced.bUseCommonSampler)
    {
        common_params_sampling SamplingParams;
        
        if (InModelParams.Advanced.MinP != -1.f)
        {
            SamplingParams.min_p = InModelParams.Advanced.MinP;
        }
        if (InModelParams.Advanced.TopK != -1.f)
        {
            SamplingParams.top_k = InModelParams.Advanced.TopK;
        }
        if (InModelParams.Advanced.TopP != -1.f)
        {
            SamplingParams.top_p = InModelParams.Advanced.TopP;
        }
        if (InModelParams.Advanced.TypicalP != -1.f)
        {
            SamplingParams.typ_p = InModelParams.Advanced.TypicalP;
        }
        if (InModelParams.Advanced.Mirostat != -1)
        {
            SamplingParams.mirostat = InModelParams.Advanced.Mirostat;
            SamplingParams.mirostat_eta = InModelParams.Advanced.MirostatEta;
            SamplingParams.mirostat_tau = InModelParams.Advanced.MirostatTau;
        }

        //Seed is either default or the one specifically passed in for deterministic results
        if (InModelParams.Seed != -1)
        {
            SamplingParams.seed = InModelParams.Seed;
        }

        CommonSampler = common_sampler_init(LlamaModel, SamplingParams);
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
    if (InModelParams.Advanced.Mirostat != -1)
    {
        llama_sampler_chain_add(Sampler, llama_sampler_init_mirostat_v2(
            InModelParams.Advanced.Mirostat, InModelParams.Advanced.MirostatTau, InModelParams.Advanced.MirostatEta));
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
    

    ContextHistory.reserve(llama_n_ctx(Context));

    //empty by default
    Template = std::string();
    TemplateSource = FLlamaString::ToStd(InModelParams.CustomChatTemplate.TemplateSource);

    //Prioritize: custom jinja, then name, then default
    if (!InModelParams.CustomChatTemplate.Jinja.IsEmpty())
    {
        Template = FLlamaString::ToStd(InModelParams.CustomChatTemplate.Jinja);
        if (InModelParams.CustomChatTemplate.TemplateSource.IsEmpty())
        {
            TemplateSource = std::string("Custom Jinja");
        }
    }
    else if (   !InModelParams.CustomChatTemplate.TemplateSource.IsEmpty() &&
                InModelParams.CustomChatTemplate.TemplateSource != TEXT("tokenizer.chat_template"))
    {
        //apply template source name, this may fail
        std::string TemplateName = FLlamaString::ToStd(InModelParams.CustomChatTemplate.TemplateSource);
        const char* TemplatePtr = llama_model_chat_template(LlamaModel, TemplateName.c_str());

        if (TemplatePtr != nullptr)
        {
            Template = std::string(TemplatePtr);
        }
    }

    if(Template.empty())
    {
        const char* TemplatePtr = llama_model_chat_template(LlamaModel, nullptr);

        if (TemplatePtr != nullptr)
        {
            Template = std::string(TemplatePtr);
            TemplateSource = std::string("tokenizer.chat_template");
        }
    }
    
    FilledContextCharLength = 0;

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
    if (CommonSampler)
    {
        common_sampler_free(CommonSampler);
    }
    
    ContextHistory.clear();

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

void FLlamaInternal::ResetContextHistory()
{
    if (IsGenerating())
    {
        StopGeneration();
    }

    ContextHistory.clear();
    Messages.clear();

    llama_kv_cache_clear(Context);
    FilledContextCharLength = 0;
}

void FLlamaInternal::RollbackHistory(int32 ByNTokens)
{
    // clear the last n_regen tokens from the KV cache and update n_past
    llama_kv_cache_seq_rm(Context, 0, FilledContextCharLength - ByNTokens, -1);

    FilledContextCharLength -= ByNTokens;

    //Run a decode to sync everything else
    //llama_decode(Context, llama_batch_get_one(nullptr, 0));
}

int32 FLlamaInternal::InsertRawPrompt(const std::string& Prompt)
{
    if (!bIsModelLoaded)
    {
        UE_LOG(LlamaLog, Warning, TEXT("Model isn't loaded"));
        return 0;
    }

    return ProcessPrompt(Prompt);
}

int32 FLlamaInternal::InsertTemplatedPrompt(const std::string& Prompt, const std::string Role, bool bAddAssistantBoS)
{
    if (!bIsModelLoaded)
    {
        UE_LOG(LlamaLog, Warning, TEXT("Model isn't loaded"));
        return 0;
    }

    int NewLen = FilledContextCharLength;

    if (!Prompt.empty())
    {
        Messages.push_back({ Role.c_str(), _strdup(Prompt.c_str()) });
        NewLen = llama_chat_apply_template(Template.c_str(), Messages.data(), Messages.size(),
            bAddAssistantBoS, ContextHistory.data(), ContextHistory.size());

        //Resize if contexthistory can't hold it
        if (NewLen > ContextHistory.size())
        {
            ContextHistory.resize(NewLen);
            NewLen = llama_chat_apply_template(Template.c_str(), Messages.data(), Messages.size(),
                bAddAssistantBoS, ContextHistory.data(), ContextHistory.size());
        }
        if (NewLen < 0)
        {
            UE_LOG(LlamaLog, Warning, TEXT("failed to apply the chat template pre generation."));
            return 0;
        }
    }

    std::string FormattedPrompt(ContextHistory.data() + FilledContextCharLength, ContextHistory.data() + NewLen);

    int32 TokensProcessed = ProcessPrompt(FormattedPrompt);

    FilledContextCharLength = NewLen;

    return TokensProcessed;
}

std::string FLlamaInternal::InsertTemplatedPromptAndGenerate(const std::string& UserPrompt, const std::string Role, bool bAddAssistantBoS)
{
    if (!bIsModelLoaded)
    {
        UE_LOG(LlamaLog, Warning, TEXT("Model isn't loaded"));
        return "";
    }

    if (!UserPrompt.empty())
    {
        //Process initial prompt
        int32 TokensProcessed = InsertTemplatedPrompt(UserPrompt, Role, bAddAssistantBoS);
    }

    //Run generation
    std::string Response = Generate("");

    //Add the response to our templated messages
    Messages.push_back({ "assistant", _strdup(Response.c_str()) });

    //Sync ContextHistory
    FilledContextCharLength = llama_chat_apply_template(Template.c_str(), Messages.data(), Messages.size(), false, ContextHistory.data(), ContextHistory.size());
    if (FilledContextCharLength < 0)
    {
        UE_LOG(LlamaLog, Warning, TEXT("failed to apply the chat template post generation."));
        return "";
    }

    return Response;
}

std::string FLlamaInternal::ResumeGeneration()
{
    //run an empty user prompt
    return InsertTemplatedPromptAndGenerate(std::string());
}

int32 FLlamaInternal::ProcessPrompt(const std::string& Prompt)
{
    //Grab vocab
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

    // run it through the decode (input)
    if (llama_decode(Context, Batch))
    {
        GGML_ABORT("failed to decode\n");
    }

    return NPromptTokens;
}

std::string FLlamaInternal::Generate(const std::string& Prompt)
{
    const auto StartTime = ggml_time_us();
 
    bGenerationActive = true;
    
    if (!Prompt.empty())
    {
        int32 TokensProcessed = ProcessPrompt(Prompt);
    }

    std::string Response;

    const llama_vocab* Vocab = llama_model_get_vocab(LlamaModel);

    llama_batch Batch;
    
    llama_token NewTokenId;
    int32 NDecoded = 0;

    // check if we have enough space in the context to evaluate this batch - might need to be inside loop
    int NContext = llama_n_ctx(Context);
    int NContextUsed = llama_get_kv_cache_used_cells(Context);
    
    while (bGenerationActive) //processing can be aborted by flipping the boolean
    {
        //Common sampler is a bit faster
        if (CommonSampler)
        {
            NewTokenId = common_sampler_sample(CommonSampler, Context, -1); //sample using common sampler
            common_sampler_accept(CommonSampler, NewTokenId, true);
        }
        else
        {
            NewTokenId = llama_sampler_sample(Sampler, Context, -1);
        }

        // is it an end of generation?
        if (llama_vocab_is_eog(Vocab, NewTokenId))
        {
            break;
        }

        // convert the token to a string, print it and add it to the response
        std::string Piece = common_token_to_piece(Vocab, NewTokenId, true);
        
        Response += Piece;
        NDecoded += 1;

        if (NContextUsed + NDecoded > NContext)
        {
            UE_LOG(LlamaLog, Error, TEXT("context size %d exceeded\n"), NContext);
            bGenerationActive = false;
            return "";
        }

        if (OnTokenGenerated)
        {
            OnTokenGenerated(Piece);
        }

        // prepare the next batch with the sampled token
        Batch = llama_batch_get_one(&NewTokenId, 1);

        if (llama_decode(Context, Batch))
        {
            GGML_ABORT("failed to decode\n");
        }
    }

    bGenerationActive = false;

    const auto EndTime = ggml_time_us();
    const auto Duration = (EndTime - StartTime) / 1000000.0f;

    if (OnGenerationStats)
    {
        OnGenerationStats(Duration, NDecoded, NDecoded / Duration);
    }

    return Response;
}

FLlamaInternal::~FLlamaInternal()
{
    OnTokenGenerated = nullptr;
    UnloadModel();
    llama_backend_free();
}
