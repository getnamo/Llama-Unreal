// Copyright 2025-current Getnamo.

#include "Internal/LlamaInternal.h"
#include "common/common.h"
#include "common/sampling.h"
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
        FString ErrorMessage = FString::Printf(TEXT("Unable to load model at <%hs>"), Path.c_str());
        EmitErrorMessage(ErrorMessage, 10, __func__);
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
        FString ErrorMessage = FString::Printf(TEXT("Unable to initialize model with given context params."));
        EmitErrorMessage(ErrorMessage, 11, __func__);
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
    
    //NB: this is just a starting heuristic, 
    ContextHistory.reserve(1024);

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
        CommonSampler = nullptr;
    }
    
    ContextHistory.clear();

    bIsModelLoaded = false;
}

std::string FLlamaInternal::WrapPromptForRole(const std::string& Text, EChatTemplateRole Role, const std::string& OverrideTemplate, bool bAddAssistantBoS)
{
    std::vector<llama_chat_message> MessageListWrapper;
    MessageListWrapper.push_back({ RoleForEnum(Role), _strdup(Text.c_str()) });

    //pre-allocate buffer 2x the size of text
    std::vector<char> Buffer;

    int32 NewLen = 0;

    if (OverrideTemplate.empty())
    {
        NewLen = ApplyTemplateFromMessagesToBuffer(Template, MessageListWrapper, Buffer, bAddAssistantBoS);
    }
    else
    {
        NewLen = ApplyTemplateFromMessagesToBuffer(OverrideTemplate, MessageListWrapper, Buffer, bAddAssistantBoS);
    }

    if(NewLen > 0)
    {
        return std::string(Buffer.data(), Buffer.data() + NewLen);
    }
    else
    {
        return std::string("");
    }
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

void FLlamaInternal::ResetContextHistory(bool bKeepSystemsPrompt)
{
    if (!bIsModelLoaded)
    {
        return;
    }

    if (IsGenerating())
    {
        StopGeneration();
    }

    if (bKeepSystemsPrompt)
    {
        //Valid trim case
        if (Messages.size() > 1)
        {
            //Rollback all the messages except the first one
            RollbackContextHistoryByMessages(Messages.size() - 1);
            return;
        }
    }

    //Full Reset
    ContextHistory.clear();
    Messages.clear();

    llama_kv_cache_clear(Context);
    FilledContextCharLength = 0;
}

void FLlamaInternal::RollbackContextHistoryByTokens(int32 NTokensToErase)
{
    // clear the last n_regen tokens from the KV cache and update n_past
    int32 TokensUsed = llama_get_kv_cache_used_cells(Context); //FilledContextCharLength

    llama_kv_cache_seq_rm(Context, 0, TokensUsed - NTokensToErase, -1);

    //FilledContextCharLength -= NTokensToErase;

    //Run a decode to sync everything else
    //llama_decode(Context, llama_batch_get_one(nullptr, 0));
}

void FLlamaInternal::RollbackContextHistoryByMessages(int32 NMessagesToErase)
{
    //cannot do rollback if model isn't loaded, ignore.
    if (!bIsModelLoaded)
    {
        return;
    }

    if (IsGenerating())
    {
        StopGeneration();
    }

    if (NMessagesToErase <= Messages.size()) 
    {
        Messages.resize(Messages.size() - NMessagesToErase);
    }

    //Obtain full prompt before it gets deleted
    std::string FullPrompt(ContextHistory.data(), ContextHistory.data() + FilledContextCharLength);
    
    //resize the context history
    int32 NewLen = ApplyTemplateToContextHistory(false);

    //tokenize to find out how many tokens we need to remove

    //Obtain new prompt, find delta
    std::string FormattedPrompt(ContextHistory.data(), ContextHistory.data() + NewLen);

    std::string PromptToRemove(FullPrompt.substr(FormattedPrompt.length()));

    const llama_vocab* Vocab = llama_model_get_vocab(LlamaModel);
    const int NPromptTokens = -llama_tokenize(Vocab, PromptToRemove.c_str(), PromptToRemove.size(), NULL, 0, false, true);

    //now rollback KV-cache
    RollbackContextHistoryByTokens(NPromptTokens);

    //Sync resized length;
    FilledContextCharLength = NewLen;

    //Shrink to fit
    ContextHistory.resize(FilledContextCharLength);
}

std::string FLlamaInternal::InsertRawPrompt(const std::string& Prompt, bool bGenerateReply)
{
    if (!bIsModelLoaded)
    {
        UE_LOG(LlamaLog, Warning, TEXT("Model isn't loaded"));
        return 0;
    }

    int32 TokensProcessed = ProcessPrompt(Prompt);

    FLlamaString::AppendToCharVector(ContextHistory, Prompt);

    if (bGenerateReply)
    {
        std::string Response = Generate("", false);
        FLlamaString::AppendToCharVector(ContextHistory, Response);
    }
    return "";
}

std::string FLlamaInternal::InsertTemplatedPrompt(const std::string& Prompt, EChatTemplateRole Role, bool bAddAssistantBoS, bool bGenerateReply)
{
    if (!bIsModelLoaded)
    {
        UE_LOG(LlamaLog, Warning, TEXT("Model isn't loaded"));
        return std::string();
    }

    int32 NewLen = FilledContextCharLength;

    if (!Prompt.empty())
    {
        Messages.push_back({ RoleForEnum(Role), _strdup(Prompt.c_str()) });

        NewLen = ApplyTemplateToContextHistory(bAddAssistantBoS);
    }

    std::string FormattedPrompt(ContextHistory.data() + FilledContextCharLength, ContextHistory.data() + NewLen);

    int32 TokensProcessed = ProcessPrompt(FormattedPrompt, Role);

    FilledContextCharLength = NewLen;

    //Check for a reply if we want to generate one, otherwise return an empty reply
    std::string Response;
    if (bGenerateReply)
    {
        //Run generation
        Response = Generate();
    }

    return Response;
}

std::string FLlamaInternal::ResumeGeneration()
{
    //Todo: erase last assistant message to merge the two messages if the last message was the assistant one.

    //run an empty user prompt
    return Generate();
}

int32 FLlamaInternal::ProcessPrompt(const std::string& Prompt, EChatTemplateRole Role)
{
    const auto StartTime = ggml_time_us();

    //Grab vocab
    const llama_vocab* Vocab = llama_model_get_vocab(LlamaModel);
    const bool IsFirst = llama_get_kv_cache_used_cells(Context) == 0;

    // tokenize the prompt
    const int NPromptTokens = -llama_tokenize(Vocab, Prompt.c_str(), Prompt.size(), NULL, 0, IsFirst, true);
    std::vector<llama_token> PromptTokens(NPromptTokens);
    if (llama_tokenize(Vocab, Prompt.c_str(), Prompt.size(), PromptTokens.data(), PromptTokens.size(), IsFirst, true) < 0)
    {
        EmitErrorMessage(TEXT("failed to tokenize the prompt"), 21, __func__);
        return NPromptTokens;
    }

    // prepare a batch for the prompt
    llama_batch Batch = llama_batch_get_one(PromptTokens.data(), PromptTokens.size());

    //check sizing before running prompt decode
    int NContext = llama_n_ctx(Context);
    int NContextUsed = llama_get_kv_cache_used_cells(Context);

    if (NContextUsed + NPromptTokens > NContext)
    {
        EmitErrorMessage(FString::Printf(
            TEXT("Failed to insert, tried to insert %d tokens to currently used %d tokens which is more than the max %d context size. Try increasing the context size and re-run prompt."),
            NPromptTokens, NContextUsed, NContext
            ), 22, __func__);
        return 0;
    }

    // run it through the decode (input)
    if (llama_decode(Context, Batch))
    {
        EmitErrorMessage(TEXT("Failed to decode, could not find a KV slot for the batch (try reducing the size of the batch or increase the context)."), 23, __func__);
        return NPromptTokens;
    }

    const auto StopTime = ggml_time_us();
    const float Duration = (StopTime - StartTime) / 1000000.0f;

    if (OnPromptProcessed)
    {
        float Speed = NPromptTokens / Duration;
        OnPromptProcessed(NPromptTokens, Role, Speed);
    }

    return NPromptTokens;
}

std::string FLlamaInternal::Generate(const std::string& Prompt, bool bAppendToMessageHistory)
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
    bool bEOGExit = false;
    
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
            bEOGExit = true;
            break;
        }

        // convert the token to a string, print it and add it to the response
        std::string Piece = common_token_to_piece(Vocab, NewTokenId, true);
        
        Response += Piece;
        NDecoded += 1;

        if (NContextUsed + NDecoded > NContext)
        {
            FString ErrorMessage = FString::Printf(TEXT("Context size %d exceeded on generation. Try increasing the context size and re-run prompt"), NContext);

            EmitErrorMessage(ErrorMessage, 31, __func__);
            return Response;
        }

        if (OnTokenGenerated)
        {
            OnTokenGenerated(Piece);
        }

        // prepare the next batch with the sampled token
        Batch = llama_batch_get_one(&NewTokenId, 1);

        if (llama_decode(Context, Batch))
        {
            bGenerationActive = false;
            FString ErrorMessage = TEXT("Failed to decode. Could not find a KV slot for the batch (try reducing the size of the batch or increase the context)");
            EmitErrorMessage(ErrorMessage, 32, __func__);
            //Return partial response
            return Response;
        }
    }

    bGenerationActive = false;

    const auto StopTime = ggml_time_us();
    const float Duration = (StopTime - StartTime) / 1000000.0f;

    if (bAppendToMessageHistory)
    {
        //Add the response to our templated messages
        Messages.push_back({ RoleForEnum(EChatTemplateRole::Assistant), _strdup(Response.c_str()) });

        //Sync ContextHistory
        FilledContextCharLength = ApplyTemplateToContextHistory(false);
    }

    if (OnGenerationComplete)
    {
        OnGenerationComplete(Response, Duration, NDecoded, NDecoded / Duration);
    }

    return Response;
}

void FLlamaInternal::EmitErrorMessage(const FString& ErrorMessage, int32 ErrorCode, const FString& FunctionName)
{
    UE_LOG(LlamaLog, Error, TEXT("[%s error %d]: %s"), *FunctionName, ErrorCode,  *ErrorMessage);
    if (OnError)
    {
        OnError(ErrorMessage, ErrorCode);
    }
}

//NB: this function will apply out of range errors in log, this is normal behavior due to how templates are applied
int32 FLlamaInternal::ApplyTemplateToContextHistory(bool bAddAssistantBOS)
{
    return ApplyTemplateFromMessagesToBuffer(Template, Messages, ContextHistory, bAddAssistantBOS);
}

int32 FLlamaInternal::ApplyTemplateFromMessagesToBuffer(const std::string& InTemplate, std::vector<llama_chat_message>& FromMessages, std::vector<char>& ToBuffer, bool bAddAssistantBoS)
{
    int32 NewLen = llama_chat_apply_template(InTemplate.c_str(), FromMessages.data(), FromMessages.size(),
        bAddAssistantBoS, ToBuffer.data(), ToBuffer.size());

    //Resize if ToBuffer can't hold it
    if (NewLen > ToBuffer.size())
    {
        ToBuffer.resize(NewLen);
        NewLen = llama_chat_apply_template(InTemplate.c_str(), FromMessages.data(), FromMessages.size(),
            bAddAssistantBoS, ToBuffer.data(), ToBuffer.size());
    }
    if (NewLen < 0)
    {
        EmitErrorMessage(TEXT("Failed to apply the chat template ApplyTemplateFromMessagesToBuffer."), 101, __func__);
    }
    return NewLen;
}

const char* FLlamaInternal::RoleForEnum(EChatTemplateRole Role)
{
    if (Role == EChatTemplateRole::User)
    {
        return "user";
    }
    else if (Role == EChatTemplateRole::Assistant)
    {
        return "assistant";
    }
    else if (Role == EChatTemplateRole::System)
    {
        return "system";
    }
    else {
        return "unknown";
    }
}

FLlamaInternal::FLlamaInternal()
{

}

FLlamaInternal::~FLlamaInternal()
{
    OnTokenGenerated = nullptr;
    UnloadModel();
    llama_backend_free();
}
