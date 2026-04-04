// Copyright 2025-current Getnamo.

#include "Internal/LlamaInternal.h"
#include "common/common.h"
#include "common/sampling.h"
#include "mtmd/mtmd.h"
#include "mtmd/mtmd-helper.h"
#include "LlamaDataTypes.h"
#include "LlamaUtility.h"
#include "HardwareInfo.h"

bool FLlamaInternal::LoadModelFromParams(const FLLMModelParams& InModelParams)
{
    FString RHI = FHardwareInfo::GetHardwareDetailsString();
    FString GPU = FPlatformMisc::GetPrimaryGPUBrand();

    UE_LOG(LogTemp, Log, TEXT("Device Found: %s %s"), *GPU, *RHI);

    LastLoadedParams = InModelParams;

    // only print errors
    llama_log_set([](enum ggml_log_level level, const char* text, void* /* user_data */)
    {
        if (level >= GGML_LOG_LEVEL_ERROR) {
            // Route to UE log so it appears in editor Output Log, not just stderr
            UE_LOG(LlamaLog, Warning, TEXT("[llama] %hs"), text);
        }
    }, nullptr);

    // load dynamic backends
    ggml_backend_load_all();

    std::string ModelPath = TCHAR_TO_UTF8(*FLlamaPaths::ParsePathIntoFullPath(InModelParams.PathToModel));


    //Regular init
    llama_model_params LlamaModelParams = llama_model_default_params();
    LlamaModelParams.n_gpu_layers = InModelParams.GPULayers;

    LlamaModel = llama_model_load_from_file(ModelPath.c_str(), LlamaModelParams);
    if (!LlamaModel)
    {
        FString ErrorMessage = FString::Printf(TEXT("Unable to load model at <%hs>"), ModelPath.c_str());
        EmitErrorMessage(ErrorMessage, 10, __func__);
        return false;
    }

    llama_context_params ContextParams = llama_context_default_params();
    ContextParams.n_ctx = InModelParams.MaxContextLength;
    ContextParams.n_batch = InModelParams.MaxBatchLength;
    ContextParams.n_threads = InModelParams.Threads;
    ContextParams.n_threads_batch = InModelParams.Threads;

    if (InModelParams.Advanced.bEmbeddingMode)
    {
        ContextParams.embeddings = InModelParams.Advanced.bEmbeddingMode;
    }

    // Let the model decide flash attention — AUTO picks the best supported mode.
    // Required for vision models (Qwen2.5-Omni etc.) where the mmproj encoder needs it.
    if (!InModelParams.MmprojPath.IsEmpty())
    {
        ContextParams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
    }

    SavedFlashAttnType = ContextParams.flash_attn_type;
    Context = llama_init_from_model(LlamaModel, ContextParams);
    
    if (!Context)
    {
        FString ErrorMessage = FString::Printf(TEXT("Unable to initialize model with given context params."));
        EmitErrorMessage(ErrorMessage, 11, __func__);
        return false;
    }

    //Only standard mode uses sampling
    if (!InModelParams.Advanced.bEmbeddingMode)
    {
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

    }//End non-embedding mode

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

    if (InModelParams.Advanced.bEmbeddingMode)
    {
        Template = std::string("");
        TemplateSource = std::string("embedding mode, templates not used");
    }
    else
    {

        if (Template.empty())
        {
            const char* TemplatePtr = llama_model_chat_template(LlamaModel, nullptr);

            if (TemplatePtr != nullptr)
            {
                Template = std::string(TemplatePtr);
                TemplateSource = std::string("tokenizer.chat_template");
            }
        }
    }
    
    FilledContextCharLength = 0;

    bIsModelLoaded = true;

    //Initialize multimodal if mmproj path is provided
    if (!InModelParams.MmprojPath.IsEmpty())
    {
        InitMultimodal(InModelParams.MmprojPath);
    }

    return true;
}

void FLlamaInternal::UnloadModel()
{
    //Free mtmd before context/model since it holds references to them
    FreeMultimodal();

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
        return llama_memory_seq_pos_max(llama_get_memory(Context), 0);
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
        else
        {
            //Only message is the system's prompt, nothing to do
            return;
        }
    }

    //Full Reset
    ContextHistory.clear();
    Messages.clear();

    llama_memory_clear(llama_get_memory(Context), false);
    FilledContextCharLength = 0;
}

void FLlamaInternal::RollbackContextHistoryByTokens(int32 NTokensToErase)
{
    // clear the last n_regen tokens from the KV cache and update n_past
    int32 TokensUsed = llama_memory_seq_pos_max(llama_get_memory(Context), 0); //FilledContextCharLength

    llama_memory_seq_rm(llama_get_memory(Context), 0, TokensUsed - NTokensToErase, -1);

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

    //Check for invalid lengths
    if (NewLen < 0)
    {
        UE_LOG(LlamaLog, Warning, TEXT("Inserted prompt after templating has an invalid length of %d, skipping generation. Check your jinja template or model gguf. NB: some templates merge system prompts with user prompts (e.g. gemma) and it's considered normal behavior."), NewLen);
        return std::string();
    }

    //Only process non-zero prompts
    if (NewLen > 0)
    {
        std::string FormattedPrompt(ContextHistory.data() + FilledContextCharLength, ContextHistory.data() + NewLen);
        int32 TokensProcessed = ProcessPrompt(FormattedPrompt, Role);
    }

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

void FLlamaInternal::GetPromptEmbeddings(const std::string& Text, std::vector<float>& Embeddings)
{
    //apply https://github.com/ggml-org/llama.cpp/blob/master/examples/embedding/embedding.cpp wrapping logic

    if (!Context)
    {
        EmitErrorMessage(TEXT("Context invalid, did you load the model?"), 43, __func__);
        return;
    }

    //Tokenize prompt - we're crashing out here... 
    //Check if our sampling/etc params are wrong or vocab is wrong.
    //Try tokenizing using normal method?
    //CONTINUE HERE:

    UE_LOG(LogTemp, Log, TEXT("Trying to sample <%hs>"), Text.c_str());

    auto Input = common_tokenize(Context, Text, true, true);

    //int32 NBatch = llama_n_ctx(Context);    //todo: get this from our params
    int32 NBatch = Input.size();    //todo: get this from our params

    llama_batch Batch = llama_batch_init(NBatch, 0, 1);
    //llama_batch Batch = llama_batch_get_one(Input.data(), Input.size());

    //add single batch
    BatchAddSeq(Batch, Input, 0);

    enum llama_pooling_type PoolingType = llama_pooling_type(Context);

    //Count number of embeddings
    int32 EmbeddingCount = 0;
    
    if (PoolingType == llama_pooling_type::LLAMA_POOLING_TYPE_NONE)
    {
        EmbeddingCount = Input.size();
    }
    else
    {
        EmbeddingCount = 1;
    }
    
    int32 NEmbd = llama_model_n_embd(LlamaModel);

    //allocate
    Embeddings = std::vector<float>(EmbeddingCount * NEmbd, 0);

    float* EmbeddingsPtr = Embeddings.data();

    //decode
    BatchDecodeEmbedding(Context, Batch, EmbeddingsPtr, 0, NEmbd, 2);

    UE_LOG(LogTemp, Log, TEXT("Embeddings count: %d"), Embeddings.size());
}

int32 FLlamaInternal::ProcessPrompt(const std::string& Prompt, EChatTemplateRole Role)
{
    const auto StartTime = ggml_time_us();

    //Grab vocab
    const llama_vocab* Vocab = llama_model_get_vocab(LlamaModel);
    const bool IsFirst = llama_memory_seq_pos_max(llama_get_memory(Context), 0) == 0;

    // tokenize the prompt
    const int NPromptTokens = -llama_tokenize(Vocab, Prompt.c_str(), Prompt.size(), NULL, 0, IsFirst, true);
    std::vector<llama_token> PromptTokens(NPromptTokens);
    if (llama_tokenize(Vocab, Prompt.c_str(), Prompt.size(), PromptTokens.data(), PromptTokens.size(), IsFirst, true) < 0)
    {
        EmitErrorMessage(TEXT("failed to tokenize the prompt"), 21, __func__);
        return NPromptTokens;
    }

    //All in one batch
    if (LastLoadedParams.Advanced.PromptProcessingPacingSleep == 0.f)
    {
        // prepare a batch for the prompt
        llama_batch Batch = llama_batch_get_one(PromptTokens.data(), PromptTokens.size());

        //check sizing before running prompt decode
        int NContext = llama_n_ctx(Context);
        int NContextUsed = llama_memory_seq_pos_max(llama_get_memory(Context), 0);

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
    }
    //Split it and sleep between batches for pacing purposes
    else
    {
        int32 BatchCount = LastLoadedParams.Advanced.PromptProcessingPacingSplitN;

        int32 TotalTokens = PromptTokens.size();
        int32 TokensPerBatch = TotalTokens / BatchCount;
        int32 Remainder = TotalTokens % BatchCount;

        int32 StartIndex = 0;

        for (int32 i = 0; i < BatchCount; i++)
        {
            // Calculate how many tokens to put in this batch
            int32 CurrentBatchSize = TokensPerBatch + (i < Remainder ? 1 : 0);

            // Slice the relevant tokens for this batch
            std::vector<llama_token> BatchTokens(
                PromptTokens.begin() + StartIndex,
                PromptTokens.begin() + StartIndex + CurrentBatchSize
            );

            // Prepare the batch
            llama_batch Batch = llama_batch_get_one(BatchTokens.data(), BatchTokens.size());

            // Check context before running decode
            int NContext = llama_n_ctx(Context);
            int NContextUsed = llama_memory_seq_pos_max(llama_get_memory(Context), 0);

            if (NContextUsed + BatchTokens.size() > NContext)
            {
                EmitErrorMessage(FString::Printf(
                    TEXT("Failed to insert, tried to insert %d tokens to currently used %d tokens which is more than the max %d context size. Try increasing the context size and re-run prompt."),
                    BatchTokens.size(), NContextUsed, NContext
                ), 22, __func__);
                return 0;
            }

            // Decode this batch
            if (llama_decode(Context, Batch))
            {
                EmitErrorMessage(TEXT("Failed to decode, could not find a KV slot for the batch (try reducing the size of the batch or increase the context)."), 23, __func__);
                return BatchTokens.size();
            }

            StartIndex += CurrentBatchSize;
            FPlatformProcess::Sleep(LastLoadedParams.Advanced.PromptProcessingPacingSleep);
        }
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

    llama_token NewTokenId;
    int32 NDecoded = 0;

    // check if we have enough space in the context to evaluate this batch - might need to be inside loop
    int NContext = llama_n_ctx(Context);
    bool bEOGExit = false;

    // For M-RoPE models (e.g. Qwen2VL), seq_pos_max reflects the max 2D spatial position of
    // image tokens and is NOT the correct next text position. Use NextGenerationNPast when it has
    // been set by ProcessMultimodalPrompt; otherwise fall back to seq_pos_max+1 (text-only path).
    llama_pos NPast = (NextGenerationNPast > 0)
        ? NextGenerationNPast
        : llama_memory_seq_pos_max(llama_get_memory(Context), 0) + 1;
    NextGenerationNPast = 0; // consumed

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

        if (NPast + NDecoded > NContext)
        {
            FString ErrorMessage = FString::Printf(TEXT("Context size %d exceeded on generation. Try increasing the context size and re-run prompt"), NContext);

            EmitErrorMessage(ErrorMessage, 31, __func__);
            return Response;
        }

        if (OnTokenGenerated)
        {
            OnTokenGenerated(Piece);
        }

        // Use explicit n_past position (mirrors mtmd-cli reference implementation).
        // This is critical for M-RoPE models where seq_pos_max != true next text position.
        llama_batch SingleBatch = llama_batch_get_one(&NewTokenId, 1);
        SingleBatch.pos = &NPast;  // override auto-position with tracked n_past

        if (llama_decode(Context, SingleBatch))
        {
            bGenerationActive = false;
            FString ErrorMessage = TEXT("Failed to decode. Could not find a KV slot for the batch (try reducing the size of the batch or increase the context)");
            EmitErrorMessage(ErrorMessage, 32, __func__);
            //Return partial response
            return Response;
        }

        NPast++;

        //sleep pacing
        if (LastLoadedParams.Advanced.TokenGenerationPacingSleep > 0.f)
        {
            FPlatformProcess::Sleep(LastLoadedParams.Advanced.TokenGenerationPacingSleep);
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
    //Handle empty template case
    char* templatePtr = (char*)InTemplate.c_str();
    if (InTemplate.length() == 0)
    {
        templatePtr = nullptr;
    }

    int32 NewLen = llama_chat_apply_template(templatePtr, FromMessages.data(), FromMessages.size(),
            bAddAssistantBoS, ToBuffer.data(), ToBuffer.size());

    //Resize if ToBuffer can't hold it
    if (NewLen > ToBuffer.size())
    {
        ToBuffer.resize(NewLen);
        NewLen = llama_chat_apply_template(InTemplate.c_str(), FromMessages.data(), FromMessages.size(),
            bAddAssistantBoS, ToBuffer.data(), ToBuffer.size());
    }
    else 
    {
        if (NewLen < 0)
        {
            EmitErrorMessage(TEXT("Failed to apply the chat template ApplyTemplateFromMessagesToBuffer, negative length"), 101, __func__);
        }
        else if (NewLen == 0)
        {
            //This isn't an error but needs to be handled by downstream
            
            //EmitErrorMessage(TEXT("Failed to apply the chat template ApplyTemplateFromMessagesToBuffer, length is 0."), 102, __func__);
        }
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

//from https://github.com/ggml-org/llama.cpp/blob/master/examples/embedding/embedding.cpp
void FLlamaInternal::BatchDecodeEmbedding(llama_context* InContext, llama_batch& Batch, float* Output, int NSeq, int NEmbd, int EmbdNorm)
{
    const enum llama_pooling_type pooling_type = llama_pooling_type(InContext);
    const struct llama_model* model = llama_get_model(InContext);

    // clear previous kv_cache values (irrelevant for embeddings)
    llama_memory_clear(llama_get_memory(InContext), false);

    // run model
    
    //Debug info
    //UE_LOG(LlamaLog, Log, TEXT("%hs: n_tokens = %d, n_seq = %d"), __func__, Batch.n_tokens, NSeq);

    if (llama_model_has_encoder(model) && !llama_model_has_decoder(model))
    {
        // encoder-only model
        if (llama_encode(InContext, Batch) < 0) 
        {
            UE_LOG(LlamaLog, Error, TEXT("%hs : failed to encode"), __func__);
        }
    }
    else if (!llama_model_has_encoder(model) && llama_model_has_decoder(model)) 
    {
        // decoder-only model
        if (llama_decode(InContext, Batch) < 0) 
        {
            UE_LOG(LlamaLog, Log, TEXT("%hs : failed to decode"), __func__);
        }
    }

    for (int i = 0; i < Batch.n_tokens; i++) 
    {
        if (Batch.logits && !Batch.logits[i]) 
        {
            continue;
        }

        const float* Embd = nullptr;
        int EmbdPos = 0;
        
        if (pooling_type == LLAMA_POOLING_TYPE_NONE) 
        {
            // try to get token embeddings
            Embd = llama_get_embeddings_ith(InContext, i);
            EmbdPos = i;
            GGML_ASSERT(Embd != NULL && "failed to get token embeddings");
        }
        else if (Batch.seq_id)
        {
            // try to get sequence embeddings - supported only when pooling_type is not NONE
            Embd = llama_get_embeddings_seq(InContext, Batch.seq_id[i][0]);
            EmbdPos = Batch.seq_id[i][0];
            GGML_ASSERT(Embd != NULL && "failed to get sequence embeddings");
        }
        else
        {
            //NB: this generally won't work, we should crash here.
            Embd = llama_get_embeddings(InContext);
        }

        float* Out = Output + EmbdPos * NEmbd;
        common_embd_normalize(Embd, Out, NEmbd, EmbdNorm);
    }
}

void FLlamaInternal::BatchAddSeq(llama_batch& batch, const std::vector<int32_t>& tokens, llama_seq_id seq_id)
{
    size_t n_tokens = tokens.size();
    for (size_t i = 0; i < n_tokens; i++) 
    {
        common_batch_add(batch, tokens[i], i, { seq_id }, true);
    }
}

bool FLlamaInternal::InitMultimodal(const FString& MmprojPath)
{
    if (MmprojPath.IsEmpty())
    {
        return false;
    }
    if (!LlamaModel)
    {
        EmitErrorMessage(TEXT("Cannot init multimodal: model not loaded"), 50, __func__);
        return false;
    }

    std::string Path = TCHAR_TO_UTF8(*FLlamaPaths::ParsePathIntoFullPath(MmprojPath));

    mtmd_context_params Params = mtmd_context_params_default();
    Params.use_gpu = true;
    Params.n_threads = LastLoadedParams.Threads;
    Params.flash_attn_type = SavedFlashAttnType;

    MtmdContext = mtmd_init_from_file(Path.c_str(), LlamaModel, Params);
    if (!MtmdContext)
    {
        EmitErrorMessage(FString::Printf(TEXT("Failed to load multimodal projector from <%hs>"), Path.c_str()), 50, __func__);
        bMtmdLoaded = false;
        return false;
    }

    // Route mtmd-helper logs (image/audio batch errors) through UE log
    mtmd_helper_log_set([](enum ggml_log_level level, const char* text, void* /*user_data*/)
    {
        if (level >= GGML_LOG_LEVEL_ERROR) {
            UE_LOG(LlamaLog, Warning, TEXT("[mtmd] %hs"), text);
        }
    }, nullptr);

    bMtmdLoaded = true;
    UE_LOG(LlamaLog, Log, TEXT("Multimodal projector loaded. Vision: %s, Audio: %s"),
        mtmd_support_vision(MtmdContext) ? TEXT("yes") : TEXT("no"),
        mtmd_support_audio(MtmdContext) ? TEXT("yes") : TEXT("no"));
    return true;
}

void FLlamaInternal::FreeMultimodal()
{
    if (MtmdContext)
    {
        mtmd_free(MtmdContext);
        MtmdContext = nullptr;
    }
    bMtmdLoaded = false;
}

bool FLlamaInternal::IsMultimodalLoaded()
{
    return bMtmdLoaded;
}

bool FLlamaInternal::SupportsVision()
{
    return MtmdContext && mtmd_support_vision(MtmdContext);
}

bool FLlamaInternal::SupportsAudio()
{
    return MtmdContext && mtmd_support_audio(MtmdContext);
}

int32 FLlamaInternal::GetAudioSampleRate()
{
    if (MtmdContext)
    {
        return mtmd_get_audio_sample_rate(MtmdContext);
    }
    return 0;
}

int32 FLlamaInternal::ProcessMultimodalPrompt(const std::string& FormattedPrompt, const TArray<FLlamaMediaEntry>& MediaEntries, EChatTemplateRole Role, bool bLogitsLast)
{
    const auto StartTime = ggml_time_us();

    // 1. Build bitmaps from media entries
    TArray<mtmd_bitmap*> Bitmaps;
    TArray<const mtmd_bitmap*> BitmapPtrs;

    for (const FLlamaMediaEntry& Entry : MediaEntries)
    {
        mtmd_bitmap* Bmp = nullptr;

        if (!Entry.FilePath.IsEmpty())
        {
            std::string FilePath = TCHAR_TO_UTF8(*FLlamaPaths::ParsePathIntoFullPath(Entry.FilePath));
            Bmp = mtmd_helper_bitmap_init_from_file(MtmdContext, FilePath.c_str());
        }
        else if (Entry.MediaType == ELlamaMediaType::Image)
        {
            if (Entry.ImageRGBData.Num() > 0 && Entry.ImageWidth > 0 && Entry.ImageHeight > 0)
            {
                Bmp = mtmd_bitmap_init(Entry.ImageWidth, Entry.ImageHeight, Entry.ImageRGBData.GetData());
            }
        }
        else // Audio
        {
            if (Entry.AudioPCMData.Num() > 0)
            {
                Bmp = mtmd_bitmap_init_from_audio(Entry.AudioPCMData.Num(), Entry.AudioPCMData.GetData());
            }
        }

        if (!Bmp)
        {
            // Free any previously created bitmaps
            for (mtmd_bitmap* B : Bitmaps)
            {
                mtmd_bitmap_free(B);
            }
            EmitErrorMessage(TEXT("Failed to create mtmd bitmap from media entry"), 52, __func__);
            return 0;
        }

        Bitmaps.Add(Bmp);
        BitmapPtrs.Add(Bmp);
    }

    // 2. Tokenize with mtmd
    mtmd_input_chunks* Chunks = mtmd_input_chunks_init();
    // seq_pos_max returns -1 when the KV cache is empty; only add BOS on the very first prompt
    const llama_pos SeqPosMax = llama_memory_seq_pos_max(llama_get_memory(Context), 0);
    const bool IsFirst = (SeqPosMax < 0);

    mtmd_input_text InputText;
    InputText.text = FormattedPrompt.c_str();
    InputText.add_special = IsFirst;
    InputText.parse_special = true;

    int32_t TokenizeResult = mtmd_tokenize(MtmdContext, Chunks, &InputText, BitmapPtrs.GetData(), BitmapPtrs.Num());

    if (TokenizeResult != 0)
    {
        FString Msg = (TokenizeResult == 1)
            ? TEXT("Number of <__media__> markers does not match number of media entries")
            : TEXT("Image/audio preprocessing error during mtmd_tokenize");
        EmitErrorMessage(Msg, (TokenizeResult == 1) ? 51 : 53, __func__);

        mtmd_input_chunks_free(Chunks);
        for (mtmd_bitmap* B : Bitmaps) { mtmd_bitmap_free(B); }
        return 0;
    }

    // 3. Eval all chunks into KV cache
    // seq_pos_max is the last *occupied* position; next token must go at seq_pos_max + 1.
    // For an empty cache (seq_pos_max == -1), -1 + 1 = 0 which is correct.
    llama_pos NPast = SeqPosMax + 1;
    llama_pos NewNPast = NPast;
    const int32 NCtx = llama_n_ctx(Context);
    const size_t NChunks = mtmd_input_chunks_size(Chunks);
    const size_t NTokensInChunks = mtmd_helper_get_n_tokens(Chunks);

    const int32 ModelRopeType = (int32)llama_model_rope_type(LlamaModel);
    UE_LOG(LlamaLog, Log, TEXT("[ProcessMultimodalPrompt] n_past=%d n_ctx=%d chunks=%zu tokens_in_chunks=%zu n_batch=%d uses_mrope=%d model_rope_type=%d"),
        (int32)NPast, NCtx, NChunks, NTokensInChunks,
        LastLoadedParams.MaxBatchLength,
        (int32)mtmd_decode_use_mrope(MtmdContext),
        ModelRopeType);

    int32_t EvalResult = mtmd_helper_eval_chunks(
        MtmdContext, Context, Chunks,
        NPast, 0,
        LastLoadedParams.MaxBatchLength,
        bLogitsLast, &NewNPast);

    int32 TokensProcessed = (int32)NTokensInChunks;

    // Cleanup
    mtmd_input_chunks_free(Chunks);
    for (mtmd_bitmap* B : Bitmaps) { mtmd_bitmap_free(B); }

    if (EvalResult != 0)
    {
        EmitErrorMessage(FString::Printf(TEXT("mtmd_helper_eval_chunks failed (ret=%d, n_past=%d, n_ctx=%d, chunks=%zu, tokens=%zu)"),
            EvalResult, (int32)NPast, NCtx, NChunks, NTokensInChunks), 54, __func__);
        return 0;
    }

    // Record the correct next KV position from mtmd (NOT seq_pos_max, which is wrong for M-RoPE
    // because 2D spatial positions from image tokens inflate seq_pos_max beyond the true text position).
    NextGenerationNPast = NewNPast;

    const auto StopTime = ggml_time_us();
    const float Duration = (StopTime - StartTime) / 1000000.0f;

    if (OnPromptProcessed)
    {
        float Speed = (Duration > 0.f) ? TokensProcessed / Duration : 0.f;
        OnPromptProcessed(TokensProcessed, Role, Speed);
    }

    return TokensProcessed;
}

std::string FLlamaInternal::InsertMultimodalPrompt(const std::string& TextWithMarkers, const TArray<FLlamaMediaEntry>& MediaEntries, EChatTemplateRole Role, bool bAddAssistantBoS, bool bGenerateReply)
{
    if (!bIsModelLoaded)
    {
        UE_LOG(LlamaLog, Warning, TEXT("Model isn't loaded"));
        return std::string();
    }

    if (!bMtmdLoaded)
    {
        EmitErrorMessage(TEXT("Multimodal projector not loaded. Set MmprojPath in ModelParams before calling LoadModel."), 50, __func__);
        return std::string();
    }

    int32 NewLen = FilledContextCharLength;

    if (!TextWithMarkers.empty())
    {
        Messages.push_back({ RoleForEnum(Role), _strdup(TextWithMarkers.c_str()) });
        NewLen = ApplyTemplateToContextHistory(bAddAssistantBoS);
    }

    if (NewLen < 0)
    {
        UE_LOG(LlamaLog, Warning, TEXT("Multimodal prompt after templating has an invalid length of %d"), NewLen);
        return std::string();
    }

    if (NewLen > 0)
    {
        std::string FormattedPrompt(ContextHistory.data() + FilledContextCharLength, ContextHistory.data() + NewLen);
        int32 TokensProcessed = ProcessMultimodalPrompt(FormattedPrompt, MediaEntries, Role, bGenerateReply);
    }

    FilledContextCharLength = NewLen;

    std::string Response;
    if (bGenerateReply)
    {
        Response = Generate();
    }

    return Response;
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
