// Copyright 2025-current Getnamo.

#pragma once

#include <string>
#include <vector>
#include "LlamaDataTypes.h"
#include "llama.h"

/** 
* Uses mostly Llama.cpp native API, meant to be embedded in LlamaNative that wraps 
* unreal threading and data types.
*/
class FLlamaInternal
{
public:
    //Core State
    llama_model* LlamaModel = nullptr;
    llama_context* Context = nullptr;
    llama_sampler* Sampler = nullptr;
    struct common_sampler* CommonSampler = nullptr;

    //main streaming callback
    TFunction<void(const std::string& TokenPiece)>OnTokenGenerated = nullptr;
    TFunction<void(int32 TokensProcessed, EChatTemplateRole ForRole, float Speed)>OnPromptProcessed = nullptr;   //useful for waiting for system prompt ready
    TFunction<void(const std::string& Response, float Time, int32 Tokens, float Speed)>OnGenerationComplete = nullptr;

    //NB basic error codes: 1x == Load Error, 2x == Process Prompt error, 3x == Generate error. 1xx == Misc errors
    TFunction<void(const FString& ErrorMessage, int32 ErrorCode)> OnError = nullptr;     //doesn't use std::string due to expected consumer

    //Messaging state
    std::vector<llama_chat_message> Messages;
    std::vector<char> ContextHistory;

    //Loaded state
    std::string Template;
    std::string TemplateSource;

    //Cached params, should be accessed on BT
    FLLMModelParams LastLoadedParams;

    //Model loading
    bool LoadModelFromParams(const FLLMModelParams& InModelParams);
    void UnloadModel();
    bool IsModelLoaded();

    //Generation
    void ResetContextHistory(bool bKeepSystemsPrompt = false);
    void RollbackContextHistoryByTokens(int32 NTokensToErase);
    void RollbackContextHistoryByMessages(int32 NMessagesToErase);

    //raw prompt insert doesn't not update messages, just context history
    std::string InsertRawPrompt(const std::string& Prompt, bool bGenerateReply = true);

    //main function for structure insert and generation
    std::string InsertTemplatedPrompt(const std::string& Prompt, EChatTemplateRole Role = EChatTemplateRole::User, bool bAddAssistantBoS = true, bool bGenerateReply = true);

    //continue generating from last stop
    std::string ResumeGeneration();

    //Feature todo: delete the last message and try again
    //std::string RerollLastGeneration();

    std::string WrapPromptForRole(const std::string& Text, EChatTemplateRole Role, const std::string& OverrideTemplate, bool bAddAssistantBoS = false);


    //flips bGenerationActive which will stop generation on next token. Threadsafe call.
    void StopGeneration();
    bool IsGenerating();

    int32 MaxContext();
    int32 UsedContext();

    FLlamaInternal();
    ~FLlamaInternal();


    //for embedding models

    //take a prompt and return an array of floats signifying the embeddings
    void GetPromptEmbeddings(const std::string& Text, std::vector<float>& Embeddings);

protected:
    //Wrapper for user<->assistant templated conversation
    int32 ProcessPrompt(const std::string& Prompt, EChatTemplateRole Role = EChatTemplateRole::Unknown);
    std::string Generate(const std::string& Prompt = "", bool bAppendToMessageHistory = true);

    void EmitErrorMessage(const FString& ErrorMessage, int32 ErrorCode = -1, const FString& FunctionName = TEXT("unknown"));

    int32 ApplyTemplateToContextHistory(bool bAddAssistantBOS = false);
    int32 ApplyTemplateFromMessagesToBuffer(const std::string& Template, std::vector<llama_chat_message>& FromMessages, std::vector<char>& ToBuffer, bool bAddAssistantBoS = false);

    const char* RoleForEnum(EChatTemplateRole Role);

    FThreadSafeBool bIsModelLoaded = false;
    int32 FilledContextCharLength = 0;
    FThreadSafeBool bGenerationActive = false;

    //Embedding Decoding utilities
    void BatchDecodeEmbedding(llama_context* ctx, llama_batch& batch, float* output, int n_seq, int n_embd, int embd_norm);
    void BatchAddSeq(llama_batch& batch, const std::vector<int32_t>& tokens, llama_seq_id seq_id);
};