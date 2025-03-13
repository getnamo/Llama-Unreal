#pragma once

#include <string>
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

    //Messaging state
    std::vector<llama_chat_message> Messages;
    std::vector<char> ContextHistory;

    //Loaded state
    std::string Template;
    std::string TemplateSource;

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

    //delete the last message and tries again
    //std::string RerollLastGeneration();

    //flips bGenerationActive which will stop generation on next token. Threadsafe call.
    void StopGeneration();
    bool IsGenerating();

    int32 MaxContext();
    int32 UsedContext();

    FLlamaInternal();
    ~FLlamaInternal();

protected:
    //Wrapper for user<->assistant templated conversation
    int32 ProcessPrompt(const std::string& Prompt, EChatTemplateRole Role = EChatTemplateRole::Unknown);
    std::string Generate(const std::string& Prompt = "", bool bAppendToMessageHistory = true);

    int32 ApplyTemplateToContextHistory(bool bAddAssistantBOS = false);

    const char* RoleForEnum(EChatTemplateRole Role);

    bool bIsModelLoaded = false;
    int32 FilledContextCharLength = 0;
    FThreadSafeBool bGenerationActive = false;
};