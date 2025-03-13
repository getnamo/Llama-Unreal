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
    TFunction<void(float Time, int32 Tokens, float Speed)>OnGenerationStats = nullptr;

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
    void ResetContextHistory();
    void RollbackContextHistoryByTokens(int32 NTokensToErase);
    void RollbackContextHistoryByMessages(int32 NMessagesToErase);

    int32 InsertRawPrompt(const std::string& Prompt);
    int32 InsertTemplatedPrompt(const std::string& Prompt, EChatTemplateRole Role = EChatTemplateRole::User, bool bAddAssistantBoS = true);

    //main internal function - synchronous so should be called from bg thread. Will emit OnTokenGenerated for each token.
    std::string InsertTemplatedPromptAndGenerate(const std::string& Prompt, EChatTemplateRole Role = EChatTemplateRole::User, bool bAddAssistantBoS = true);


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
    int32 ProcessPrompt(const std::string& Prompt);
    std::string Generate(const std::string& Prompt);

    const char* RoleForEnum(EChatTemplateRole Role);

    bool bIsModelLoaded = false;
    int32 FilledContextCharLength = 0;
    FThreadSafeBool bGenerationActive = false;
};