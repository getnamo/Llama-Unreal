#pragma once

#include <string>
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
    TArray<llama_chat_message> Messages;
    TArray<char> ContextHistory;

    //Loaded state
    std::string Template;
    std::string TemplateSource;

    //Model loading
    bool LoadModelFromParams(const FLLMModelParams& InModelParams);
    void UnloadModel();
    bool IsModelLoaded();


    //Generation
    std::string InsertRawPrompt(const std::string& Prompt);

    //main internal function - synchronous so should be called from bg thread. Will emit OnTokenGenerated for each token.
    std::string InsertTemplatedPrompt(const std::string& Prompt, const std::string& role = "user");


    //continue generating from last stop
    std::string ResumeGeneration();

    //delete the last message and tries again
    //std::string RerollLastGeneration();

    //flips bGenerationActive which will stop generation on next token. Threadsafe call.
    void StopGeneration();
    bool IsGenerating();

    int32 MaxContext();
    int32 UsedContext();

    ~FLlamaInternal();

protected:
    //Wrapper for user<->assistant templated conversation
    std::string Generate(const std::string& Prompt);

    bool bIsModelLoaded = false;
    int32 PrevLen = 0;
    FThreadSafeBool bGenerationActive = false;
};