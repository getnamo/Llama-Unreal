#pragma once

#include <string>
#include "llama.h"
#include "Internal/common.h"

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

    //main streaming callback
    TFunction<void(const std::string& TokenPiece)>OnTokenGenerated = nullptr;

    //Messaging state
    TArray<llama_chat_message> Messages;
    TArray<char> Formatted;

    //Loaded state
    char* Template;

    //Model loading
    bool LoadModelFromParams(const FLLMModelParams& InModelParams);
    void UnloadModel();
    bool IsModelLoaded();


    //Generation

    //main internal function - synchronous so should be called from bg thread. Will emit OnTokenGenerated for each token.
    std::string InsertPrompt(const std::string& Prompt);

    //flips bGenerationActive which will stop generation on next token. Threadsafe call.
    void StopGeneration();
    bool IsGenerating();

    ~FLlamaInternal();

protected:
    //Wrapper for user<->assistant templated conversation
    std::string Generate(const std::string& Prompt);

    bool bIsModelLoaded = false;
    int32 PrevLen = 0;
    FThreadSafeBool bGenerationActive = false;
};