#pragma once

#include <string>
#include "llama.h"
#include "common/common.h"

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

    TFunction<void(const std::string& TokenPiece)>OnTokenGenerated = nullptr;

    bool bIsLoaded = false;

    FThreadSafeBool bShouldGenerate = false;

    //Messaging state
    TArray<llama_chat_message> Messages;
    TArray<char> Formatted;

    char* Template;
    int32 PrevLen = 0;

    //FThreadSafeBool bIsThreadRunning;

    bool LoadFromParams(const FLLMModelParams& InModelParams);

    //Wrapper for user<->assistant templated conversation
    std::string Generate(const std::string& Prompt);

    std::string InsertPrompt(const std::string& Prompt);

    void Unload();

    ~FLlamaInternal();
};