// Copyright 2025-current Getnamo.

#include "LlamaNative.h"
#include "LlamaInternal.h"

FLlamaNative::FLlamaNative()
{
    Internal = new FLlamaInternal();
}

FLlamaNative::~FLlamaNative()
{
    delete Internal;
}

void FLlamaNative::SetModelParams(const FLLMModelParams& Params)
{
	ModelParams = Params;
}

bool FLlamaNative::LoadModel()
{
    //Unload first if any is loaded
    UnloadModel();

    //Now load it
    bool bSuccess = Internal->LoadFromParams(ModelParams);

    //Update model state

    //llama_n_ctx(Internal->Context);
    if (bSuccess)
    {
        ModelState.ChatTemplateLlamaString = FString(Internal->Template);
    }

    return bSuccess;
}

bool FLlamaNative::UnloadModel()
{
    if (bIsModelLoaded())
    {
        Internal->Unload();
    }
    return true;
}

bool FLlamaNative::bIsModelLoaded()
{
    return Internal->bIsLoaded;
}

void FLlamaNative::InsertPrompt(const FString& UserPrompt)
{
    if (!bIsModelLoaded())
    {
        UE_LOG(LlamaLog, Warning, TEXT("Model isn't loaded, can't run prompt."));
        return;
    }

    std::string UserStdString = TCHAR_TO_UTF8(*UserPrompt);
    /* Internal->Messages.Push({"user", _strdup(UserStdString.c_str())});
    int NewLen = llama_chat_apply_template(Internal->Template, Internal->Messages.GetData(), Internal->Messages.Num(), 
        true, Internal->Formatted.GetData(), Internal->Formatted.Num());


    if (NewLen > Internal->Formatted.Num())
    {
        Internal->Formatted.Reserve(NewLen);
        NewLen = llama_chat_apply_template(Internal->Template, Internal->Messages.GetData(), Internal->Messages.Num(),
            true, Internal->Formatted.GetData(), Internal->Formatted.Num());
    }
    if (NewLen < 0)
    {
        UE_LOG(LlamaLog, Warning, TEXT("failed to apply the chat template p1"));
        return;
    }

    std::string PromptStd(Internal->Formatted.GetData() + Internal->PrevLen, Internal->Formatted.GetData() + NewLen);
    FString Prompt = FString(UTF8_TO_TCHAR(PromptStd.c_str()));

    FString Response = Generate(Prompt);
    std::string AssistantStdString = TCHAR_TO_UTF8(*Response);

    Internal->Messages.Push({ "assistant", _strdup(AssistantStdString.c_str()) });

    Internal->PrevLen = llama_chat_apply_template(Internal->Template, Internal->Messages.GetData(), Internal->Messages.Num(), false, nullptr, 0);
    if (Internal->PrevLen < 0) 
    {
        UE_LOG(LlamaLog, Warning, TEXT("failed to apply the chat template p2"));
        return;
    }*/

    if (OnResponseGenerated)
    {
        //OnResponseGenerated(Response);
    }
}
