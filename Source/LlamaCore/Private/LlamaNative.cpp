// Copyright 2025-current Getnamo.

#include "LlamaNative.h"
#include "LlamaUtility.h"
#include "Internal/LlamaInternal.h"
#include "Async/Async.h"

FLlamaNative::FLlamaNative()
{
    Internal = new FLlamaInternal();

    //Hookup internal listeners
    Internal->OnTokenGenerated = [this](const std::string& TokenPiece)
    {
        FString Token = FLlamaString::ToUE(TokenPiece);

        if (OnTokenGenerated)
        {
            OnTokenGenerated(Token);
        }
    };
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
    Async(EAsyncExecution::Thread, [this]
    {
        //Unload first if any is loaded
        UnloadModel();

        //Now load it
        bool bSuccess = Internal->LoadFromParams(ModelParams);

        //Sync model state
        if (bSuccess)
        {
            FString TemplateString = FString(Internal->Template);

            //update model params on game thread
            Async(EAsyncExecution::TaskGraphMainThread, [this, TemplateString]
            {
                ModelState.ChatTemplateLlamaString = TemplateString;
            });
        }

    });

    return true;
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

    const std::string UserStdString = FLlamaString::ToStd(UserPrompt);

    //run prompt insert on background thread
    Async(EAsyncExecution::Thread, [this, UserStdString]
    {
        FString Response = FLlamaString::ToUE(Internal->InsertPrompt(UserStdString));

        Async(EAsyncExecution::TaskGraphMainThread, [this, Response]
        {
            if (OnResponseGenerated)
            {
                OnResponseGenerated(Response);
            }
        });
    });
}
