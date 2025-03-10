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
        const FString Token = FLlamaString::ToUE(TokenPiece);

        if (ModelParams.Advanced.bEmitOnGameThread && OnTokenGenerated)
        {
            Async(EAsyncExecution::TaskGraphMainThread, [this, Token]()
            {
                if(OnTokenGenerated)
                {
                    OnTokenGenerated(Token);
                }
            });
        }
        else
        {
            if (OnTokenGenerated)
            {
                OnTokenGenerated(Token);
            }
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

                if (OnModelLoaded)
                {
                    OnModelLoaded(ModelParams.PathToModel);
                }
            });
        }
        else
        {
            Async(EAsyncExecution::TaskGraphMainThread, [this]
            {
                if (OnError)
                {
                    OnError("Failed loading model see logs.");
                }
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

    //check if we're currently generating
    if (Internal->bGenerationActive)
    {
        UE_LOG(LlamaLog, Warning, TEXT("Aborting: already generating, in this version this fails to queue up."));
        return;
    }

    const std::string UserStdString = FLlamaString::ToStd(UserPrompt);

    //run prompt insert on background thread (NB: should we do one parked thread for llama inference instead of this?)
    Async(EAsyncExecution::ThreadPool, [this, UserStdString]
    {
        FString Response = FLlamaString::ToUE(Internal->InsertPrompt(UserStdString));

        if (ModelParams.Advanced.bEmitOnGameThread && OnResponseGenerated)
        {
            Async(EAsyncExecution::TaskGraphMainThread, [this, Response]
            {
                if (OnResponseGenerated)
                {
                    OnResponseGenerated(Response);
                }
            });
        }
        else
        {
            if (OnResponseGenerated)
            {
                OnResponseGenerated(Response);
            }
        }
    });
}
