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
        bool bSuccess = Internal->LoadModelFromParams(ModelParams);

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
    if (IsModelLoaded())
    {
        Internal->UnloadModel();
    }
    return true;
}

bool FLlamaNative::IsModelLoaded()
{
    return Internal->IsModelLoaded();
}

void FLlamaNative::InsertPrompt(const FString& UserPrompt)
{
    if (!IsModelLoaded())
    {
        UE_LOG(LlamaLog, Warning, TEXT("Model isn't loaded, can't run prompt."));
        return;
    }

    //check if we're currently generating
    if (Internal->IsGenerating())
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

bool FLlamaNative::IsGenerating()
{
    return Internal->IsGenerating();
}

void FLlamaNative::StopGeneration()
{
    Internal->StopGeneration();
}

void FLlamaNative::ResumeGeneration()
{
    Internal->ResumeGeneration();
}

FString FLlamaNative::RawContextHistory()
{
    if (IsGenerating())
    {
        //Todo: handle this case gracefully
        UE_LOG(LlamaLog, Warning, TEXT("RawContextString cannot be called yet during generation."));
        return FString();
    }

    if (Internal->ContextHistory.Num() == 0)
    {
        return FString();
    }

    // Find the first null terminator (0) in the buffer
    int32 ValidLength = Internal->ContextHistory.Num();
    for (int32 i = 0; i < Internal->ContextHistory.Num(); i++)
    {
        if (Internal->ContextHistory[i] == '\0')
        {
            ValidLength = i;
            break;
        }
    }

    // Convert only the valid part to an FString
    return FString(ValidLength, ANSI_TO_TCHAR(Internal->ContextHistory.GetData()));
}

void FLlamaNative::GetStructuredChatHistory(FStructuredChatHistory& OutChatHistory)
{
    if (IsGenerating())
    {
        //Todo: handle this case gracefully
        UE_LOG(LlamaLog, Warning, TEXT("GetStructuredChatHistory cannot be called yet during generation."));
        return;
    }

    for (const llama_chat_message& Msg : Internal->Messages)
    {
        FStructuredChatMessage StructuredMsg;

        // Convert role
        FString RoleStr = FString(ANSI_TO_TCHAR(Msg.role));
        if (RoleStr.Equals(TEXT("system"), ESearchCase::IgnoreCase))
        {
            StructuredMsg.Role = EChatTemplateRole::System;
        }
        else if (RoleStr.Equals(TEXT("user"), ESearchCase::IgnoreCase))
        {
            StructuredMsg.Role = EChatTemplateRole::User;
        }
        else if (RoleStr.Equals(TEXT("assistant"), ESearchCase::IgnoreCase))
        {
            StructuredMsg.Role = EChatTemplateRole::Assistant;
        }
        else
        {
            // Default/fallback role (adjust if needed)
            StructuredMsg.Role = EChatTemplateRole::Assistant;
        }

        // Convert content
        StructuredMsg.Content = FString(ANSI_TO_TCHAR(Msg.content));

        // Add to history
        OutChatHistory.History.Add(StructuredMsg);
    }
}
