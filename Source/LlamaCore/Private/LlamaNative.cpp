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

        if (ModelParams.Advanced.bEmitOnGameThread && OnTokenGenerated && bCallbacksAreValid)
        {
            Async(EAsyncExecution::TaskGraphMainThread, [this, Token]()
            {
                if(OnTokenGenerated && bCallbacksAreValid)
                {
                    OnTokenGenerated(Token);
                }
            });
        }
        else
        {
            if (OnTokenGenerated && bCallbacksAreValid)
            {
                OnTokenGenerated(Token);
            }
        }
    };

    bCallbacksAreValid = true;
}

FLlamaNative::~FLlamaNative()
{
    bCallbacksAreValid = false;
    StopGeneration();

    while (bThreadIsActive) 
    {
        FPlatformProcess::Sleep(0.01f);
    }
    delete Internal;
}

void FLlamaNative::SetModelParams(const FLLMModelParams& Params)
{
	ModelParams = Params;
}

bool FLlamaNative::LoadModel()
{
    bThreadIsActive = true;

    Async(EAsyncExecution::Thread, [this]
    {
        //Unload first if any is loaded
        UnloadModel();

        //Now load it
        bool bSuccess = Internal->LoadModelFromParams(ModelParams);

        //Sync model state
        if (bSuccess)
        {
            const FString TemplateString = FLlamaString::ToUE(Internal->Template);
            const FString TemplateSource = FLlamaString::ToUE(Internal->TemplateSource);

            //update model params on game thread
            Async(EAsyncExecution::TaskGraphMainThread, [this, TemplateString, TemplateSource]
            {
                if (!bCallbacksAreValid)
                {
                    bThreadIsActive = false;
                    return;
                }

                FJinjaChatTemplate ChatTemplate;
                ChatTemplate.TemplateSource = TemplateSource;
                ChatTemplate.Jinja = TemplateString;

                ModelState.ChatTemplateInUse = ChatTemplate;
                
                bThreadIsActive = false;

                if (OnModelStateChanged)
                {
                    OnModelStateChanged(ModelState);
                }

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
                if (OnError && bCallbacksAreValid)
                {
                    OnError("Failed loading model see logs.");
                }
                bThreadIsActive = false;
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

    if (bThreadIsActive)
    {
        UE_LOG(LlamaLog, Warning, TEXT("Prompting while generation is active isn't currently supported, prompt not sent."));
        return;
    }

    const std::string UserStdString = FLlamaString::ToStd(UserPrompt);

    bThreadIsActive = true;

    //run prompt insert on background thread (NB: should we do one parked thread for llama inference instead of this?)
    Async(EAsyncExecution::ThreadPool, [this, UserStdString]
    {
        FString Response = FLlamaString::ToUE(Internal->InsertTemplatedPrompt(UserStdString));

        //It's now safe to sync our history - only
        GetStructuredChatHistory(ModelState.ChatHistory);
        RawContextHistory(ModelState.ContextHistory);
        ModelState.ContextLength = UsedContextLength();

        if (ModelParams.Advanced.bEmitOnGameThread && OnResponseGenerated && bCallbacksAreValid)
        {
            Async(EAsyncExecution::TaskGraphMainThread, [this, Response]
            {
                if (bCallbacksAreValid)
                {
                    if (OnModelStateChanged)
                    {
                        OnModelStateChanged(ModelState);
                    }

                    if (OnResponseGenerated)
                    {
                        OnResponseGenerated(Response);
                    }
                }

                bThreadIsActive = false;
            });
        }
        else
        {
            if (bCallbacksAreValid)
            {
                if (OnModelStateChanged)
                {
                    OnModelStateChanged(ModelState);
                }
                if (OnResponseGenerated)
                {
                    OnResponseGenerated(Response);
                }
            }
            bThreadIsActive = false;
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

void FLlamaNative::OnTick(float DeltaTime)
{
}

void FLlamaNative::ResetContextHistory()
{
    //TODO: implement
}

void FLlamaNative::RemoveLastInput()
{
    //TODO: implement
}

void FLlamaNative::RemoveLastReply()
{
    //Rollback messages
    //Reformat history from rollback
    //Ready to continue
    //TODO: implement
}

void FLlamaNative::RegenerateLastReply()
{
    RemoveLastReply();
    //Change seed?
    ResumeGeneration();
}

int32 FLlamaNative::RawContextHistory(FString& OutContextString)
{
    if (IsGenerating())
    {
        //Todo: handle this case gracefully
        UE_LOG(LlamaLog, Warning, TEXT("RawContextString cannot be called yet during generation."));
        return -1;
    }

    if (Internal->ContextHistory.Num() == 0)
    {
        return 0;
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
    OutContextString = FString(ValidLength, ANSI_TO_TCHAR(Internal->ContextHistory.GetData()));

    return ValidLength;
}

void FLlamaNative::GetStructuredChatHistory(FStructuredChatHistory& OutChatHistory)
{
    if (IsGenerating())
    {
        //Todo: handle this case gracefully
        UE_LOG(LlamaLog, Warning, TEXT("GetStructuredChatHistory cannot be called yet during generation."));
        return;
    }

    OutChatHistory.History.Empty();

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

int32 FLlamaNative::UsedContextLength()
{
    return Internal->UsedContext();
}
