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

        //Accumalate
        CombinedPieceText += Token;

        FString Partial;

        //Compute Partials
        if (ModelParams.Advanced.bEmitPartials)
        {
            bool bSplitFound = false;
            //Check new token for separators
            for (const FString& Separator : ModelParams.Advanced.PartialsSeparators)
            {
                if (Token.Contains(Separator))
                {
                    bSplitFound = true;
                }
            }
            if (bSplitFound)
            {
                Partial = FLlamaString::GetLastSentence(CombinedPieceText);
            }
        }

        //Emit token to game thread
        if (OnTokenGenerated && bCallbacksAreValid)
        {
            Async(EAsyncExecution::TaskGraphMainThread, [this, Token, Partial]()
            {
                if(bCallbacksAreValid)
                {
                    if (OnTokenGenerated)
                    {
                        OnTokenGenerated(Token);
                    }
                    if (OnPartialGenerated && !Partial.IsEmpty())
                    {
                        OnPartialGenerated(Partial);
                    }
                }
            });
        }
    };

    Internal->OnGenerationComplete = [this](const std::string& Response, float Duration, int32 TokensGenerated, float SpeedTPS)
    {
        if (ModelParams.Advanced.bLogGenerationStats)
        {
            UE_LOG(LlamaLog, Log, TEXT("Generated %d tokens in %1.2fs (%1.2ftps)"), TokensGenerated, Duration, SpeedTPS);
        }

        FStructuredChatHistory ChatHistory;
        FString ContextHistory;

        //It's now safe to sync our history - only
        GetStructuredChatHistory(ChatHistory);
        RawContextHistory(ContextHistory);
        int32 UsedContext = UsedContextLength();

        //Clear our partial text parser
        CombinedPieceText.Empty();

        FString ResponseString = FLlamaString::ToUE(Response);

        if (bCallbacksAreValid)
        {
            Async(EAsyncExecution::TaskGraphMainThread, [this, ResponseString, ChatHistory, ContextHistory, UsedContext]
            {
                if (bCallbacksAreValid)
                {
                    //Sync state information
                    ModelState.ContextLength = UsedContext;
                    ModelState.ChatHistory = ChatHistory;
                    ModelState.ContextHistory = ContextHistory;
                    if (ChatHistory.History.Num() > 0)
                    {
                        ModelState.LastRole = ChatHistory.History.Last().Role;
                    }

                    if (OnModelStateChanged)
                    {
                        OnModelStateChanged(ModelState);
                    }

                    if (OnResponseGenerated)
                    {
                        OnResponseGenerated(ResponseString);
                    }
                }
            });
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

void FLlamaNative::InsertTemplatedPrompt(const FString& Prompt, EChatTemplateRole Role, bool bAddAssistantBOS, bool bGenerateReply)
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

    const std::string UserStdString = FLlamaString::ToStd(Prompt);

    bThreadIsActive = true;

    //run prompt insert on a background thread
    Async(EAsyncExecution::ThreadPool, [this, UserStdString, Role, bAddAssistantBOS, bGenerateReply]
    {
        if (bGenerateReply)
        {
            FLlamaString::ToUE(Internal->InsertTemplatedPromptAndGenerate(UserStdString, Role, bAddAssistantBOS));
        }
        else
        {
            Internal->InsertTemplatedPrompt(UserStdString, Role, bAddAssistantBOS);
        }

        bThreadIsActive = false;
    });
}

void FLlamaNative::InsertRawPrompt(const FString& Prompt)
{
}

void FLlamaNative::RemoveLastNMessages(int32 MessageCount)
{
    Internal->RollbackContextHistoryByMessages(MessageCount);
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

    if (Internal->ContextHistory.size() == 0)
    {
        return 0;
    }

    // Find the first null terminator (0) in the buffer
    int32 ValidLength = Internal->ContextHistory.size();
    for (int32 i = 0; i < Internal->ContextHistory.size(); i++)
    {
        if (Internal->ContextHistory[i] == '\0')
        {
            ValidLength = i;
            break;
        }
    }

    // Convert only the valid part to an FString
    OutContextString = FString(ValidLength, ANSI_TO_TCHAR(Internal->ContextHistory.data()));

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
