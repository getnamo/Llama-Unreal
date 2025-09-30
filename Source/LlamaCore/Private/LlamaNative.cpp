// Copyright 2025-current Getnamo.

#include "LlamaNative.h"
#include "LlamaUtility.h"
#include "Internal/LlamaInternal.h"
#include "Async/TaskGraphInterfaces.h"
#include "Async/Async.h"
#include "Tickable.h"

FLlamaNative::FLlamaNative()
{
    Internal = new FLlamaInternal();

    //Hookup internal listeners - these get called on BG thread
    Internal->OnTokenGenerated = [this](const std::string& TokenPiece)
    {
        const FString Token = FLlamaString::ToUE(TokenPiece);

        //Accumulate
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
            if (!Partial.IsEmpty())
            {
                CombinedTextOnPartialEmit = CombinedPieceText;
            }
        }

        //Emit token to game thread
        if (OnTokenGenerated)
        {
            EnqueueGTTask([this, Token, Partial]()
            {
                if (OnTokenGenerated)
                {
                    OnTokenGenerated(Token);
                }
                if (OnPartialGenerated && !Partial.IsEmpty())
                {
                    OnPartialGenerated(Partial);
                }
            });
        }
    };

    Internal->OnGenerationComplete = [this](const std::string& Response, float Duration, int32 TokensGenerated, float SpeedTps)
    {
        if (ModelParams.Advanced.bLogGenerationStats)
        {
            UE_LOG(LlamaLog, Log, TEXT("TGS - Generated %d tokens in %1.2fs (%1.2ftps)"), TokensGenerated, Duration, SpeedTps);
        }

        int32 UsedContext = UsedContextLength();

        //Sync history data on bg thread
        SyncModelStateToInternal([this, UsedContext, SpeedTps]
        {
            ModelState.ContextUsed = UsedContext;
            ModelState.LastTokenGenerationSpeed = SpeedTps;
        });

        FString Partial;

        //Emit last full partial if we didn't end on punctuation
        if (ModelParams.Advanced.bEmitPartials && CombinedTextOnPartialEmit.Len() != CombinedPieceText.Len())
        {
            Partial = FLlamaString::GetLastSentence(CombinedPieceText);
        }

        //Clear our partial text parser
        CombinedPieceText.Empty();
        CombinedTextOnPartialEmit.Empty();

        //Emit response generated to general listeners
        FString ResponseString = FLlamaString::ToUE(Response);
        EnqueueGTTask([this, ResponseString, Partial]
        {
            //ensure partials are fully emitted too
            if (OnPartialGenerated && !Partial.IsEmpty())
            {
                OnPartialGenerated(Partial);
            }
            if (OnResponseGenerated)
            {
                OnResponseGenerated(ResponseString);
            }
        });
    };

    Internal->OnPromptProcessed = [this](int32 TokensProcessed, EChatTemplateRole RoleProcessed, float SpeedTps)
    {
        if (ModelParams.Advanced.bLogGenerationStats)
        {
            UE_LOG(LlamaLog, Log, TEXT("PPS - Processed %d tokens at %1.2ftps"), TokensProcessed, SpeedTps);
        }

        int32 UsedContext = UsedContextLength();

        //Sync history data with additional state updates
        SyncModelStateToInternal([this, UsedContext, SpeedTps]
        {
            ModelState.ContextUsed = UsedContext;
            ModelState.LastPromptProcessingSpeed = SpeedTps;
        });

        //Separate enqueue to ensure it happens after modelstate update
        EnqueueGTTask([this, TokensProcessed, RoleProcessed, SpeedTps]
        {
            if (OnPromptProcessed)
            {
                OnPromptProcessed(TokensProcessed, RoleProcessed, SpeedTps);
            }
        });
    };

    Internal->OnError = [this](const FString& ErrorMessage, int32 ErrorCode)
    {
        const FString ErrorMessageGTSafe = ErrorMessage;
        EnqueueGTTask([this, ErrorMessageGTSafe, ErrorCode]
        {
            if (OnError)
            {
                OnError(ErrorMessageGTSafe, ErrorCode);
            }
        });
    };
}

FLlamaNative::~FLlamaNative()
{
    StopGeneration();
    bThreadShouldRun = false;
    
    //Remove ticker if active
    RemoveTicker();

    //Wait for the thread to stop
    while (bThreadIsActive) 
    {
        FPlatformProcess::Sleep(0.01f);
    }
    delete Internal;
}

void FLlamaNative::SyncModelStateToInternal(TFunction<void()> AdditionalGTStateUpdates)
{
    TFunction<void(int64)> BGSyncAction = [this, AdditionalGTStateUpdates](int64 TaskId)
    {
        //Copy states internal states, emit to 
        FStructuredChatHistory ChatHistory;
        FString ContextHistory;
        GetStructuredChatHistory(ChatHistory);
        RawContextHistory(ContextHistory);

        EnqueueGTTask([this, ChatHistory, ContextHistory, AdditionalGTStateUpdates]
        {
            //Update state on gamethread
            ModelState.ChatHistory = ChatHistory;
            ModelState.ContextHistory = ContextHistory;

            //derived state update
            if (ModelState.ChatHistory.History.Num() > 0)
            {
                ModelState.LastRole = ModelState.ChatHistory.History.Last().Role;
            }

            //Run the updates before model state changes happen
            if (AdditionalGTStateUpdates)
            {
                AdditionalGTStateUpdates();
            }

            //Emit model state update to GT listeners
            if (OnModelStateChanged)
            {
                OnModelStateChanged(ModelState);
            }
        }, TaskId);
    };

    if (IsInGameThread())
    {
        EnqueueBGTask(BGSyncAction);
    }
    else
    {
        //Call directly
        BGSyncAction(GetNextTaskId());
    }
    
}

void FLlamaNative::StartLLMThread()
{
    bThreadShouldRun = true;
    Async(EAsyncExecution::Thread, [this]
    {
        bThreadIsActive = true;

        while (bThreadShouldRun)
        {
            //Run all queued tasks
            while (!BackgroundTasks.IsEmpty())
            {
                FLLMThreadTask Task;
                BackgroundTasks.Dequeue(Task);
                if (Task.TaskFunction)
                {
                    //Run Task
                    Task.TaskFunction(Task.TaskId);
                }
            }

            FPlatformProcess::Sleep(ThreadIdleSleepDuration);
        }

        bThreadIsActive = false;
    });
}

int64 FLlamaNative::GetNextTaskId()
{
    //technically returns an int32
    return TaskIdCounter.Increment();
}

void FLlamaNative::EnqueueBGTask(TFunction<void(int64)> TaskFunction)
{
    //Lazy start the thread on first enqueue
    if (!bThreadIsActive)
    {
        StartLLMThread();
    }

    FLLMThreadTask Task;
    Task.TaskId = GetNextTaskId();
    Task.TaskFunction = TaskFunction;

    BackgroundTasks.Enqueue(Task);
}

void FLlamaNative::EnqueueGTTask(TFunction<void()> TaskFunction, int64 LinkedTaskId)
{
    FLLMThreadTask Task;
    
    if (LinkedTaskId == -1)
    {
        Task.TaskId = GetNextTaskId();
    }
    else
    {
        Task.TaskId = LinkedTaskId;
    }

    Task.TaskFunction = [TaskFunction](int64 InTaskId) 
    {
        TaskFunction();
    };

    GameThreadTasks.Enqueue(Task);
}

void FLlamaNative::SetModelParams(const FLLMModelParams& Params)
{
	ModelParams = Params;
}

void FLlamaNative::LoadModel(bool bForceReload, TFunction<void(const FString&, int32 StatusCode)> ModelLoadedCallback)
{
    if (IsModelLoaded() && !bForceReload)
    {
        //already loaded, we're done
        return ModelLoadedCallback(ModelParams.PathToModel, 0);
    }
    bModelLoadInitiated = true;

    //Copy so these dont get modified during enqueue op
    const FLLMModelParams ParamsAtLoad = ModelParams;

    EnqueueBGTask([this, ParamsAtLoad, ModelLoadedCallback](int64 TaskId)
    {
        //Unload first if any is loaded
        Internal->UnloadModel();

        //Now load it
        bool bSuccess = Internal->LoadModelFromParams(ParamsAtLoad);

        //Sync model state
        if (bSuccess)
        {
            const FString TemplateString = FLlamaString::ToUE(Internal->Template);
            const FString TemplateSource = FLlamaString::ToUE(Internal->TemplateSource);

            //Before we release the BG thread, ensure we enqueue the system prompt
            //If we do it later, other queued calls will frontrun it. This enables startup chaining correctly
            if (ParamsAtLoad.bAutoInsertSystemPromptOnLoad)
            {
                Internal->InsertTemplatedPrompt(FLlamaString::ToStd(ParamsAtLoad.SystemPrompt), EChatTemplateRole::System, false, false);
            }

            //Callback on game thread for data sync
            EnqueueGTTask([this, TemplateString, TemplateSource, ModelLoadedCallback]
            {
                FJinjaChatTemplate ChatTemplate;
                ChatTemplate.TemplateSource = TemplateSource;
                ChatTemplate.Jinja = TemplateString;

                ModelState.ChatTemplateInUse = ChatTemplate;
                ModelState.bModelIsLoaded = true;

                bModelLoadInitiated = false;

                if (OnModelStateChanged)
                {
                    OnModelStateChanged(ModelState);
                }

                if (ModelLoadedCallback)
                {
                    ModelLoadedCallback(ModelParams.PathToModel, 0);
                }
            }, TaskId);
        }
        else
        {
            EnqueueGTTask([this, ModelLoadedCallback]
            {
                bModelLoadInitiated = false;

                //On error will be triggered earlier in the chain, but forward our model loading error status here
                if (ModelLoadedCallback)
                {
                    ModelLoadedCallback(ModelParams.PathToModel, 15);
                }
            }, TaskId);
        }
    });
}

void FLlamaNative::UnloadModel(TFunction<void(int32 StatusCode)> ModelUnloadedCallback)
{
    bModelLoadInitiated = false;

    EnqueueBGTask([this, ModelUnloadedCallback](int64 TaskId)
    {
        if (IsModelLoaded())
        {
            Internal->UnloadModel();
        }

        //Reply with code
        EnqueueGTTask([this, ModelUnloadedCallback]
        {
            ModelState.bModelIsLoaded = false;

            if (OnModelStateChanged)
            {
                OnModelStateChanged(ModelState);
            }

            if (ModelUnloadedCallback)
            {
                ModelUnloadedCallback(0);
            }
        });
    });
}

bool FLlamaNative::IsModelLoaded()
{
    return Internal->IsModelLoaded();
}

void FLlamaNative::InsertTemplatedPrompt(const FLlamaChatPrompt& Prompt, TFunction<void(const FString& Response)> OnResponseFinished)
{
    if (!IsModelLoaded() && !bModelLoadInitiated)
    {
        UE_LOG(LlamaLog, Warning, TEXT("Model isn't loaded, can't run prompt."));
        return;
    }

    //Copy so we can deal with it on different threads
    FLlamaChatPrompt ThreadSafePrompt = Prompt;

    //run prompt insert on a background thread
    EnqueueBGTask([this, ThreadSafePrompt, OnResponseFinished](int64 TaskId)
    {
        const std::string UserStdString = FLlamaString::ToStd(ThreadSafePrompt.Prompt);
        
        if (ThreadSafePrompt.bGenerateReply)
        {
            FString Response = FLlamaString::ToUE(Internal->InsertTemplatedPrompt(UserStdString, ThreadSafePrompt.Role, ThreadSafePrompt.bAddAssistantBOS, true));

            //NB: OnResponseGenerated will also be called separately from this
            EnqueueGTTask([this, Response, OnResponseFinished]()
            {
                if (OnResponseFinished)
                {
                    OnResponseFinished(Response);
                }
            });
        }
        else
        {
            //We don't want to generate a reply, just append a prompt. (last param = false turns it off)
            Internal->InsertTemplatedPrompt(UserStdString, ThreadSafePrompt.Role, ThreadSafePrompt.bAddAssistantBOS, false);
        }
    });
}

void FLlamaNative::InsertRawPrompt(const FString& Prompt, bool bGenerateReply, TFunction<void(const FString& Response)>OnResponseFinished)
{
    if (!IsModelLoaded() && !bModelLoadInitiated)
    {
        UE_LOG(LlamaLog, Warning, TEXT("Model isn't loaded, can't run prompt."));
        return;
    }

    const std::string PromptStdString = FLlamaString::ToStd(Prompt);

    EnqueueBGTask([this, PromptStdString, OnResponseFinished, bGenerateReply](int64 TaskId)
    {
        FString Response = FLlamaString::ToUE(Internal->InsertRawPrompt(PromptStdString, bGenerateReply));
        EnqueueGTTask([this, Response, OnResponseFinished]
        {
            if (OnResponseFinished)
            {
                OnResponseFinished(Response);
            }
        });
    });
}

void FLlamaNative::ImpersonateTemplatedPrompt(const FLlamaChatPrompt& Prompt)
{
    //modify model state
    if (IsModelLoaded())
    {
        //insert it but make sure we don't do any token generation
        FLlamaChatPrompt ModifiedPrompt = Prompt;
        ModifiedPrompt.bGenerateReply = false;

        InsertTemplatedPrompt(ModifiedPrompt);
    }
    else
    {
        //no model, so just run this in sync mode
        FStructuredChatMessage Message;
        Message.Role = Prompt.Role;
        Message.Content = Prompt.Prompt;

        //modify our chat history state
        ModelState.ChatHistory.History.Add(Message);

        if (OnModelStateChanged)
        {
            OnModelStateChanged(ModelState);
        }
        //was this an assistant message? emit response generated callback
        if (Message.Role == EChatTemplateRole::Assistant)
        {
            if (OnResponseGenerated)
            {
                OnResponseGenerated(Prompt.Prompt);
            }
        }
    }
}

void FLlamaNative::ImpersonateTemplatedToken(const FString& Token, EChatTemplateRole Role, bool bEoS)
{
    //Should be called on game thread.
    
    //NB: we don't support updating model internal state atm

    //Check if we need to add a message before modifying it
    bool bLastRoleWasMatchingRole = false;

    if (ModelState.ChatHistory.History.Num() != 0)
    {
        FStructuredChatMessage& Message = ModelState.ChatHistory.History.Last();
        bLastRoleWasMatchingRole = Message.Role == Role;
    }

    FString CurrentReplyText;
    
    if (!bLastRoleWasMatchingRole)
    {
        FStructuredChatMessage Message;
        Message.Role = Role;
        Message.Content = Token;

        ModelState.ChatHistory.History.Add(Message);

        ThenTimeStamp = FPlatformTime::Seconds();
        ImpersonationTokenCount = 1;


        CurrentReplyText += Token;
    }
    else
    {
        FStructuredChatMessage& Message = ModelState.ChatHistory.History.Last();
        Message.Content += Token;
        ImpersonationTokenCount++;

        CurrentReplyText += Message.Content;
    }

    FStructuredChatMessage& Message = ModelState.ChatHistory.History.Last();

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
            Partial = FLlamaString::GetLastSentence(CurrentReplyText);
        }
    }

    //Emit token to game thread
    if (OnTokenGenerated)
    {
        OnTokenGenerated(Token);

        if (OnPartialGenerated && !Partial.IsEmpty())
        {
            OnPartialGenerated(Partial);
        }
    }

    //full response reply on finish
    if (bEoS)
    {
        double Duration = FPlatformTime::Seconds() - ThenTimeStamp;
        double TotalTokens = ImpersonationTokenCount;
        ImpersonationTokenCount = 0;

        ModelState.LastPromptProcessingSpeed = 0;   //this can't be measured without more imput
        ModelState.LastTokenGenerationSpeed = TotalTokens / Duration;
        ModelState.LastRole = EChatTemplateRole::Assistant;

        if (OnModelStateChanged)
        {
            OnModelStateChanged(ModelState);
        }
        if (OnResponseGenerated)
        {
            OnResponseGenerated(CurrentReplyText);
        }
    }
}

void FLlamaNative::RemoveLastNMessages(int32 MessageCount)
{
    EnqueueBGTask([this, MessageCount](int64 TaskId)
    {
        Internal->RollbackContextHistoryByMessages(MessageCount);

        //Sync state
        SyncModelStateToInternal();
    });
}

void FLlamaNative::RemoveLastNTokens(int32 TokensCount)
{
    EnqueueBGTask([this, TokensCount](int64 TaskId)
    {
        Internal->RollbackContextHistoryByTokens(TokensCount);

        //Sync state
        SyncModelStateToInternal();
    });
}

bool FLlamaNative::IsGenerating()
{
    //this is threadsafe
    return Internal->IsGenerating();
}

void FLlamaNative::StopGeneration()
{
    //this is threadsafe
    Internal->StopGeneration();
}

void FLlamaNative::ResumeGeneration()
{
    if (!IsModelLoaded())
    {
        UE_LOG(LlamaLog, Warning, TEXT("Model isn't loaded, can't ResumeGeneration."));
        return;
    }

    EnqueueBGTask([this](int64 TaskId)
    {
        Internal->ResumeGeneration();
    });
}

void FLlamaNative::ClearPendingTasks(bool bClearGameThreadCallbacks)
{
    BackgroundTasks.Empty();

    if (bClearGameThreadCallbacks)
    {
        GameThreadTasks.Empty();
    }
}

void FLlamaNative::OnGameThreadTick(float DeltaTime)
{
    //Handle all the game thread callbacks
    if (!GameThreadTasks.IsEmpty())
    {
        //Run all queued tasks
        while (!GameThreadTasks.IsEmpty())
        {
            FLLMThreadTask Task;
            GameThreadTasks.Dequeue(Task);
            if (Task.TaskFunction)
            {
                //Run Task
                Task.TaskFunction(Task.TaskId);
            }
        }
    }
}

void FLlamaNative::AddTicker()
{
    TickDelegateHandle = FTSTicker::GetCoreTicker().AddTicker(FTickerDelegate::CreateLambda([this](float DeltaTime)
    {
        OnGameThreadTick(DeltaTime);
        return true;
    }));
}

void FLlamaNative::RemoveTicker()
{
    if (IsNativeTickerActive())
    {
        FTSTicker::GetCoreTicker().RemoveTicker(TickDelegateHandle);
        TickDelegateHandle = nullptr;
    }
}

bool FLlamaNative::IsNativeTickerActive()
{
    return TickDelegateHandle.IsValid();
}

void FLlamaNative::ResetContextHistory(bool bKeepSystemPrompt)
{
    EnqueueBGTask([this, bKeepSystemPrompt](int64 TaskId)
    {
        Internal->ResetContextHistory(bKeepSystemPrompt);

        //Lazy keep version, just re-insert. TODO: implement optimized reset
        /*if (bKeepSystemPrompt)
        {
            Internal->InsertTemplatedPrompt(ModelParams.SystemPrompt, EChatTemplateRole::System, false, false);
        }*/

        SyncModelStateToInternal();
    });
}

void FLlamaNative::RemoveLastUserInput()
{
    //lazily removes last reply and last input
    RemoveLastNMessages(2);
}

void FLlamaNative::RemoveLastReply()
{
    RemoveLastNMessages(1);
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

void FLlamaNative::SyncPassedModelStateToNative(FLLMModelState& StateToSync)
{
    StateToSync = ModelState;
}

int32 FLlamaNative::UsedContextLength()
{
    return Internal->UsedContext();
}

FString FLlamaNative::WrapPromptForRole(const FString& Text, EChatTemplateRole Role, const FString& OverrideTemplate, bool bAddAssistantBoS)
{
    return FLlamaString::ToUE( Internal->WrapPromptForRole(FLlamaString::ToStd(Text), Role, FLlamaString::ToStd(OverrideTemplate), bAddAssistantBoS) );
}


void FLlamaNative::GetPromptEmbeddings(const FString& Text, TFunction<void(const TArray<float>& Embeddings, const FString& SourceText)> OnEmbeddings)
{
    const FString SourceText = Text;    //copy to safely traverse threads

    EnqueueBGTask([this, SourceText, OnEmbeddings](int64 TaskId)
    {
        std::string TextStd = FLlamaString::ToStd(SourceText);
        std::vector<float> EmbeddingVector;
        Internal->GetPromptEmbeddings(TextStd, EmbeddingVector);

        TArray<float> Embeddings;
        Embeddings.Append(EmbeddingVector.data(), EmbeddingVector.size());

        EnqueueGTTask([this, OnEmbeddings, Embeddings, SourceText]
        {
            if (OnEmbeddings)
            {
                OnEmbeddings(Embeddings, SourceText);
            }
        });
    });
}