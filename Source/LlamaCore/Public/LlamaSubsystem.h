// Copyright 2025-current Getnamo.

#pragma once
#include "LlamaDataTypes.h"
#include "Tickable.h"
#include "Subsystems/EngineSubsystem.h"

#include "LlamaSubsystem.generated.h"

/** 
* Engine Sub-system type access to LLM. Survives level transitions and PIE start/stop. 
* Limited to one active model, if more are needed in parallel, use LlamaComponent API.
*/

UCLASS(Category = "LLM")
class LLAMACORE_API ULlamaSubsystem : public UEngineSubsystem
{
    GENERATED_BODY()
public:
    virtual void Initialize(FSubsystemCollectionBase& Collection) override;
    virtual void Deinitialize() override;

    //Main callback, updates for each token generated
    UPROPERTY(BlueprintAssignable)
    FOnTokenGeneratedSignature OnTokenGenerated;

    //Only called when full response has been received (EOS/etc)
    UPROPERTY(BlueprintAssignable)
    FOnResponseGeneratedSignature OnResponseGenerated;

    //Utility split emit e.g. sentence level emits, useful for speech generation
    UPROPERTY(BlueprintAssignable)
    FOnPartialSignature OnPartialGenerated;

    UPROPERTY(BlueprintAssignable)
    FOnPromptProcessedSignature OnPromptProcessed;

    UPROPERTY(BlueprintAssignable)
    FVoidEventSignature OnStartEval;

    //Whenever the model stops generating
    UPROPERTY(BlueprintAssignable)
    FOnEndOfStreamSignature OnEndOfStream;

    UPROPERTY(BlueprintAssignable)
    FVoidEventSignature OnContextReset;

    UPROPERTY(BlueprintAssignable)
    FModelNameSignature OnModelLoaded;

    //Catch internal errors
    UPROPERTY(BlueprintAssignable)
    FOnErrorSignature OnError;

    //Modify these before loading model to apply settings
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Component")
    FLLMModelParams ModelParams;

    //This state gets updated typically after every response
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Component")
    FLLMModelState ModelState;

    //Settings
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Component")
    bool bDebugLogModelOutput = false;

    //toggle to pay copy cost or not, default true
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Component")
    bool bSyncPromptHistory = true;

    //loads model from ModelParams
    UFUNCTION(BlueprintCallable, Category = "LLM Model Component")
    void LoadModel();

    UFUNCTION(BlueprintCallable, Category = "LLM Model Component")
    void UnloadModel();

    //Clears the prompt, allowing a new context - optionally keeping the initial system prompt
    UFUNCTION(BlueprintCallable, Category = "LLM Model Component")
    void ResetContextHistory(bool bKeepSystemPrompt = false);

    //removes what the LLM replied
    UFUNCTION(BlueprintCallable, Category = "LLM Model Component")
    void RemoveLastAssistantReply();

    //removes what you said and what the LLM replied
    UFUNCTION(BlueprintCallable, Category = "LLM Model Component")
    void RemoveLastUserInput();

    //Main input function
    UFUNCTION(BlueprintCallable, Category = "LLM Model Component")
    void InsertTemplatedPrompt(UPARAM(meta=(MultiLine=true)) const FString& Text, EChatTemplateRole Role = EChatTemplateRole::User, bool bAddAssistantBOS = false, bool bGenerateReply = true);

    UFUNCTION(BlueprintCallable, Category = "LLM Model Component")
    void InsertTemplatedPromptStruct(const FLlamaChatPrompt& ChatPrompt);

    //does not apply formatting before running inference
    UFUNCTION(BlueprintCallable, Category = "LLM Model Component")
    void InsertRawPrompt(UPARAM(meta = (MultiLine = true)) const FString& Text, bool bGenerateReply = true);

    //Force stop generating new tokens
    UFUNCTION(BlueprintCallable, Category = "LLM Model Component")
    void StopGeneration();

    UFUNCTION(BlueprintCallable, Category = "LLM Model Component")
    void ResumeGeneration();

    //Obtain the currently formatted context
    UFUNCTION(BlueprintPure, Category = "LLM Model Component")
    FString RawContextHistory();

    UFUNCTION(BlueprintPure, Category = "LLM Model Component")
    FStructuredChatHistory GetStructuredChatHistory();

private:
    class FLlamaNative* LlamaNative;
};
