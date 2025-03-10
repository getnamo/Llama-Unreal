// Copyright 2025-current Getnamo.

#pragma once
#include <Components/ActorComponent.h>
#include <CoreMinimal.h>
#include "LlamaDataTypes.h"

#include "LlamaComponent.generated.h"

UCLASS(Category = "LLM", BlueprintType, meta = (BlueprintSpawnableComponent))
class LLAMACORE_API ULlamaComponent : public UActorComponent
{
    GENERATED_BODY()
public:
    ULlamaComponent(const FObjectInitializer &ObjectInitializer);
    ~ULlamaComponent();

    virtual void Activate(bool bReset) override;
    virtual void Deactivate() override;
    virtual void TickComponent(float DeltaTime,
                                ELevelTick TickType,
                                FActorComponentTickFunction* ThisTickFunction) override;

    //Main callback, updates for each token generated
    UPROPERTY(BlueprintAssignable)
    FOnTokenGeneratedSignature OnTokenGenerated;

    //Only called when full response has been received (EOS/etc)
    UPROPERTY(BlueprintAssignable)
    FOnTokenGeneratedSignature OnResponseGenerated;

    //Utility split emit e.g. sentence level emits, useful for speech generation
    UPROPERTY(BlueprintAssignable)
    FOnPartialSignature OnPartialParsed;

    UPROPERTY(BlueprintAssignable)
    FVoidEventSignature OnStartEval;

    //Whenever the model stops generating
    UPROPERTY(BlueprintAssignable)
    FOnEndOfStreamSignature OnEndOfStream;

    UPROPERTY(BlueprintAssignable)
    FVoidEventSignature OnContextReset;

    //Catch internal errors
    UPROPERTY(BlueprintAssignable)
    FOnErrorSignature OnError;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Component")
    FLLMModelParams ModelParams;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Component")
    FLLMModelState ModelState;

    //Settings
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Component")
    bool bDebugLogModelOutput = false;

    //toggle to pay copy cost or not, default true
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Component")
    bool bSyncPromptHistory = true;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Component")
    TMap<FString, FChatTemplate> CommonChatTemplates;

    UFUNCTION(BlueprintCallable, Category = "LLM Model Component")
    void InsertPrompt(UPARAM(meta=(MultiLine=true)) const FString &Text);

    /** 
    * Use this function to bypass input from AI, e.g. streaming input from another source. 
    * All downstream event functions will trigger from this call as if it came from the LLM.
    * Won't make a new message split until role is swapped from last. */
    UFUNCTION(BlueprintCallable, Category = "LLM Model Component")
    void UserImpersonateText(const FString& Text, EChatTemplateRole Role = EChatTemplateRole::Assistant,  bool bIsEos = false);

    UFUNCTION(BlueprintPure, Category = "LLM Model Component")
    FString WrapPromptForRole(const FString& Text, EChatTemplateRole Role, bool AppendModelRolePrefix=false);

    UFUNCTION(BlueprintPure, Category = "LLM Model Component")
    FString GetRolePrefix(EChatTemplateRole Role = EChatTemplateRole::Assistant);

    //This will wrap your input given the specific role using chat template specified
    UFUNCTION(BlueprintCallable, Category = "LLM Model Component")
    void InsertPromptTemplated(UPARAM(meta=(MultiLine=true)) const FString& Text, EChatTemplateRole Role);


    UFUNCTION(BlueprintCallable, Category = "LLM Model Component")
    void StartStopQThread(bool bShouldRun = true);

    //Force stop generating new tokens
    UFUNCTION(BlueprintCallable, Category = "LLM Model Component")
    void StopGenerating();

    UFUNCTION(BlueprintCallable, Category = "LLM Model Component")
    void ResumeGenerating();

    UFUNCTION(BlueprintCallable, Category = "LLM Model Component")
    void SyncParamsToLlama();


    UFUNCTION(BlueprintPure, Category = "LLM Model Component")
    FString GetTemplateStrippedPrompt();

    FStructuredChatMessage FirstChatMessageInHistory(const FString& History, FString& Remainder);

    UFUNCTION(BlueprintPure, Category = "LLM Model Component")
    FStructuredChatHistory GetStructuredHistory();


    //String Utility
    bool IsSentenceEndingPunctuation(const TCHAR Char);
    FString GetLastSentence(const FString& InputString);

    EChatTemplateRole LastRoleFromStructuredHistory();


    //Utility function for debugging model location and file enumeration
    UFUNCTION(BlueprintCallable, Category = "LLM Model Component")
    TArray<FString> DebugListDirectoryContent(const FString& InPath);

private:
    class FLlamaNative* LlamaNative;

    TFunction<void(FString, int32)> TokenCallbackInternal;
};
