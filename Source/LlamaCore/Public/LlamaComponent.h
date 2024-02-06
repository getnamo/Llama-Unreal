// 2023 (c) Mika Pi, Modifications Getnamo

#pragma once
#include <Components/ActorComponent.h>
#include <CoreMinimal.h>
#include <memory>

#include "LlamaComponent.generated.h"

namespace Internal
{
  class FLlama;
}

DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnErrorSignature, FString, ErrorMessage);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnNewTokenGeneratedSignature, FString, NewToken);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnPartialSignature, const FString&, Partial);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnPromptHistorySignature, FString, History);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_TwoParams(FOnEndOfStreamSignature, bool, bStopSequenceTriggered, float, TokensPerSecond);
DECLARE_DYNAMIC_MULTICAST_DELEGATE(FVoidEventSignature);

USTRUCT(BlueprintType)
struct FLLMModelAdvancedParams
{
    GENERATED_USTRUCT_BODY();

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Advanced Params")
    float Temp = 0.80f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Advanced Params")
    int32 TopK = 40;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Advanced Params")
    float TopP = 0.95f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Advanced Params")
    float TfsZ = 1.00f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Advanced Params")
    float TypicalP = 1.00f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Advanced Params")
    int32 RepeatLastN = 64;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Advanced Params")
    float RepeatPenalty = 1.10f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Advanced Params")
    float AlphaPresence = 0.00f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Advanced Params")
    float AlphaFrequency = 0.00f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Advanced Params")
    int32 Mirostat = 0;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Advanced Params")
    float MirostatTau = 5.f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Advanced Params")
    float MirostatEta = 0.1f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Advanced Params")
    int MirostatM = 100;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Advanced Params")
    bool PenalizeNl = true;

    //automatically loads template from gguf. Use Empty default template to not override this value.
    //not yet properly implemented
    //UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Params")
    bool bLoadTemplateFromGGUFIfAvailable = true;

    //If true, upon EOS, it will cleanup history such that correct EOS is placed
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Params")
    bool bEnforceModelEOSFormat = true;

    //synced per eos
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Params")
    bool bSyncStructuredChatHistory = true;

    //run processing to emit e.g. sentence level breakups
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Params")
    bool bEmitPartials = true;

    //usually . ? !
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Params")
    TArray<FString> PartialsSeparators;
};

UENUM(BlueprintType)
enum class EChatTemplateRole : uint8
{
    User,
    Assistant,
    System,
    Unknown = 255
};

USTRUCT(BlueprintType)
struct FStructuredChatMessage
{
    GENERATED_USTRUCT_BODY();

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Structured Chat Message")
    EChatTemplateRole Role;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Structured Chat Message")
    FString Content;
};

USTRUCT(BlueprintType)
struct FStructuredChatHistory
{
    GENERATED_USTRUCT_BODY();
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Structured Chat History")
    TArray<FStructuredChatMessage> History;
};


//Easy user-specified chat template, or use common templates. Don't specify if you wish to load GGUF template.
USTRUCT(BlueprintType)
struct FChatTemplate
{
    GENERATED_USTRUCT_BODY();

    //Role: System
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Chat Template")
    FString System;

    //Role: User
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Chat Template")
    FString User;

    //Role: Assistant
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Chat Template")
    FString Assistant;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Chat Template")
    FString CommonSuffix;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Chat Template")
    FString Delimiter;

    FChatTemplate()
    {
        System = TEXT("");
        User = TEXT("");
        Assistant = TEXT("");
        CommonSuffix = TEXT("");
        Delimiter = TEXT("");
    }
    bool IsEmptyTemplate()
    {
        return (
            System == TEXT("") &&
            User == TEXT("") &&
            Assistant == TEXT("") &&
            CommonSuffix == TEXT("") && 
            Delimiter == TEXT(""));
    }
};

//Initial state fed into the model
USTRUCT(BlueprintType)
struct FLLMModelParams
{
    GENERATED_USTRUCT_BODY();

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Params")
    FString PathToModel = "./model.gguf";

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Params")
    FString Prompt = "You are a helpful assistant.";

    //If not different than default empty, no template will be applied
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Params")
    FChatTemplate ChatTemplate;

    //If set anything other than unknown, AI chat role will be enforced. Assistant is default
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Params")
    EChatTemplateRole ModelRole = EChatTemplateRole::Assistant;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Params")
    TArray<FString> StopSequences;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Params")
    int32 MaxContextLength = 2048;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Params")
    int32 GPULayers = 50;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Params")
    int32 Threads = 8;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Params")
    int32 BatchCount = 512;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Params")
    int32 Seed = -1;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Params")
    FLLMModelAdvancedParams Advanced;
};

//Current State
USTRUCT(BlueprintType)
struct FLLMModelState
{
    GENERATED_USTRUCT_BODY();

    //One true store that should be synced to the model internal history, accessible on game thread.
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model State")
    FString PromptHistory;

    //Where prompt history is raw, chat is an ordered structure. May not be relevant for non-chat type llm data
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model State")
    FStructuredChatHistory ChatHistory;

    //Optional split according to partials
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model State")
    TArray<FString> Partials;

    //Synced with current context length
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model State")
    int32 ContextLength = 0;

    //Stored the last speed reading on this model
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model State")
    float LastTokensPerSecond = 0.f;
};

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

    //Main callback
    UPROPERTY(BlueprintAssignable)
    FOnNewTokenGeneratedSignature OnNewTokenGenerated;

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

    UFUNCTION(BlueprintCallable)
    void InsertPrompt(const FString &Text);

    UFUNCTION(BlueprintPure)
    FString WrapPromptForRole(const FString& Text, EChatTemplateRole Role, bool AppendModelRolePrefix=false);

    UFUNCTION(BlueprintPure)
    FString GetModelRolePrefix();

    //This will wrap your input given the specific role using chat template specified
    UFUNCTION(BlueprintCallable)
    void InsertPromptTemplated(const FString& Text, EChatTemplateRole Role);


    UFUNCTION(BlueprintCallable)
    void StartStopQThread(bool bShouldRun = true);

    //Force stop generating new tokens
    UFUNCTION(BlueprintCallable)
    void StopGenerating();

    UFUNCTION(BlueprintCallable)
    void ResumeGenerating();

    UFUNCTION(BlueprintCallable)
    void SyncParamsToLlama();


    UFUNCTION(BlueprintPure)
    FString GetTemplateStrippedPrompt();

    FStructuredChatMessage FirstChatMessageInHistory(const FString& History, FString& Remainder);

    UFUNCTION(BlueprintPure)
    FStructuredChatHistory GetStructuredHistory();


    //String Utility
    bool IsSentenceEndingPunctuation(const TCHAR Char);
    FString GetLastSentence(const FString& InputString);


    //Utility function for debugging model location and file enumeration
    UFUNCTION(BlueprintCallable)
    TArray<FString> DebugListDirectoryContent(const FString& InPath);

private:
    std::unique_ptr<Internal::FLlama> llama;
};
