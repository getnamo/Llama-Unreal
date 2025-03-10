// Copyright 2025-current Getnamo.

#pragma once

#include "LlamaDataTypes.generated.h"

DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnErrorSignature, FString, ErrorMessage);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnNewTokenGeneratedSignature, FString, NewToken);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnPartialSignature, const FString&, Partial);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnPromptHistorySignature, FString, History);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_TwoParams(FOnEndOfStreamSignature, bool, bStopSequenceTriggered, float, TokensPerSecond);
DECLARE_DYNAMIC_MULTICAST_DELEGATE(FVoidEventSignature);

USTRUCT(BlueprintType)
struct FLlamaRunTimings
{
    GENERATED_USTRUCT_BODY();

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Advanced Params")
    float SampleTime = 0.f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Advanced Params")
    float PromptEvalTime = 0.f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Advanced Params")
    float EvalTime = 0.f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Advanced Params")
    float TotalTime = 0.f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Advanced Params")
    float TokensPerSecond = 0.f;
};


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
    EChatTemplateRole Role = EChatTemplateRole::Assistant;

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

    //If path begins with a . it's considered relative to Saved/Models path, otherwise it's an absolute path.
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Params")
    FString PathToModel = "./model.gguf";

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Params", meta=(MultiLine=true))
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

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model State")
    EChatTemplateRole LastRole = EChatTemplateRole::Unknown;
};