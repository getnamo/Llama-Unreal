// Copyright 2025-current Getnamo.

#pragma once

#include "LlamaDataTypes.generated.h"

UENUM(BlueprintType)
enum class EChatTemplateRole : uint8
{
    User,
    Assistant,
    System,
    Unknown = 255
};

DECLARE_DYNAMIC_MULTICAST_DELEGATE_TwoParams(FOnErrorSignature, const FString&, ErrorMessage, int32, ErrorCode);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnTokenGeneratedSignature, const FString&, Token);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnResponseGeneratedSignature, const FString&, Response);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FModelNameSignature, const FString&, ModelName);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnPartialSignature, const FString&, Partial);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnPromptHistorySignature, FString, History);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_TwoParams(FOnEndOfStreamSignature, bool, bStopSequenceTriggered, float, TokensPerSecond);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_ThreeParams(FOnPromptProcessedSignature, int32, TokensProcessed, EChatTemplateRole, Role, float, TokensPerSecond);
DECLARE_DYNAMIC_MULTICAST_DELEGATE(FVoidEventSignature);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_TwoParams(FOnEmbeddingsSignature, const TArray<float>&, Embeddings, const FString&, SourceText);

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

    //Updates the logits l_i` = l_i/t. When t <= 0.0f, the maximum logit is kept at it's original value, the rest are set to -inf
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Advanced Params - Sampling")
    float Temp = 0.80f;

    //Minimum P sampling as described in https://github.com/ggml-org/llama.cpp/pull/3841. if non -1 it will apply, typically good value ~0.05f
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Advanced Params - Sampling")
    float MinP = 0.05f;

    //Top-K sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751. if non -1 it will apply, typically good value ~40
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Advanced Params - Sampling")
    int32 TopK = -1;

    //Nucleus sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751. if non -1 it will apply, typically good value ~0.95f
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Advanced Params - Sampling")
    float TopP = -1.f;

    //Locally Typical Sampling implementation described in the paper https://arxiv.org/abs/2202.00666. If non -1 it will apply, typically good value 1.f
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Advanced Params - Sampling")
    float TypicalP = -1.f;

    //Repetition Penalty; avoid using on the full vocabulary as searching for repeated tokens can become slow, consider using Top-k and top-p smapling first. 0 is off, -1 is context
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Advanced Params - Penalties")
    int32 PenaltyLastN = 0;

    //Repetition Penalty. 1 is disabled
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Advanced Params - Penalties")
    float PenaltyRepeat = 1.f;

    //Repetition Penalty - frequency based. 0 is disabled
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Advanced Params - Penalties")
    float PenaltyFrequency = 0.f;

    //Repetition Penalty - presence based. 0 is disabled
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Advanced Params - Penalties")
    float PenaltyPresence = 0.f;

    //Mirostat 2.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. 
    //If Mirostat != -1 then it will apply this seed value using mirostat v2 algorithm
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Advanced Params - Mirostat")
    int32 Mirostat = -1;

    //MirostatSeed -1 disables this
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Advanced Params - Mirostat")
    float MirostatTau = 5.f;

    //MirostatSeed -1 disables this
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Advanced Params - Mirostat")
    float MirostatEta = 0.1f;

    //synced per eos
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Params")
    bool bSyncStructuredChatHistory = true;

    //run processing to emit e.g. sentence level breakups
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Params")
    bool bEmitPartials = true;

    //Process callbacks on gamethread - NB: always emits on game thread for now, option doesn't do anything.
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Params")
    bool bEmitOnGameThread = true;

    //temporarily defaulted on during dev
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Params")
    bool bLogGenerationStats = true;

    //if true sampling params won't be passed (v0.8)
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Params")
    bool bUseCommonSampler = true;

    //use common_init instead of normal - may break functionality, use with care
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Params")
    bool bUseCommonParams = false;

    //set to true if you want to use GeneratePromptEmbeddingsForText
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Params")
    bool bEmbeddingMode = false;

    //if set above 0.f it will sleep between generation passes to ease gpu pressure
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Params")
    float TokenGenerationPacingSleep = 0.f;

    //if set above 0.f it will sleep between prompt passes (chunking) to ease gpu pressure
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Params")
    float PromptProcessingPacingSleep = 0.f;

    //this part is only active if PromptProcessingPacingSleep > 0.f. Splits prompts into n chunks with sleep
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Params")
    int32 PromptProcessingPacingSplitN = 4;

    //usually . ? !
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Params")
    TArray<FString> PartialsSeparators;
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


//Todo: refactor to jinja style string
// 
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


USTRUCT(BlueprintType)
struct FJinjaChatTemplate
{
    GENERATED_USTRUCT_BODY();

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Jinja Chat Template")
    FString TemplateSource = TEXT("");

    UPROPERTY(EditAnywhere, BlueprintReadWrite, meta = (MultiLine = true), Category = "Jinja Chat Template")
    FString Jinja = TEXT("");
};

//Initial state fed into the model
USTRUCT(BlueprintType)
struct FLLMModelParams
{
    GENERATED_USTRUCT_BODY();

    //If path begins with a . it's considered relative to Saved/Models path, otherwise it's an absolute path.
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Params")
    FString PathToModel = "./model.gguf";

    //Gets embedded on first input after a model load
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Params", meta=(MultiLine=true))
    FString SystemPrompt = "You are a helpful assistant.";

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Params")
    bool bAutoInsertSystemPromptOnLoad = true;

    //applies to component API
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Params")
    bool bAutoLoadModelOnStartup = true;

    //If true, all prompt inserts/rollbacks only modify modelstate and do not forward to llama component (see impersonation)
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Params")
    bool bRemoteMode = false;

    //If not different than default empty, no template will be applied
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Params")
    FJinjaChatTemplate CustomChatTemplate = "";

    //If set anything other than unknown, AI chat role will be enforced. Assistant is default
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Params")
    EChatTemplateRole ModelRole = EChatTemplateRole::Assistant;

    //Additional stop sequences - not currently active
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Params")
    TArray<FString> StopSequences;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Params")
    int32 MaxContextLength = 4096;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Params")
    int32 GPULayers = 50;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Params")
    int32 Threads = 8;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Params")
    int32 MaxBatchLength = 1024;

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

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model State")
    bool bModelIsLoaded = false;

    //The raw context history with formatting applied
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model State")
    FString ContextHistory;

    //Where prompt history is raw, chat is an ordered structure. May not be relevant for non-chat type llm data
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model State")
    FStructuredChatHistory ChatHistory;

    //Optional split according to partials
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model State")
    TArray<FString> Partials;

    //Synced with current context length
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model State")
    int32 ContextUsed = 0;

    //Updates after each eos1
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model State")
    float LastTokenGenerationSpeed = 0.f;

    //Updates after each prompt processing
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model State")
    float LastPromptProcessingSpeed = 0.f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model State")
    EChatTemplateRole LastRole = EChatTemplateRole::Unknown;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model State")
    FJinjaChatTemplate ChatTemplateInUse;
};

USTRUCT()
struct FLLMThreadTask
{
    GENERATED_USTRUCT_BODY();

    TFunction<void(int64)> TaskFunction;

    UPROPERTY()
    int64 TaskId = 0;
};


USTRUCT(BlueprintType)
struct FLlamaChatPrompt
{
    GENERATED_BODY()

public:
    /** The prompt string */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Chat")
    FString Prompt;

    /** The role of the chat message */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Chat")
    EChatTemplateRole Role = EChatTemplateRole::User;

    /** Whether to add Assistant Beginning-of-Stream token */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Chat")
    bool bAddAssistantBOS = false;

    /** Whether to generate a reply */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Chat")
    bool bGenerateReply = true;

    FLlamaChatPrompt() {}

    FLlamaChatPrompt(const FString& InPrompt, EChatTemplateRole InRole = EChatTemplateRole::User, bool bInAddAssistantBOS = false, bool bInGenerateReply = true)
        : Prompt(InPrompt)
        , Role(InRole)
        , bAddAssistantBOS(bInAddAssistantBOS)
        , bGenerateReply(bInGenerateReply)
    {
    }
};