// 2023 (c) Mika Pi, Modifications Getnamo

#pragma once
#include <Components/ActorComponent.h>
#include <CoreMinimal.h>
#include <memory>

#include "LlamaComponent.generated.h"

namespace Internal
{
  class Llama;
}

DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnNewTokenGeneratedSignature, FString, NewToken);
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
    bool PenalizeNl = true;
};

//Initial state fed into the model
USTRUCT(BlueprintType)
struct FLLMModelParams
{
    GENERATED_USTRUCT_BODY();

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Params")
    FString PathToModel = "/model.gguf";

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Params")
    FString Prompt = "You are a helpful assistant.";

    //Currently unsupported - should add support for this and filter results
    //UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Params")
    //FString PromptTemplate = "ChatML";

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Params")
    TArray<FString> StopSequences;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Params")
    int32 MaxContextLength = 2048;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Params")
    int32 GPULayers = 50;

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

    //Synced with current context length
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model State")
    int32 ContextLength;

    //Stored the last speed reading on this model
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model State")
    float LastTokensPerSecond;
};

UCLASS(Category = "LLM", BlueprintType, meta = (BlueprintSpawnableComponent))
class UELLAMA_API ULlamaComponent : public UActorComponent
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

    UPROPERTY(BlueprintAssignable)
    FVoidEventSignature OnStartEval;

    //Whenever the model stops generating
    UPROPERTY(BlueprintAssignable)
    FOnEndOfStreamSignature OnEndOfStream;

    UPROPERTY(BlueprintAssignable)
    FVoidEventSignature OnContextReset;

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

    UFUNCTION(BlueprintCallable)
    void InsertPrompt(const FString &Text);

    UFUNCTION(BlueprintCallable)
    void StartStopQThread(bool bShouldRun = true);

    //Force stop generating new tokens
    UFUNCTION(BlueprintCallable)
    void StopGenerating();

    UFUNCTION(BlueprintCallable)
    void ResumeGenerating();

private:
    std::unique_ptr<Internal::Llama> llama;
};
