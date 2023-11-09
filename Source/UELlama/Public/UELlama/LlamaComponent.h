// 2023 (c) Mika Pi

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
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnEndOfStreamSignature, bool, bStopSequenceTriggered);
DECLARE_DYNAMIC_MULTICAST_DELEGATE(FVoidEventSignature);

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


    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    FString Prompt = "Hello";

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    FString PathToModel = "/media/mika/Michigan/prj/llama-2-13b-chat.ggmlv3.q8_0.bin";

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    TArray<FString> StopSequences;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    bool bDebugLogModelOutput = false;

    UFUNCTION(BlueprintCallable)
    void InsertPrompt(const FString &Text);

    UFUNCTION(BlueprintCallable)
    void StartStopQThread(bool bShouldRun = true);

    //Force stop generating new tokens
    UFUNCTION(BlueprintCallable)
    void StopGenerating();

    UFUNCTION(BlueprintCallable)
    void ResumeGenerating();

    //One true store that should be synced to the model internal history, accessible on game thread.
    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    FString PromptHistory;

    //toggle to pay copy cost or not, default true
    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    bool bSyncPromptHistory = true;

private:
    std::unique_ptr<Internal::Llama> llama;
};
