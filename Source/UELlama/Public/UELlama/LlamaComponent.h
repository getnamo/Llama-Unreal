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
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnEndOfStreamSignature, bool, bStopSequenceTriggered);

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

    //Whenever the model stops generating
    UPROPERTY(BlueprintAssignable)
    FOnEndOfStreamSignature OnEndOfStream;

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

    //Force stop generating new tokens
    UFUNCTION(BlueprintCallable)
    void StopGenerating();

private:
    std::unique_ptr<Internal::Llama> llama;
};
