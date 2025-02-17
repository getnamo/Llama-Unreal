#pragma once
#include "EChatTemplateRole.h"
#include "FChatTemplate.h"
#include "FLLMModelParams.generated.h"


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
