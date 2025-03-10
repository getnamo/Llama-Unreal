
#pragma once

#include "LlamaDataTypes.h"
#include "CoreMinimal.h"
//#include "LlamaNative.generated.h"

class LLAMACORE_API FLlamaNative
{
public:

	TFunction<void(const FString& Token)> OnTokenGenerated;
	TFunction<void()> OnGenerationStarted;
	TFunction<void(const FLlamaRunTimings& Timings)> OnGenerationFinished;
	TFunction<void(const FString& ErrorMessage)> OnError;

	void SetModelParams(const FLLMModelParams& Params);

	//Loads the model found at ModelParams.PathToModel
	bool LoadModel();


	FLlamaNative();
	~FLlamaNative();
private:


	FLLMModelParams ModelParams;
};