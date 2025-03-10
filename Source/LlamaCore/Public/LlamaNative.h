// Copyright 2025-current Getnamo.

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

	//Expected to be set before load model
	void SetModelParams(const FLLMModelParams& Params);

	//Loads the model found at ModelParams.PathToModel, use SetModelParams to specify params before loading
	bool LoadModel();

	bool UnloadModel();


	void InsertPrompt(const FString& Prompt);


	FLlamaNative();
	~FLlamaNative();
private:

	//Helper function
	FString Generate(const FString& Prompt);

	FLLMModelParams ModelParams;

	class FLlamaInternalState* Internal = nullptr;
};