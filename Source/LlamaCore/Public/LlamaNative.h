// Copyright 2025-current Getnamo.

#pragma once

#include "LlamaDataTypes.h"
#include "CoreMinimal.h"
//#include "LlamaNative.generated.h"


/** 
* C++ native wrapper in Unreal styling for Llama.cpp with threading and callbacks. Embed in final place
* where it should be used e.g. ActorComponent, UObject, or Subsystem subclass.
*/
class LLAMACORE_API FLlamaNative
{
public:

	//Callbacks
	TFunction<void(const FString& Token)> OnTokenGenerated;
	TFunction<void(const FString& int64)> OnTaskCompletion;
	TFunction<void(const FString& Partial)> OnPartialGenerated;		//usually considered sentences, good for TTS.
	TFunction<void(const FString& Response)> OnResponseGenerated;	//per round
	TFunction<void(int32 TokensProcessed, EChatTemplateRole ForRole, float Speed)> OnPromptProcessed;	//when an inserted prompt has finished processing (non-generation prompt)
	TFunction<void()> OnGenerationStarted;
	TFunction<void(const FLlamaRunTimings& Timings)> OnGenerationFinished;
	TFunction<void(const FString& ErrorMessage)> OnError;
	TFunction<void(const FLLMModelState& UpdatedModelState)> OnModelStateChanged;

	//Expected to be set before load model
	void SetModelParams(const FLLMModelParams& Params);

	//Loads the model found at ModelParams.PathToModel, use SetModelParams to specify params before loading
	void LoadModel(TFunction<void(const FString&, int32 StatusCode)> ModelLoadedCallback = nullptr);
	void UnloadModel(TFunction<void(int32 StatusCode)> ModelUnloadedCallback = nullptr);
	bool IsModelLoaded();

	//Prompt input
	void InsertTemplatedPrompt(const FString& Prompt, EChatTemplateRole Role = EChatTemplateRole::User, bool bAddAssistantBOS = false, bool bGenerateReply = true);
	void InsertRawPrompt(const FString& Prompt);
	bool IsGenerating();
	void StopGeneration();
	void ResumeGeneration();

	//tick forward for safely consuming game thread messages without hanging
	void OnTick(float DeltaTime);

	//Context change - not yet implemented
	void ResetContextHistory(bool bKeepSystemPrompt = false);	//full reset
	void RemoveLastUserInput();		//chat rollback to undo last user input
	void RemoveLastReply();		//chat rollback to undo last assistant input.
	void RegenerateLastReply(); //removes last reply and regenerates (changing seed?)

	//Todo: Use this api + state checks to use RemoveLastInput and RemoveLastReply wrappers.
	void RemoveLastNMessages(int32 MessageCount);	//rollback

	//Pure query of current context - not threadsafe, be careful when these get called - TBD: make it safe
	int32 RawContextHistory(FString& OutContextString);
	void GetStructuredChatHistory(FStructuredChatHistory& OutChatHistory);
	void SyncPassedModelStateToNative(FLLMModelState& StateToSync);
	int32 UsedContextLength();

	FString WrapPromptForRole(const FString& Text, EChatTemplateRole Role, const FString& OverrideTemplate, bool bAddAssistantBoS = false);

	FLlamaNative();
	~FLlamaNative();

	float ThreadIdleSleepDuration = 0.005f; //5ms sleep timer for BG thread

protected:

	void SyncModelStateToInternal();

	//State
	FLLMModelParams ModelParams;
	FLLMModelState ModelState;
	FString CombinedPieceText;	//accumulates tokens into full string during per-token inference.

	//likely to be removed
	FThreadSafeBool bCallbacksAreValid = false;

	//Threading
	void StartLLMThread();
	TQueue<FLLMThreadTask> BackgroundTasks;
	TQueue<FLLMThreadTask> GameThreadTasks;
	FThreadSafeBool bThreadIsActive = false;
	FThreadSafeBool bThreadShouldRun = false;
	int64 TaskIdCounter = 0;
	int64 GetNextTaskId();

	void EnqueueBGTask(TFunction<void(int64)> Task);
	void EnqueueGTTask(TFunction<void()> Task, int64 LinkedTaskId = -1);

	class FLlamaInternal* Internal = nullptr;
};