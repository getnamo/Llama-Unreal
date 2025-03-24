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
	void InsertTemplatedPrompt(const FLlamaChatPrompt& Prompt, 
		TFunction<void(const FString& Response)>OnResponseFinished = nullptr);
	void InsertRawPrompt(const FString& Prompt, bool bGenerateReply = true, 
		TFunction<void(const FString& Response)>OnResponseFinished = nullptr);
	bool IsGenerating();
	void StopGeneration();
	void ResumeGeneration();

	//if you've queued up a lot of BG tasks, you can clear the queue with this call
	void ClearPendingTasks(bool bClearGameThreadCallbacks = false);

	//tick forward for safely consuming game thread messages without hanging
	void OnGameThreadTick(float DeltaTime);
	void AddTicker();	 //optional if you don't forward ticks from e.g. component tick
	void RemoveTicker(); //if you use AddTicker, use remove ticker to balance on exit. Will happen on destruction of native component.
	bool IsNativeTickerActive();

	//Context change - not yet implemented
	void ResetContextHistory(bool bKeepSystemPrompt = false);	//full reset
	void RemoveLastUserInput();		//chat rollback to undo last user input
	void RemoveLastReply();		//chat rollback to undo last assistant input.
	void RegenerateLastReply(); //removes last reply and regenerates (changing seed?)

	//Base api to do message rollback
	void RemoveLastNMessages(int32 MessageCount);	//rollback

	//Pure query of current game thread context
	void SyncPassedModelStateToNative(FLLMModelState& StateToSync);

	FString WrapPromptForRole(const FString& Text, EChatTemplateRole Role, const FString& OverrideTemplate, bool bAddAssistantBoS = false);

	FLlamaNative();
	~FLlamaNative();

	float ThreadIdleSleepDuration = 0.005f;        //5ms sleep timer for BG thread'

protected:

	//can be safely called on game thread or the bg thread, handles either logic
	void SyncModelStateToInternal(TFunction<void()>AdditionalGTStateUpdates = nullptr);

	//utility functions, only safe to call on bg thread
	int32 RawContextHistory(FString& OutContextString);
	void GetStructuredChatHistory(FStructuredChatHistory& OutChatHistory);
	int32 UsedContextLength();

	//GT State - safely accesible on game thread
	FLLMModelParams ModelParams;
	FLLMModelState ModelState;

	//BG State
	FString CombinedPieceText;	//accumulates tokens into full string during per-token inference.

	//Threading
	void StartLLMThread();
	TQueue<FLLMThreadTask> BackgroundTasks;
	TQueue<FLLMThreadTask> GameThreadTasks;
	FThreadSafeBool bThreadIsActive = false;
	FThreadSafeBool bThreadShouldRun = false;
	FThreadSafeCounter TaskIdCounter = 0;
	int64 GetNextTaskId();

	void EnqueueBGTask(TFunction<void(int64)> Task);
	void EnqueueGTTask(TFunction<void()> Task, int64 LinkedTaskId = -1);

	class FLlamaInternal* Internal = nullptr;
	FTSTicker::FDelegateHandle TickDelegateHandle = nullptr; //optional tick handle - used in subsystem example where tick isn't natively supported
};