// Copyright 2025-current Getnamo.

#pragma once

#include "LlamaDataTypes.h"
#include "CoreMinimal.h"


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
	TFunction<void(const FString& ErrorMessage, int32 ErrorCode)> OnError;
	TFunction<void(const FLLMModelState& UpdatedModelState)> OnModelStateChanged;

	//Expected to be set before load model
	void SetModelParams(const FLLMModelParams& Params);

	//Loads the model found at ModelParams.PathToModel, use SetModelParams to specify params before loading
	void LoadModel(bool bForceReload = false, TFunction<void(const FString&, int32 StatusCode)> ModelLoadedCallback = nullptr);
	void UnloadModel(TFunction<void(int32 StatusCode)> ModelUnloadedCallback = nullptr);
	bool IsModelLoaded();

	//Prompt input
	void InsertTemplatedPrompt(const FLlamaChatPrompt& Prompt, 
		TFunction<void(const FString& Response)>OnResponseFinished = nullptr);
	void InsertRawPrompt(const FString& Prompt, bool bGenerateReply = true, 
		TFunction<void(const FString& Response)>OnResponseFinished = nullptr);
	void ImpersonateTemplatedPrompt(const FLlamaChatPrompt& Prompt);
	void ImpersonateTemplatedToken(const FString& Token, EChatTemplateRole Role = EChatTemplateRole::Assistant, bool bEoS = false);
	bool IsGenerating();
	void StopGeneration();
	void ResumeGeneration();

	//if you've queued up a lot of BG tasks, you can clear the queue with this call
	void ClearPendingTasks(bool bClearGameThreadCallbacks = false);

	//tick forward for safely consuming game thread messages
	void OnGameThreadTick(float DeltaTime);
	void AddTicker();	 //optional call this once if you don't forward ticks from e.g. component/actor tick
	void RemoveTicker(); //if you use AddTicker, use remove ticker to balance on exit. Will happen on destruction of FLlamaNative if not called earlier.
	bool IsNativeTickerActive();

	//Context change - not yet implemented
	void ResetContextHistory(bool bKeepSystemPrompt = false);	//full reset
	void RemoveLastUserInput();		//chat rollback to undo last user input
	void RemoveLastReply();		//chat rollback to undo last assistant input.
	void RegenerateLastReply(); //removes last reply and regenerates (changing seed?)

	//Base api to do message rollback
	void RemoveLastNMessages(int32 MessageCount);	//rollback

	void RemoveLastNTokens(int32 TokensCount = 1);	//fine rollback

	//Pure query of current game thread context
	void SyncPassedModelStateToNative(FLLMModelState& StateToSync);

	FString WrapPromptForRole(const FString& Text, EChatTemplateRole Role, const FString& OverrideTemplate, bool bAddAssistantBoS = false);

	//Embedding mode

	//Embed a prompt and return the embeddings
	void GetPromptEmbeddings(const FString& Text, TFunction<void(const TArray<float>& Embeddings, const FString& SourceText)>OnEmbeddings = nullptr);

	FLlamaNative();
	~FLlamaNative();

	float ThreadIdleSleepDuration = 0.005f;        //default sleep timer for BG thread in sec.

protected:

	//can be safely called on game thread or the bg thread
	void SyncModelStateToInternal(TFunction<void()>AdditionalGTStateUpdates = nullptr);

	//utility functions, only safe to call on bg thread
	int32 RawContextHistory(FString& OutContextString);
	void GetStructuredChatHistory(FStructuredChatHistory& OutChatHistory);
	int32 UsedContextLength();

	//GT State - safely accesible on game thread
	FLLMModelParams ModelParams;
	FLLMModelState ModelState;
	bool bModelLoadInitiated = false; //tracking model load attempts

	//Temp states
	double ThenTimeStamp = 0.f;
	int32 ImpersonationTokenCount = 0;

	//BG State - do not read/write on GT
	FString CombinedPieceText;	//accumulates tokens into full string during per-token inference.
	FString CombinedTextOnPartialEmit; //state needed to check if on finish we've emitted all partials (broken grammar).

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