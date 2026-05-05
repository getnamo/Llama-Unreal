// Copyright 2025-current Getnamo.

#pragma once

#include "CoreMinimal.h"
#include "LlamaComponent.h"
#include "Interfaces/IHttpRequest.h"
#include "LlamaMarkdownSplitter.h"
#include "Remote/LlamaRemoteTypes.h"
#include "Remote/LlamaRemoteClient.h"
#include "LlamaRemoteComponent.generated.h"

/**
 * Remote variant of ULlamaComponent — routes inference through an OpenAI-compatible HTTP endpoint
 * (e.g. llama-server) while preserving the same Blueprint-facing surface: history, delegates,
 * streaming tokens, partials, rollback, multimodal prompts. Drop-in replacement: point `Endpoint.BaseUrl`
 * at your server and everything else behaves like a local ULlamaComponent.
 *
 * Both the local FLlamaNative backend and the remote HTTP client are allocated. SetUseRemote(bool)
 * picks which one services the next inference call. When toggling, any active stream is cancelled
 * and the chat history is lazily synced to the new backend on the next prompt insertion (going
 * remote->local does a full KV rebuild via FLlamaNative::RebuildContextFromHistory).
 */
UCLASS(ClassGroup = (LLM), meta = (BlueprintSpawnableComponent))
class LLAMACORE_API ULlamaRemoteComponent : public ULlamaComponent
{
    GENERATED_BODY()
public:
    ULlamaRemoteComponent(const FObjectInitializer& ObjectInitializer);
    virtual ~ULlamaRemoteComponent();

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Remote")
    FLlamaRemoteEndpoint Endpoint;

    /** True (default) = next inference call hits the remote endpoint. False = uses the local FLlamaNative backend.
     *  Read-only in editor — mutate via SetUseRemote() so toggle side effects (stream cancel, slot erase, history sync) run. */
    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "LLM Remote")
    bool bUseRemote = true;

    /** Toggle which backend services the next inference call. Cancels any active stream on the outgoing backend,
     *  best-effort erases server slot if leaving remote, auto-loads the new backend if its config looks valid
     *  (Endpoint.BaseUrl for remote, ModelParams.PathToModel for local), and marks history sync pending so the
     *  next Insert* call rebuilds the destination's context from ModelState.ChatHistory. */
    UFUNCTION(BlueprintCallable, Category = "LLM Remote")
    void SetUseRemote(bool bNewUseRemote);

    UFUNCTION(BlueprintPure, Category = "LLM Remote")
    bool IsUsingRemote() const { return bUseRemote; }

    virtual void LoadModel(bool bForceReload = true) override;
    virtual void UnloadModel() override;
    virtual bool IsModelLoaded() override;

    virtual void InsertTemplatedPrompt(const FString& Text, EChatTemplateRole Role = EChatTemplateRole::User, bool bAddAssistantBOS = false, bool bGenerateReply = true) override;
    virtual void InsertTemplatedPromptStruct(const FLlamaChatPrompt& ChatPrompt) override;
    virtual void InsertRawPrompt(const FString& Text, bool bGenerateReply = true) override;

    virtual void StopGeneration() override;
    virtual void ResumeGeneration() override;
    virtual void ResetContextHistory(bool bKeepSystemPrompt = false) override;

    virtual void InsertTemplateImagePrompt(UTexture2D* Image, const FString& Text, EChatTemplateRole Role = EChatTemplateRole::User, bool bAddAssistantBOS = false, bool bGenerateReply = true) override;
    virtual void InsertTemplateImagePromptFromFile(const FString& ImagePath, const FString& Text, EChatTemplateRole Role = EChatTemplateRole::User, bool bAddAssistantBOS = false, bool bGenerateReply = true) override;
    virtual void InsertTemplateAudioPrompt(const TArray<float>& PCMAudio, const FString& Text, EChatTemplateRole Role = EChatTemplateRole::User, bool bAddAssistantBOS = false, bool bGenerateReply = true) override;
    virtual void InsertMultimodalPrompt(const FLlamaMultimodalPrompt& Prompt) override;

    virtual bool IsMultimodalLoaded() const override;
    virtual bool SupportsVision() const override;
    virtual bool SupportsAudio() const override;
    virtual int32 GetAudioSampleRate() const override;

protected:
    virtual void BeginDestroy() override;

    /** Kick off a streaming chat completion from the current ModelState.ChatHistory. */
    void BeginStreamFromHistory(bool bAttachPendingMedia);

    /** Append a message to local state and fire OnPromptProcessed like the native backend does. */
    void AppendUserMessage(const FString& Content, EChatTemplateRole Role);

    /** Called at the top of every Insert/StopGeneration override. If bPendingHistorySync is set
     *  and we're routing local, calls LlamaNative->RebuildContextFromHistory(ModelState.ChatHistory)
     *  so the local KV cache mirrors the conversation built up on the remote side. */
    void FlushPendingHistorySyncIfNeeded();

    /** Simple sentence-level partial emitter matching native behavior (uses ModelParams.Advanced.Output.PartialsSeparators). */
    void HandleIncomingDelta(const FString& Delta);
    void FlushPendingPartial(bool bForceEmitRemainder);

    static void EncodeTextureToPng(UTexture2D* Image, TArray<uint8>& OutPng, FString& OutMime);
    static bool LoadImageFileAsPng(const FString& Path, TArray<uint8>& OutPng, FString& OutMime);
    static void EncodePcmFloatToWav(const TArray<float>& Pcm, int32 SampleRate, TArray<uint8>& OutWav);

private:
    TUniquePtr<FLlamaRemoteClient> Client;
    FHttpRequestPtr ActiveStream;

    bool bRemoteModelLoaded = false;
    bool bRemoteVision = false;
    bool bRemoteAudio = false;
    /** Auto-detected from /props chat_template (mirrors FLlamaInternal::bModelSupportsThinking).
     *  When true and ModelParams.Advanced.Thinking.bStripThinkingFromResponse, the final response
     *  broadcast strips the <think>...</think> block. Streaming tokens/partials see the raw stream. */
    bool bRemoteSupportsThinking = false;
    int32 RemoteAudioSampleRate = 16000;
    int32 AssignedSlotId = -1;

    FString RemoteModelName;

    /** Media queued to attach to the next outgoing user message. */
    TArray<FLlamaRemoteMediaBlob> PendingUserMedia;

    /** Sentence-splitter buffer for OnPartialGenerated emulation. */
    FString PartialBuffer;

    /** Streaming markdown splitter — emits OnMarkdownPartialGenerated on separator hits, mirroring native behavior. */
    FLlamaMarkdownSplitter MdSplitter;

    /** Set true on SetUseRemote() when there is non-empty history; cleared after FlushPendingHistorySyncIfNeeded
     *  successfully syncs into the destination backend. */
    bool bPendingHistorySync = false;
};
