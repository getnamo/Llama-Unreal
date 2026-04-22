// Copyright 2025-current Getnamo.

#pragma once

#include "CoreMinimal.h"
#include "LlamaComponent.h"
#include "Interfaces/IHttpRequest.h"
#include "Remote/LlamaRemoteTypes.h"
#include "Remote/LlamaRemoteClient.h"
#include "LlamaRemoteComponent.generated.h"

/**
 * Remote variant of ULlamaComponent — routes inference through an OpenAI-compatible HTTP endpoint
 * (e.g. llama-server) while preserving the same Blueprint-facing surface: history, delegates,
 * streaming tokens, partials, rollback, multimodal prompts. Drop-in replacement: point `Endpoint.BaseUrl`
 * at your server and everything else behaves like a local ULlamaComponent.
 *
 * The local FLlamaNative backend is opted out (CreateNativeBackend returns nullptr), and
 * ModelParams.bRemoteMode is forced true so inherited rollback helpers manipulate state only.
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
    virtual class FLlamaNative* CreateNativeBackend() override { return nullptr; }
    virtual void BeginDestroy() override;

    /** Kick off a streaming chat completion from the current ModelState.ChatHistory. */
    void BeginStreamFromHistory(bool bAttachPendingMedia);

    /** Append a message to local state and fire OnPromptProcessed like the native backend does. */
    void AppendUserMessage(const FString& Content, EChatTemplateRole Role);

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
    int32 RemoteAudioSampleRate = 16000;
    int32 AssignedSlotId = -1;

    FString RemoteModelName;

    /** Media queued to attach to the next outgoing user message. */
    TArray<FLlamaRemoteMediaBlob> PendingUserMedia;

    /** Sentence-splitter buffer for OnPartialGenerated emulation. */
    FString PartialBuffer;
};
