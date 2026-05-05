// Copyright 2025-current Getnamo.

#pragma once

#include "CoreMinimal.h"
#include "LlamaDataTypes.h"
#include "LlamaMarkdownSplitter.h"
#include "LlamaMediaCaptureTypes.h"
#include "Remote/LlamaRemoteTypes.h"
#include "Interfaces/IHttpRequest.h"

class FLlamaNative;
class FLlamaRemoteClient;
class UTexture2D;

/**
 * Non-UObject orchestrator that owns a local FLlamaNative and a remote HTTP client and
 * routes inference between them based on bUseRemote. Embedded by ULlamaComponent and
 * ULlamaSubsystem so they can share a single, exhaustively-tested implementation of the
 * dual-backend state machine (history sync on toggle, slot management, streaming
 * partial / markdown emulation, multimodal blob queue).
 *
 * Callbacks are TFunction-typed; each host (component / subsystem) wires them to its
 * Blueprint multicast delegates in its constructor / Initialize().
 *
 * Threading: matches FLlamaNative — model methods are GT-safe; native callbacks fire on GT
 * via the FLlamaNative ticker; HTTP callbacks fire on GT via UE's HTTP module.
 */
class LLAMACORE_API FLlamaDualBackend : public ILlamaAudioConsumer
{
public:
    // --- Configuration / state (GT-only mutators unless noted) -------------

    FLLMModelParams ModelParams;
    FLLMModelState  ModelState;
    FLlamaRemoteEndpoint Endpoint;

    /** When true, inference is routed through the remote HTTP endpoint; when false, through the local FLlamaNative. */
    bool bUseRemote = false;

    /** Audio prompt template applied when an FLlamaAudioSegment arrives via OnAudioSegment.
     *  Mirrored down into FLlamaNative when local; used to format the user message when remote. */
    FString AudioPromptTemplate = TEXT("<__media__>\nRespond to what was said.");
    EChatTemplateRole AudioPromptRole = EChatTemplateRole::User;

    /** When true (default), a remote→local toggle whose chat-history prefix matches the local
     *  KV cache's last-synced prefix appends only the *new* messages instead of wiping and
     *  replaying the entire history. Hash-verified — falls back to a full rebuild whenever the
     *  prefix has diverged. Flip to false to force the slower-but-bulletproof rebuild path on
     *  every sync (useful if the smart path ever surfaces a KV-state bug). */
    bool bUseIncrementalKVSyncOnToggle = true;

    // --- Callback hooks (host wires to Blueprint multicasts) ---------------

    TFunction<void(const FString& Token)>                    OnTokenGenerated;
    TFunction<void(const FString& Partial)>                  OnPartialGenerated;
    TFunction<void(const FString& Partial, EMarkdownStreamState State)> OnMarkdownPartialGenerated;
    TFunction<void(const FString& Response)>                 OnResponseGenerated;
    TFunction<void(int32 Tokens, EChatTemplateRole, float Speed)> OnPromptProcessed;
    TFunction<void(bool bStopSeq, float Tps)>                OnEndOfStream;
    TFunction<void()>                                        OnContextReset;
    TFunction<void(const FString& ModelName)>                OnModelLoaded;
    TFunction<void(const FString& Err, int32 Code)>          OnError;
    TFunction<void(const TArray<float>&, const FString&)>    OnEmbeddings;
    TFunction<void(const TArray<FString>&)>                  OnAllEmbeddingsGenerated;
    TFunction<void(const FLLMModelState&)>                   OnModelStateChanged;

    // --- Lifecycle ----------------------------------------------------------

    FLlamaDualBackend();
    virtual ~FLlamaDualBackend();

    /** Build local FLlamaNative + remote client. Call once after constructing. */
    void Initialize();

    /** Tear down both backends. Cancels any active stream. */
    void Shutdown();

    /** Forward GT tick into FLlamaNative so its native callbacks drain on GT. */
    void OnGameThreadTick(float DeltaTime);

    /** Returns the underlying FLlamaNative for advanced consumers (audio source registration, etc.).
     *  Null when Initialize() hasn't run. */
    FLlamaNative* GetLlamaNative() const { return LlamaNative; }

    // --- Backend toggle -----------------------------------------------------

    /** Switches between local and remote routing. Cancels in-flight inference on the outgoing
     *  backend, best-effort erases server slot when leaving remote, lazily rebuilds destination
     *  KV cache from ModelState.ChatHistory on the next Insert*. */
    void SetUseRemote(bool bNew);
    bool IsUsingRemote() const { return bUseRemote; }

    // --- Model load / unload -----------------------------------------------

    void LoadModel(bool bForceReload);
    void UnloadModel();
    bool IsModelLoaded() const;

    // --- Chat / inference ---------------------------------------------------

    void InsertTemplatedPrompt(const FLlamaChatPrompt& Prompt);
    void InsertRawPrompt(const FString& Text, bool bGenerateReply);
    void StopGeneration();
    void ResumeGeneration();
    void ResetContextHistory(bool bKeepSystemPrompt);
    void RebuildContextFromHistory(const FStructuredChatHistory& History);

    void RemoveLastReply();
    void RemoveLastUserInput();
    void RemoveLastNTokens(int32 N);

    void ImpersonateTemplatedPrompt(const FLlamaChatPrompt& Prompt);
    void ImpersonateTemplatedToken(const FString& Token, EChatTemplateRole Role, bool bEoS);
    FString WrapPromptForRole(const FString& Text, EChatTemplateRole Role, const FString& OverrideTemplate);

    // --- Multimodal --------------------------------------------------------

    void InsertTemplateImagePromptFromTexture(UTexture2D* Image, const FString& Text,
        EChatTemplateRole Role, bool bAddAssistantBOS, bool bGenerateReply);
    void InsertTemplateImagePromptFromFile(const FString& ImagePath, const FString& Text,
        EChatTemplateRole Role, bool bAddAssistantBOS, bool bGenerateReply);
    void InsertTemplateAudioPrompt(const TArray<float>& PCMAudio, const FString& Text,
        EChatTemplateRole Role, bool bAddAssistantBOS, bool bGenerateReply);
    void InsertMultimodalPrompt(const FLlamaMultimodalPrompt& Prompt);

    bool IsMultimodalLoaded() const;
    bool SupportsVision() const;
    bool SupportsAudio() const;
    int32 GetAudioSampleRate() const;

    // --- Embedding ---------------------------------------------------------

    void GeneratePromptEmbeddingsForText(const FString& Text);
    void GeneratePromptEmbeddingsForTexts(const TArray<FString>& Texts);
    int32 GetEmbeddingDimension() const;

    /** Exclusive-callback variant for tools (URagStore etc.) that need a private embedding round-trip. */
    void EmbedTextsAsync(const TArray<FString>& Texts,
        TFunction<void(const TArray<TArray<float>>&, const TArray<FString>&)> OnDone);

    // --- ILlamaAudioConsumer (route audio segments through whichever backend is active) ----

    virtual void OnAudioSegment(const FLlamaAudioSegment& Segment) override;

    // --- KV-bypass predicate (used by Component / Subsystem for rollback decisions) ----

    bool ShouldBypassNativeKV() const { return bUseRemote || !LlamaNative || ModelParams.bImpersonationMode; }

    /** Hash of the [0..Count) prefix of `History` (role + content per message). Public for tests. */
    static uint32 ComputePrefixHash(const TArray<FStructuredChatMessage>& History, int32 Count);

private:
    // Local backend
    FLlamaNative* LlamaNative = nullptr;

    // Remote backend
    TUniquePtr<FLlamaRemoteClient> Client;
    FHttpRequestPtr ActiveStream;

    // Remote-mode mirrors
    bool bRemoteModelLoaded = false;
    bool bRemoteVision = false;
    bool bRemoteAudio = false;
    bool bRemoteSupportsThinking = false;
    int32 RemoteAudioSampleRate = 16000;
    int32 AssignedSlotId = -1;
    FString RemoteModelName;

    /** Media queued to attach to the next outgoing remote user message. */
    TArray<FLlamaRemoteMediaBlob> PendingUserMedia;

    /** Sentence-splitter buffer for OnPartialGenerated emulation over HTTP deltas. */
    FString PartialBuffer;

    /** Streaming markdown splitter — emits OnMarkdownPartialGenerated to mirror native cadence. */
    FLlamaMarkdownSplitter MdSplitter;

    /** Set true on SetUseRemote when there is non-empty history; cleared after a successful sync. */
    bool bPendingHistorySync = false;

    /** Number of messages currently reflected in the LOCAL FLlamaNative KV cache. The local
     *  prefix [0..LocalKVMessageCount) is what the model has decoded so far. Updated automatically
     *  via OnModelStateChanged after every native action; explicitly invalidated in bypass-path
     *  mutations (state-only RemoveLast*, RebuildContextFromHistory while remote, etc). */
    int32 LocalKVMessageCount = 0;

    /** Hash of the [0..LocalKVMessageCount) prefix at the moment of the last local sync. On
     *  history-flush we recompute the prefix hash and compare — if it matches we can append
     *  only the new messages; if it doesn't (someone edited the prefix while remote) we full
     *  rebuild. Cheap insurance against silent KV/state divergence. */
    uint32 LocalKVPrefixHash = 0;


    // --- Internal helpers ---------------------------------------------------

    void HookNativeCallbacks();

    void BeginStreamFromHistory(bool bAttachPendingMedia);
    void AppendUserMessage(const FString& Content, EChatTemplateRole Role);
    void FlushPendingHistorySyncIfNeeded();
    void HandleIncomingDelta(const FString& Delta);
    void FlushPendingPartial(bool bForceEmitRemainder);

    static void EncodeTextureToPng(UTexture2D* Image, TArray<uint8>& OutPng, FString& OutMime);
    static bool LoadImageFileAsPng(const FString& Path, TArray<uint8>& OutPng, FString& OutMime);
    static void EncodePcmFloatToWav(const TArray<float>& Pcm, int32 SampleRate, TArray<uint8>& OutWav);

    // Disallow copy — owns raw FLlamaNative + HTTP request handle.
    FLlamaDualBackend(const FLlamaDualBackend&) = delete;
    FLlamaDualBackend& operator=(const FLlamaDualBackend&) = delete;
};
