// Copyright 2025-current Getnamo.

#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "Engine/Texture2D.h"
#include "LlamaDataTypes.h"
#include "Remote/LlamaRemoteTypes.h"

class ULlamaAudioCaptureComponent;
class FLlamaDualBackend;

#include "LlamaComponent.generated.h"

/**
 * Unified actor-component LLM API. Drives a local FLlamaNative backend by default; flip
 * `bUseRemote` (or call SetUseRemote(true)) to route the same Blueprint surface through an
 * OpenAI-compatible HTTP endpoint (e.g. llama-server, LM Studio, Ollama, vLLM, OpenAI).
 *
 * The local and remote backends share state (chat history, model params, multimodal blobs)
 * via FLlamaDualBackend, so toggling at runtime preserves the conversation — going local
 * ↔ remote rebuilds the destination's KV cache (or chat-message replay) lazily on the next
 * Insert*. Same delegates fire on both paths: OnTokenGenerated, OnPartialGenerated,
 * OnMarkdownPartialGenerated, OnResponseGenerated, OnEmbeddings, etc.
 *
 * Lifetime is the actor's. For a level-spanning singleton, use ULlamaSubsystem instead —
 * it exposes the same Blueprint surface backed by the same FLlamaDualBackend.
 */
UCLASS(Category = "LLM", BlueprintType, meta = (BlueprintSpawnableComponent))
class LLAMACORE_API ULlamaComponent : public UActorComponent
{
    GENERATED_BODY()
public:
    ULlamaComponent(const FObjectInitializer& ObjectInitializer);
    virtual ~ULlamaComponent();

    virtual void Activate(bool bReset) override;
    virtual void Deactivate() override;
    virtual void TickComponent(float DeltaTime, ELevelTick TickType,
                               FActorComponentTickFunction* ThisTickFunction) override;

    // ── Streaming + lifecycle delegates ──────────────────────────────────────

    UPROPERTY(BlueprintAssignable)
    FOnTokenGeneratedSignature OnTokenGenerated;

    UPROPERTY(BlueprintAssignable)
    FOnResponseGeneratedSignature OnResponseGenerated;

    UPROPERTY(BlueprintAssignable)
    FOnPartialSignature OnPartialGenerated;

    UPROPERTY(BlueprintAssignable)
    FOnMarkdownPartialSignature OnMarkdownPartialGenerated;

    UPROPERTY(BlueprintAssignable)
    FOnPromptProcessedSignature OnPromptProcessed;

    UPROPERTY(BlueprintAssignable)
    FOnEmbeddingsSignature OnEmbeddings;

    UPROPERTY(BlueprintAssignable)
    FOnEmbeddingsBatchSignature OnAllEmbeddingsGenerated;

    UPROPERTY(BlueprintAssignable)
    FOnEndOfStreamSignature OnEndOfStream;

    UPROPERTY(BlueprintAssignable)
    FVoidEventSignature OnContextReset;

    UPROPERTY(BlueprintAssignable)
    FModelNameSignature OnModelLoaded;

    UPROPERTY(BlueprintAssignable)
    FOnErrorSignature OnError;

    // ── Shared params / state ────────────────────────────────────────────────

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Component")
    FLLMModelParams ModelParams;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Component")
    FLLMModelState ModelState;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Component")
    bool bDebugLogModelOutput = false;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Component")
    bool bSyncPromptHistory = true;

    /** When true (default), a remote→local toggle whose chat-history prefix matches the local
     *  KV cache appends only the new messages instead of full-replaying the entire history.
     *  Hash-verified — falls back to full rebuild on prefix divergence. Flip to false to force
     *  full rebuild on every sync (slower but bulletproof). */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Component")
    bool bUseIncrementalKVSyncOnToggle = true;

    /** When true AND bUseRemote is on at LoadModel time, also silently warm-load the local
     *  backend so SetUseRemote(false) doesn't pay the multi-second model-load cost later.
     *  Default false (don't burn VRAM unless you specifically expect to swap). Only used in
     *  remote-active configurations — local-active always loads the local model as normal. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Remote",
              meta = (EditCondition = "bUseRemote", EditConditionHides))
    bool bPreloadLocalWhenRemote = false;

    // ── Remote routing (defaults to false; local-first) ──────────────────────

    /** When true, inference is routed through the remote HTTP endpoint defined in `Endpoint`.
     *  Read-only in editor — flip via SetUseRemote() so toggle side effects (stream cancel,
     *  slot erase, history sync) run. */
    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "LLM Remote")
    bool bUseRemote = false;

    /** OpenAI-compatible HTTP endpoint config. Used only when bUseRemote is true. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Remote",
              meta = (EditCondition = "bUseRemote", EditConditionHides))
    FLlamaRemoteEndpoint Endpoint;

    // ── Loading ──────────────────────────────────────────────────────────────

    UFUNCTION(BlueprintCallable, Category = "LLM Model Component")
    void LoadModel(bool bForceReload = true);

    UFUNCTION(BlueprintCallable, Category = "LLM Model Component")
    void UnloadModel();

    UFUNCTION(BlueprintPure, Category = "LLM Model Component")
    bool IsModelLoaded() const;

    /** Toggle local↔remote at runtime. See FLlamaDualBackend::SetUseRemote for semantics. */
    UFUNCTION(BlueprintCallable, Category = "LLM Remote")
    void SetUseRemote(bool bNewUseRemote);

    UFUNCTION(BlueprintPure, Category = "LLM Remote")
    bool IsUsingRemote() const { return bUseRemote; }

    // ── Chat / inference ─────────────────────────────────────────────────────

    UFUNCTION(BlueprintCallable, Category = "LLM Model Component")
    void ResetContextHistory(bool bKeepSystemPrompt = false);

    UFUNCTION(BlueprintCallable, Category = "LLM Model Component")
    void RebuildContextFromHistory(const FStructuredChatHistory& History);

    UFUNCTION(BlueprintCallable, Category = "LLM Model Component")
    void RemoveLastAssistantReply();

    UFUNCTION(BlueprintCallable, Category = "LLM Model Component")
    void RemoveLastUserInput();

    UFUNCTION(BlueprintCallable, Category = "LLM Model Component")
    void RemoveLastNTokens(int32 TokenCount = 1);

    UFUNCTION(BlueprintCallable, Category = "LLM Model Component")
    void InsertTemplatedPrompt(UPARAM(meta=(MultiLine=true)) const FString& Text,
                               EChatTemplateRole Role = EChatTemplateRole::User,
                               bool bAddAssistantBOS = false, bool bGenerateReply = true,
                               UPARAM(meta=(MultiLine=true)) const FString& AssistantPrefill = TEXT(""));

    UFUNCTION(BlueprintCallable, Category = "LLM Model Component")
    void InsertTemplatedPromptStruct(const FLlamaChatPrompt& ChatPrompt);

    UFUNCTION(BlueprintCallable, Category = "LLM Model Component")
    void InsertRawPrompt(UPARAM(meta = (MultiLine = true)) const FString& Text, bool bGenerateReply = true);

    UFUNCTION(BlueprintCallable, Category = "LLM Model Component - Impersonation via External API")
    void ImpersonateTemplatedPrompt(const FLlamaChatPrompt& ChatPrompt);

    UFUNCTION(BlueprintCallable, Category = "LLM Model Component - Impersonation via External API")
    void ImpersonateTemplatedToken(const FString& Token, EChatTemplateRole Role = EChatTemplateRole::Assistant,
                                   bool bIsEndOfStream = false);

    UFUNCTION(BlueprintPure, Category = "LLM Model Component")
    FString WrapPromptForRole(const FString& Text, EChatTemplateRole Role, const FString& OverrideTemplate);

    UFUNCTION(BlueprintCallable, Category = "LLM Model Component")
    void StopGeneration();

    UFUNCTION(BlueprintCallable, Category = "LLM Model Component")
    void ResumeGeneration();

    UFUNCTION(BlueprintPure, Category = "LLM Model Component")
    FString RawContextHistory();

    UFUNCTION(BlueprintPure, Category = "LLM Model Component")
    FStructuredChatHistory GetStructuredChatHistory();

    // ── Multimodal ───────────────────────────────────────────────────────────

    UFUNCTION(BlueprintCallable, Category = "LLM Model Component - Multimodal")
    void InsertTemplateImagePrompt(UTexture2D* Image, const FString& Text,
                                   EChatTemplateRole Role = EChatTemplateRole::User,
                                   bool bAddAssistantBOS = false, bool bGenerateReply = true);

    UFUNCTION(BlueprintCallable, Category = "LLM Model Component - Multimodal")
    void InsertTemplateImagePromptFromFile(const FString& ImagePath, const FString& Text,
                                           EChatTemplateRole Role = EChatTemplateRole::User,
                                           bool bAddAssistantBOS = false, bool bGenerateReply = true);

    UFUNCTION(BlueprintCallable, Category = "LLM Model Component - Multimodal")
    void InsertTemplateAudioPrompt(const TArray<float>& PCMAudio, const FString& Text,
                                   EChatTemplateRole Role = EChatTemplateRole::User,
                                   bool bAddAssistantBOS = false, bool bGenerateReply = true);

    UFUNCTION(BlueprintCallable, Category = "LLM Model Component - Multimodal")
    void InsertMultimodalPrompt(const FLlamaMultimodalPrompt& Prompt);

    UFUNCTION(BlueprintPure, Category = "LLM Model Component - Multimodal")
    bool IsMultimodalLoaded() const;

    UFUNCTION(BlueprintPure, Category = "LLM Model Component - Multimodal")
    bool SupportsVision() const;

    UFUNCTION(BlueprintPure, Category = "LLM Model Component - Multimodal")
    bool SupportsAudio() const;

    UFUNCTION(BlueprintPure, Category = "LLM Model Component - Multimodal")
    int32 GetAudioSampleRate() const;

    // ── Audio capture wiring ─────────────────────────────────────────────────

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Component - Audio Capture")
    ULlamaAudioCaptureComponent* AudioSource = nullptr;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Component - Audio Capture")
    FString AudioPromptTemplate = TEXT("<__media__>\nRespond to what was said.");

    UFUNCTION(BlueprintCallable, Category = "LLM Model Component - Audio Capture")
    void SetAudioPromptTemplate(const FString& NewTemplate);

    // ── Embedding ────────────────────────────────────────────────────────────

    UFUNCTION(BlueprintCallable, Category = "LLM Model Embedding Mode")
    void GeneratePromptEmbeddingsForText(const FString& Text);

    UFUNCTION(BlueprintCallable, Category = "LLM Model Embedding Mode")
    void GeneratePromptEmbeddingsForTexts(const TArray<FString>& Texts);

    UFUNCTION(BlueprintPure, Category = "LLM Model Embedding Mode")
    int32 GetEmbeddingDimension() const;

    /** C++ helper for tools (URagStore etc.) that need exclusive callbacks. */
    void EmbedTextsAsync(const TArray<FString>& Texts,
        TFunction<void(const TArray<TArray<float>>&, const TArray<FString>&)> OnDone);

    // ── Native escape hatch (advanced) ───────────────────────────────────────

    /** Direct access to the underlying dual-backend for advanced consumers. */
    FLlamaDualBackend* GetBackend() const { return Backend; }

protected:
    /** Owns the FLlamaNative + remote client and contains every dual-routing state machine.
     *  Created in the component constructor; deleted in the destructor. */
    FLlamaDualBackend* Backend = nullptr;

    void WireBackendCallbacks();

    /** Push current ModelParams / Endpoint / behavior toggles down into the backend.
     *  Called from every public mutator before delegating, so the backend always sees
     *  the latest values at the moment of action. */
    void SyncBackendConfig();
};
