// Copyright 2025-current Getnamo.

#pragma once

#include "CoreMinimal.h"
#include "Subsystems/EngineSubsystem.h"
#include "Tickable.h"
#include "Engine/Texture2D.h"
#include "LlamaDataTypes.h"
#include "Remote/LlamaRemoteTypes.h"

class FLlamaDualBackend;

#include "LlamaSubsystem.generated.h"

/**
 * Engine-subsystem LLM API. Functionally identical to ULlamaComponent — same delegates,
 * same Blueprint surface, same dual-backend (FLlamaDualBackend) — but lives at engine
 * scope and survives level transitions / PIE start-stop. Use this for a singleton chat
 * agent (e.g. a global NPC, system assistant) when you don't want lifetime tied to an actor.
 *
 * Limited to one active model. For multiple parallel LLMs, use ULlamaComponent (one per actor).
 */
UCLASS(Category = "LLM")
class LLAMACORE_API ULlamaSubsystem : public UEngineSubsystem
{
    GENERATED_BODY()
public:
    virtual void Initialize(FSubsystemCollectionBase& Collection) override;
    virtual void Deinitialize() override;

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
    FVoidEventSignature OnStartEval;

    UPROPERTY(BlueprintAssignable)
    FOnEndOfStreamSignature OnEndOfStream;

    UPROPERTY(BlueprintAssignable)
    FVoidEventSignature OnContextReset;

    UPROPERTY(BlueprintAssignable)
    FModelNameSignature OnModelLoaded;

    UPROPERTY(BlueprintAssignable)
    FOnErrorSignature OnError;

    // ── Shared params / state ────────────────────────────────────────────────

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Subsystem")
    FLLMModelParams ModelParams;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Subsystem")
    FLLMModelState ModelState;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Subsystem")
    bool bDebugLogModelOutput = false;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Subsystem")
    bool bSyncPromptHistory = true;

    /** When true (default), a remote→local toggle whose chat-history prefix matches the local
     *  KV cache appends only the new messages instead of full-replaying the entire history.
     *  Hash-verified — falls back to full rebuild on prefix divergence. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Subsystem")
    bool bUseIncrementalKVSyncOnToggle = true;

    // ── Remote routing (defaults to false; local-first) ──────────────────────

    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "LLM Remote")
    bool bUseRemote = false;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Remote",
              meta = (EditCondition = "bUseRemote", EditConditionHides))
    FLlamaRemoteEndpoint Endpoint;

    UFUNCTION(BlueprintCallable, Category = "LLM Remote")
    void SetUseRemote(bool bNewUseRemote);

    UFUNCTION(BlueprintPure, Category = "LLM Remote")
    bool IsUsingRemote() const { return bUseRemote; }

    // ── Loading ──────────────────────────────────────────────────────────────

    UFUNCTION(BlueprintCallable, Category = "LLM Model Subsystem")
    void LoadModel(bool bForceReload = true);

    UFUNCTION(BlueprintCallable, Category = "LLM Model Subsystem")
    void UnloadModel();

    UFUNCTION(BlueprintPure, Category = "LLM Model Subsystem")
    bool IsModelLoaded() const;

    // ── Chat / inference ─────────────────────────────────────────────────────

    UFUNCTION(BlueprintCallable, Category = "LLM Model Subsystem")
    void ResetContextHistory(bool bKeepSystemPrompt = false);

    UFUNCTION(BlueprintCallable, Category = "LLM Model Subsystem")
    void RebuildContextFromHistory(const FStructuredChatHistory& History);

    UFUNCTION(BlueprintCallable, Category = "LLM Model Subsystem")
    void RemoveLastAssistantReply();

    UFUNCTION(BlueprintCallable, Category = "LLM Model Subsystem")
    void RemoveLastUserInput();

    UFUNCTION(BlueprintCallable, Category = "LLM Model Subsystem")
    void RemoveLastNTokens(int32 TokenCount = 1);

    UFUNCTION(BlueprintCallable, Category = "LLM Model Subsystem")
    void InsertTemplatedPrompt(UPARAM(meta=(MultiLine=true)) const FString& Text,
                               EChatTemplateRole Role = EChatTemplateRole::User,
                               bool bAddAssistantBOS = false, bool bGenerateReply = true);

    UFUNCTION(BlueprintCallable, Category = "LLM Model Subsystem")
    void InsertTemplatedPromptStruct(const FLlamaChatPrompt& ChatPrompt);

    UFUNCTION(BlueprintCallable, Category = "LLM Model Subsystem")
    void InsertRawPrompt(UPARAM(meta = (MultiLine = true)) const FString& Text, bool bGenerateReply = true);

    UFUNCTION(BlueprintCallable, Category = "LLM Model Subsystem - Impersonation via External API")
    void ImpersonateTemplatedPrompt(const FLlamaChatPrompt& ChatPrompt);

    UFUNCTION(BlueprintCallable, Category = "LLM Model Subsystem - Impersonation via External API")
    void ImpersonateTemplatedToken(const FString& Token, EChatTemplateRole Role = EChatTemplateRole::Assistant,
                                   bool bIsEndOfStream = false);

    UFUNCTION(BlueprintPure, Category = "LLM Model Subsystem")
    FString WrapPromptForRole(const FString& Text, EChatTemplateRole Role, const FString& OverrideTemplate);

    UFUNCTION(BlueprintCallable, Category = "LLM Model Subsystem")
    void StopGeneration();

    UFUNCTION(BlueprintCallable, Category = "LLM Model Subsystem")
    void ResumeGeneration();

    UFUNCTION(BlueprintPure, Category = "LLM Model Subsystem")
    FString RawContextHistory();

    UFUNCTION(BlueprintPure, Category = "LLM Model Subsystem")
    FStructuredChatHistory GetStructuredChatHistory();

    // ── Multimodal ───────────────────────────────────────────────────────────

    UFUNCTION(BlueprintCallable, Category = "LLM Model Subsystem - Multimodal")
    void InsertTemplateImagePrompt(UTexture2D* Image, const FString& Text,
                                   EChatTemplateRole Role = EChatTemplateRole::User,
                                   bool bAddAssistantBOS = false, bool bGenerateReply = true);

    UFUNCTION(BlueprintCallable, Category = "LLM Model Subsystem - Multimodal")
    void InsertTemplateImagePromptFromFile(const FString& ImagePath, const FString& Text,
                                           EChatTemplateRole Role = EChatTemplateRole::User,
                                           bool bAddAssistantBOS = false, bool bGenerateReply = true);

    UFUNCTION(BlueprintCallable, Category = "LLM Model Subsystem - Multimodal")
    void InsertTemplateAudioPrompt(const TArray<float>& PCMAudio, const FString& Text,
                                   EChatTemplateRole Role = EChatTemplateRole::User,
                                   bool bAddAssistantBOS = false, bool bGenerateReply = true);

    UFUNCTION(BlueprintCallable, Category = "LLM Model Subsystem - Multimodal")
    void InsertMultimodalPrompt(const FLlamaMultimodalPrompt& Prompt);

    UFUNCTION(BlueprintPure, Category = "LLM Model Subsystem - Multimodal")
    bool IsMultimodalLoaded() const;

    UFUNCTION(BlueprintPure, Category = "LLM Model Subsystem - Multimodal")
    bool SupportsVision() const;

    UFUNCTION(BlueprintPure, Category = "LLM Model Subsystem - Multimodal")
    bool SupportsAudio() const;

    UFUNCTION(BlueprintPure, Category = "LLM Model Subsystem - Multimodal")
    int32 GetAudioSampleRate() const;

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

    // ── Diagnostics ──────────────────────────────────────────────────────────

    /** Self-recall sanity check on the FVectorDatabase implementation. */
    UFUNCTION(BlueprintCallable, Category = "TESTING")
    float TestVectorSearch();

    /** Direct access to the underlying dual-backend for advanced consumers. */
    FLlamaDualBackend* GetBackend() const { return Backend; }

protected:
    void WireBackendCallbacks();
    void SyncBackendConfig();

private:
    FLlamaDualBackend* Backend = nullptr;
};
