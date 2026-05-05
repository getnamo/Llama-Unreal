// Copyright 2025-current Getnamo.

#pragma once

#include "CoreMinimal.h"
#include "LlamaRemoteTypes.generated.h"

/**
 * Endpoint configuration for ULlamaRemoteComponent.
 * Targets an OpenAI-compatible HTTP API (e.g. llama-server from llama.cpp,
 * LM Studio, Ollama openai-compat, vLLM, or OpenAI itself).
 */
USTRUCT(BlueprintType)
struct FLlamaRemoteEndpoint
{
    GENERATED_USTRUCT_BODY();

    /** OpenAI-compatible API endpoint, no trailing slash. The server must expose /v1/chat/completions
     *  (and /v1/models, /health, /props for capability discovery). e.g. http://127.0.0.1:8080 */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Llama Remote Endpoint")
    FString BaseUrl = TEXT("http://127.0.0.1:8080");

    /** Connect / initial-response timeout in seconds. Applied to non-streaming requests (health, props, slot ops). */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Llama Remote Endpoint")
    float ConnectTimeoutSeconds = 10.f;

    /** Hard cap on a streaming request's total duration in seconds. 0 = no cap. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Llama Remote Endpoint")
    float RequestTimeoutSeconds = 0.f;

    /** true  -> POST /v1/chat/completions (messages[]) — default.
     *  false -> POST /v1/completions (raw prompt string). Chat template and history disabled. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Llama Remote Endpoint")
    bool bUseChatCompletions = true;

    /** Send `cache_prompt: true` (llama-server extension). Massively speeds up multi-turn chat. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Llama Remote Endpoint")
    bool bCachePromptOnServer = true;

    /** Preferred slot id for stateful KV-cache reuse. -1 = let server auto-assign on first request,
     *  then this component tracks whichever slot was used and reuses it for subsequent calls. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Llama Remote Endpoint")
    int32 PreferredSlotId = -1;
};

/** Non-USTRUCT inline media blob attached to the next outbound user message. */
struct FLlamaRemoteMediaBlob
{
    bool bIsImage = true;     // else audio
    FString Mime;             // e.g. "image/png", "audio/wav"
    TArray<uint8> Bytes;      // encoded file bytes (not raw PCM / raw RGB)
};
