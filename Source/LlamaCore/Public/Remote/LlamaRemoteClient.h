// Copyright 2025-current Getnamo.

#pragma once

#include "CoreMinimal.h"
#include "Interfaces/IHttpRequest.h"
#include "LlamaDataTypes.h"
#include "Remote/LlamaRemoteTypes.h"

/**
 * Request payload for FLlamaRemoteClient::StreamChat.
 * All state the client needs to assemble a /v1/chat/completions body.
 */
struct FLlamaRemoteChatRequest
{
    FLlamaRemoteEndpoint Endpoint;
    FLLMModelParams Params;
    TArray<FStructuredChatMessage> Messages;

    /** Media blobs attached to the final User message (if any). Images before audio by convention. */
    TArray<FLlamaRemoteMediaBlob> UserMedia;

    /** Active slot id (-1 = let server assign). */
    int32 SlotId = -1;

    /** For /v1/completions path: raw prompt string (bypasses chat template). */
    FString RawPrompt;
    bool bUseRawCompletion = false;
};

/**
 * Non-UObject HTTP client. Owns request lifetimes. All callbacks fire on the game thread
 * (UE's HTTP module dispatches callbacks via the main-thread ticker).
 */
class LLAMACORE_API FLlamaRemoteClient
{
public:
    FLlamaRemoteClient() = default;
    ~FLlamaRemoteClient() = default;

    /** GET {BaseUrl}/health. Done(bOk, ErrorMessage). */
    void Health(const FString& BaseUrl, float TimeoutSec,
                TFunction<void(bool /*bOk*/, const FString& /*Err*/)> Done);

    /** GET {BaseUrl}/props. Done(bOk, JsonRoot). */
    void FetchProps(const FString& BaseUrl, float TimeoutSec,
                    TFunction<void(bool /*bOk*/, TSharedPtr<FJsonObject> /*Root*/)> Done);

    /** POST {BaseUrl}/v1/chat/completions (or /v1/completions if Req.bUseRawCompletion)
     *  with stream:true. Returns the handle so the caller can cancel. */
    FHttpRequestPtr StreamChat(
        const FLlamaRemoteChatRequest& Req,
        TFunction<void(const FString& /*Delta*/)> OnDelta,
        TFunction<void(const FString& /*FullText*/, int32 /*SlotId*/, float /*TokensPerSecond*/)> OnDone,
        TFunction<void(const FString& /*Err*/, int32 /*Code*/)> OnError);

    /** POST {BaseUrl}/slots/{Id}?action=erase (llama-server ext). Ignored errors = server doesn't support slot ops. */
    void EraseSlot(const FString& BaseUrl, int32 SlotId, float TimeoutSec,
                   TFunction<void(bool /*bOk*/)> Done);

    static void CancelStream(FHttpRequestPtr Request);

private:
    static FString JoinUrl(const FString& Base, const FString& Path);
};
