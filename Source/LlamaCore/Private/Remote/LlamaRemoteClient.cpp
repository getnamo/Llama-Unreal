// Copyright 2025-current Getnamo.

#include "Remote/LlamaRemoteClient.h"
#include "LlamaUtility.h"
#include "HttpModule.h"
#include "Interfaces/IHttpResponse.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"
#include "Serialization/JsonWriter.h"
#include "Misc/Base64.h"

namespace
{
    // SSE parser state carried across progress callbacks for a single request.
    struct FSseStreamState
    {
        int64 ConsumedBytes = 0;
        FString LineBuffer;     // accumulates until double-newline
        FString DataAccum;      // accumulates multiple data: lines for one event
        bool bDone = false;
        bool bSawEndOfStream = false;
        FString AccumulatedText;
        int32 AcceptedSlotId = -1;
        double StartedAtSeconds = 0.0;
        int32 TokenCount = 0;
    };

    const TCHAR* RoleToString(EChatTemplateRole Role)
    {
        switch (Role)
        {
        case EChatTemplateRole::User:      return TEXT("user");
        case EChatTemplateRole::Assistant: return TEXT("assistant");
        case EChatTemplateRole::System:    return TEXT("system");
        default:                           return TEXT("user");
        }
    }

    void WriteSamplingFields(const TSharedRef<TJsonWriter<>>& W, const FLLMModelParams& P)
    {
        const FLLMSamplingParams& S = P.Advanced.Sampling;
        W->WriteValue(TEXT("temperature"), S.Temp);
        if (S.TopP > 0.f)     W->WriteValue(TEXT("top_p"), S.TopP);
        if (S.TopK > 0)       W->WriteValue(TEXT("top_k"), S.TopK);
        if (S.MinP > 0.f)     W->WriteValue(TEXT("min_p"), S.MinP);
        if (S.TypicalP > 0.f) W->WriteValue(TEXT("typical_p"), S.TypicalP);
        if (S.PenaltyRepeat != 1.f)    W->WriteValue(TEXT("repeat_penalty"), S.PenaltyRepeat);
        if (S.PenaltyFrequency != 0.f) W->WriteValue(TEXT("frequency_penalty"), S.PenaltyFrequency);
        if (S.PenaltyPresence  != 0.f) W->WriteValue(TEXT("presence_penalty"),  S.PenaltyPresence);
        if (S.Mirostat >= 0)
        {
            W->WriteValue(TEXT("mirostat"), S.Mirostat);
            W->WriteValue(TEXT("mirostat_tau"), S.MirostatTau);
            W->WriteValue(TEXT("mirostat_eta"), S.MirostatEta);
        }
        if (P.Seed >= 0) W->WriteValue(TEXT("seed"), P.Seed);
        if (P.StopSequences.Num() > 0)
        {
            W->WriteArrayStart(TEXT("stop"));
            for (const FString& Stop : P.StopSequences) { W->WriteValue(Stop); }
            W->WriteArrayEnd();
        }
    }

    void WriteMessages(const TSharedRef<TJsonWriter<>>& W,
                       const TArray<FStructuredChatMessage>& Messages,
                       const TArray<FLlamaRemoteMediaBlob>& UserMedia)
    {
        W->WriteArrayStart(TEXT("messages"));
        const int32 LastIdx = Messages.Num() - 1;
        for (int32 i = 0; i < Messages.Num(); ++i)
        {
            const FStructuredChatMessage& Msg = Messages[i];
            W->WriteObjectStart();
            W->WriteValue(TEXT("role"), RoleToString(Msg.Role));

            const bool bAttachMedia = (i == LastIdx) && UserMedia.Num() > 0 && Msg.Role == EChatTemplateRole::User;
            if (bAttachMedia)
            {
                W->WriteArrayStart(TEXT("content"));
                W->WriteObjectStart();
                W->WriteValue(TEXT("type"), TEXT("text"));
                W->WriteValue(TEXT("text"), Msg.Content);
                W->WriteObjectEnd();

                for (const FLlamaRemoteMediaBlob& Blob : UserMedia)
                {
                    const FString B64 = FBase64::Encode(Blob.Bytes);
                    W->WriteObjectStart();
                    if (Blob.bIsImage)
                    {
                        W->WriteValue(TEXT("type"), TEXT("image_url"));
                        W->WriteObjectStart(TEXT("image_url"));
                        W->WriteValue(TEXT("url"), FString::Printf(TEXT("data:%s;base64,%s"), *Blob.Mime, *B64));
                        W->WriteObjectEnd();
                    }
                    else
                    {
                        W->WriteValue(TEXT("type"), TEXT("input_audio"));
                        W->WriteObjectStart(TEXT("input_audio"));
                        W->WriteValue(TEXT("data"), B64);
                        // llama-server expects the bare subtype (wav / mp3). Strip "audio/" prefix if present.
                        FString Format = Blob.Mime;
                        Format.RemoveFromStart(TEXT("audio/"));
                        W->WriteValue(TEXT("format"), Format);
                        W->WriteObjectEnd();
                    }
                    W->WriteObjectEnd();
                }
                W->WriteArrayEnd();
            }
            else
            {
                W->WriteValue(TEXT("content"), Msg.Content);
            }
            W->WriteObjectEnd();
        }
        W->WriteArrayEnd();
    }

    FString BuildChatBody(const FLlamaRemoteChatRequest& Req)
    {
        FString Out;
        TSharedRef<TJsonWriter<>> W = TJsonWriterFactory<>::Create(&Out);
        W->WriteObjectStart();
        W->WriteValue(TEXT("stream"), true);
        WriteSamplingFields(W, Req.Params);
        if (Req.Endpoint.bCachePromptOnServer) W->WriteValue(TEXT("cache_prompt"), true);
        if (Req.SlotId >= 0)                   W->WriteValue(TEXT("id_slot"), Req.SlotId);

        if (Req.bUseRawCompletion)
        {
            W->WriteValue(TEXT("prompt"), Req.RawPrompt);
        }
        else
        {
            WriteMessages(W, Req.Messages, Req.UserMedia);
        }
        W->WriteObjectEnd();
        W->Close();
        return Out;
    }

    // Process any new bytes appended to Raw since the last call. Invokes OnDelta per delta chunk.
    // Returns true if we saw [DONE] / [END].
    bool FeedSseBytes(const TArray<uint8>& Raw, FSseStreamState& S,
                      TFunction<void(const FString&)> OnDelta)
    {
        const int64 Total = Raw.Num();
        if (S.ConsumedBytes >= Total) return S.bDone;

        // Split on the last LF in byte space so we never decode a partial UTF-8 code point
        // (0x0A is only ever the byte \n in valid UTF-8).
        int64 LastLf = -1;
        for (int64 i = Total - 1; i >= S.ConsumedBytes; --i)
        {
            if (Raw[i] == (uint8)'\n') { LastLf = i; break; }
        }
        if (LastLf < S.ConsumedBytes) return S.bDone; // no complete line yet

        const int32 NewLen = (int32)(LastLf + 1 - S.ConsumedBytes);
        FUTF8ToTCHAR Conv(reinterpret_cast<const ANSICHAR*>(Raw.GetData() + S.ConsumedBytes), NewLen);
        S.LineBuffer.AppendChars(Conv.Get(), Conv.Length());
        S.ConsumedBytes = LastLf + 1;

        // Walk complete lines (LF-terminated). Events end at a blank line.
        int32 Cursor = 0;
        while (Cursor < S.LineBuffer.Len())
        {
            int32 NewlinePos = INDEX_NONE;
            for (int32 i = Cursor; i < S.LineBuffer.Len(); ++i)
            {
                if (S.LineBuffer[i] == TEXT('\n')) { NewlinePos = i; break; }
            }
            if (NewlinePos == INDEX_NONE) break;

            FString Line = S.LineBuffer.Mid(Cursor, NewlinePos - Cursor);
            Cursor = NewlinePos + 1;
            if (Line.EndsWith(TEXT("\r"))) Line.LeftChopInline(1, EAllowShrinking::No);

            if (Line.IsEmpty())
            {
                // event boundary — flush DataAccum as one event
                if (!S.DataAccum.IsEmpty())
                {
                    const FString Payload = MoveTemp(S.DataAccum);
                    S.DataAccum.Reset();

                    if (Payload == TEXT("[DONE]"))
                    {
                        S.bDone = true;
                        S.bSawEndOfStream = true;
                    }
                    else
                    {
                        TSharedPtr<FJsonObject> Root;
                        TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(Payload);
                        if (FJsonSerializer::Deserialize(Reader, Root) && Root.IsValid())
                        {
                            // OpenAI chat.completion.chunk: choices[0].delta.content
                            // OpenAI text completion:       choices[0].text
                            // llama-server extensions sometimes include id_slot at top level.
                            int32 SlotFromBody = -1;
                            if (Root->TryGetNumberField(TEXT("id_slot"), SlotFromBody))
                            {
                                S.AcceptedSlotId = SlotFromBody;
                            }

                            const TArray<TSharedPtr<FJsonValue>>* ChoicesArr = nullptr;
                            if (Root->TryGetArrayField(TEXT("choices"), ChoicesArr) && ChoicesArr->Num() > 0)
                            {
                                const TSharedPtr<FJsonObject> Choice = (*ChoicesArr)[0]->AsObject();
                                if (Choice.IsValid())
                                {
                                    FString Piece;
                                    const TSharedPtr<FJsonObject>* Delta = nullptr;
                                    if (Choice->TryGetObjectField(TEXT("delta"), Delta) && (*Delta).IsValid())
                                    {
                                        (*Delta)->TryGetStringField(TEXT("content"), Piece);
                                    }
                                    else
                                    {
                                        Choice->TryGetStringField(TEXT("text"), Piece);
                                    }
                                    if (!Piece.IsEmpty())
                                    {
                                        S.TokenCount++;
                                        S.AccumulatedText += Piece;
                                        OnDelta(Piece);
                                    }

                                    // finish_reason present -> server flagged end-of-stream
                                    FString Finish;
                                    if (Choice->TryGetStringField(TEXT("finish_reason"), Finish) && !Finish.IsEmpty())
                                    {
                                        S.bSawEndOfStream = true;
                                    }
                                }
                            }
                        }
                    }
                }
                continue;
            }

            if (Line.StartsWith(TEXT(":")))
            {
                continue; // SSE comment / keep-alive
            }

            FString Field, Value;
            int32 ColonIdx;
            if (Line.FindChar(TEXT(':'), ColonIdx))
            {
                Field = Line.Left(ColonIdx);
                Value = Line.Mid(ColonIdx + 1);
                if (Value.StartsWith(TEXT(" "))) Value.RightChopInline(1, EAllowShrinking::No);
            }
            else
            {
                Field = Line;
            }

            if (Field == TEXT("data"))
            {
                if (!S.DataAccum.IsEmpty()) S.DataAccum.AppendChar(TEXT('\n'));
                S.DataAccum += Value;
            }
            // ignore event:/id:/retry:
        }

        if (Cursor > 0) S.LineBuffer.RightChopInline(Cursor, EAllowShrinking::No);
        return S.bDone;
    }
}

FString FLlamaRemoteClient::JoinUrl(const FString& Base, const FString& Path)
{
    FString B = Base;
    while (B.EndsWith(TEXT("/"))) B.LeftChopInline(1, EAllowShrinking::No);
    FString P = Path;
    if (!P.StartsWith(TEXT("/"))) P.InsertAt(0, TEXT('/'));
    return B + P;
}

void FLlamaRemoteClient::Health(const FString& BaseUrl, float TimeoutSec,
                                TFunction<void(bool, const FString&)> Done)
{
    TSharedRef<IHttpRequest> Req = FHttpModule::Get().CreateRequest();
    Req->SetURL(JoinUrl(BaseUrl, TEXT("/health")));
    Req->SetVerb(TEXT("GET"));
    if (TimeoutSec > 0.f) Req->SetTimeout(TimeoutSec);
    Req->OnProcessRequestComplete().BindLambda(
        [Done](FHttpRequestPtr, FHttpResponsePtr Resp, bool bOk)
        {
            if (!bOk || !Resp.IsValid())
            {
                Done(false, TEXT("health: no response"));
                return;
            }
            const int32 Code = Resp->GetResponseCode();
            if (Code == 200) Done(true, FString());
            else Done(false, FString::Printf(TEXT("health: HTTP %d"), Code));
        });
    Req->ProcessRequest();
}

void FLlamaRemoteClient::FetchProps(const FString& BaseUrl, float TimeoutSec,
                                    TFunction<void(bool, TSharedPtr<FJsonObject>)> Done)
{
    TSharedRef<IHttpRequest> Req = FHttpModule::Get().CreateRequest();
    Req->SetURL(JoinUrl(BaseUrl, TEXT("/props")));
    Req->SetVerb(TEXT("GET"));
    if (TimeoutSec > 0.f) Req->SetTimeout(TimeoutSec);
    Req->OnProcessRequestComplete().BindLambda(
        [Done](FHttpRequestPtr, FHttpResponsePtr Resp, bool bOk)
        {
            if (!bOk || !Resp.IsValid() || Resp->GetResponseCode() != 200)
            {
                Done(false, nullptr);
                return;
            }
            TSharedPtr<FJsonObject> Root;
            TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(Resp->GetContentAsString());
            if (FJsonSerializer::Deserialize(Reader, Root) && Root.IsValid())
            {
                Done(true, Root);
            }
            else
            {
                Done(false, nullptr);
            }
        });
    Req->ProcessRequest();
}

TSharedPtr<IHttpRequest> FLlamaRemoteClient::StreamChat(
    const FLlamaRemoteChatRequest& Req,
    TFunction<void(const FString&)> OnDelta,
    TFunction<void(const FString&, int32, float)> OnDone,
    TFunction<void(const FString&, int32)> OnError)
{
    TSharedRef<IHttpRequest> Http = FHttpModule::Get().CreateRequest();
    const FString Path = Req.bUseRawCompletion ? TEXT("/v1/completions") : TEXT("/v1/chat/completions");
    Http->SetURL(JoinUrl(Req.Endpoint.BaseUrl, Path));
    Http->SetVerb(TEXT("POST"));
    Http->SetHeader(TEXT("Content-Type"), TEXT("application/json"));
    Http->SetHeader(TEXT("Accept"), TEXT("text/event-stream"));
    if (Req.Endpoint.RequestTimeoutSeconds > 0.f) Http->SetTimeout(Req.Endpoint.RequestTimeoutSeconds);

    const FString Body = BuildChatBody(Req);
    Http->SetContentAsString(Body);

    TSharedRef<FSseStreamState> State = MakeShared<FSseStreamState>();
    State->StartedAtSeconds = FPlatformTime::Seconds();

    // Progress callback: drains any newly-arrived response bytes through the SSE parser.
    Http->OnRequestProgress64().BindLambda(
        [State, OnDelta](FHttpRequestPtr Request, uint64 /*BytesSent*/, uint64 /*BytesReceived*/)
        {
            if (!Request.IsValid() || State->bDone) return;
            FHttpResponsePtr Resp = Request->GetResponse();
            if (!Resp.IsValid()) return;
            const TArray<uint8>& Raw = Resp->GetContent();
            FeedSseBytes(Raw, *State, OnDelta);
        });

    Http->OnProcessRequestComplete().BindLambda(
        [State, OnDelta, OnDone, OnError](FHttpRequestPtr Request, FHttpResponsePtr Resp, bool bOk)
        {
            if (!bOk || !Resp.IsValid())
            {
                const int32 Code = Resp.IsValid() ? Resp->GetResponseCode() : 0;
                OnError(TEXT("remote stream: connection failed"), Code);
                return;
            }
            const int32 Code = Resp->GetResponseCode();
            if (Code < 200 || Code >= 300)
            {
                const FString Preview = Resp->GetContentAsString().Left(512);
                OnError(FString::Printf(TEXT("remote stream: HTTP %d %s"), Code, *Preview), Code);
                return;
            }

            // Final drain — some bytes may only show up on complete (no further progress ticks).
            FeedSseBytes(Resp->GetContent(), *State, OnDelta);

            const double Elapsed = FPlatformTime::Seconds() - State->StartedAtSeconds;
            const float Tps = (Elapsed > 0.0 && State->TokenCount > 0) ? float(State->TokenCount / Elapsed) : 0.f;
            OnDone(State->AccumulatedText, State->AcceptedSlotId, Tps);
        });

    Http->ProcessRequest();
    return Http;
}

void FLlamaRemoteClient::EraseSlot(const FString& BaseUrl, int32 SlotId, float TimeoutSec,
                                   TFunction<void(bool)> Done)
{
    if (SlotId < 0)
    {
        if (Done) Done(false);
        return;
    }
    TSharedRef<IHttpRequest> Req = FHttpModule::Get().CreateRequest();
    Req->SetURL(JoinUrl(BaseUrl, FString::Printf(TEXT("/slots/%d?action=erase"), SlotId)));
    Req->SetVerb(TEXT("POST"));
    Req->SetHeader(TEXT("Content-Type"), TEXT("application/json"));
    Req->SetContentAsString(TEXT("{}"));
    if (TimeoutSec > 0.f) Req->SetTimeout(TimeoutSec);
    Req->OnProcessRequestComplete().BindLambda(
        [Done](FHttpRequestPtr, FHttpResponsePtr Resp, bool bOk)
        {
            const bool bSucceeded = bOk && Resp.IsValid() && Resp->GetResponseCode() >= 200 && Resp->GetResponseCode() < 300;
            if (Done) Done(bSucceeded);
        });
    Req->ProcessRequest();
}

void FLlamaRemoteClient::CancelStream(TSharedPtr<IHttpRequest> Request)
{
    if (Request.IsValid())
    {
        Request->CancelRequest();
    }
}
