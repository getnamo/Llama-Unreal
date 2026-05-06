// Copyright 2025-current Getnamo.

#pragma once

#include "CoreMinimal.h"
#include "UObject/Object.h"
#include "Embedding/CorpusChunker.h"

#include "RagAskTestSink.generated.h"

/** Tiny UFUNCTION-marked relay so the RagAskPipeline test can subscribe to the
 *  URagStore's dynamic multicasts (which require UFUNCTION handlers, not lambdas). */
UCLASS()
class URagAskTestSink : public UObject
{
    GENERATED_BODY()
public:
    bool bAskRetrieved = false;
    bool bAskResponse  = false;
    bool bAskEnd       = false;
    bool bAskError     = false;
    int32 IngestAdded  = -1;
    TArray<FLlamaChunk> RetrievedChunks;
    FString FinalAnswer;
    FString LastError;

    UFUNCTION() void HandleRetrieved(const TArray<FLlamaChunk>& Chunks)
    {
        bAskRetrieved = true;
        RetrievedChunks = Chunks;
    }
    UFUNCTION() void HandleResponse(const FString& Response)
    {
        bAskResponse = true;
        FinalAnswer = Response;
    }
    UFUNCTION() void HandleEnd(bool bStop, float Tps) { (void)bStop; (void)Tps; bAskEnd = true; }
    UFUNCTION() void HandleError(const FString& Err, int32 Code)
    {
        (void)Code;
        bAskError = true;
        LastError = Err;
    }
    UFUNCTION() void HandleIngest(int32 Added) { IngestAdded = Added; }
};
