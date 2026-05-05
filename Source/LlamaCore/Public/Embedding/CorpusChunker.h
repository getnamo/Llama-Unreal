// Copyright 2025-current Getnamo.

#pragma once

#include "CoreMinimal.h"
#include "CorpusChunker.generated.h"

USTRUCT(BlueprintType)
struct FLlamaChunk
{
    GENERATED_USTRUCT_BODY();

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Chunk")
    FString Text;

    /** Character offset in the source document where this chunk starts. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Chunk")
    int32 StartChar = 0;

    /** Character offset (exclusive) where this chunk ends. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Chunk")
    int32 EndChar = 0;

    /** Optional source identifier set by the caller (filename, URL, doc id). */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Chunk")
    FString Source;
};

USTRUCT(BlueprintType)
struct FLlamaChunkerParams
{
    GENERATED_USTRUCT_BODY();

    /** Target chunk length in characters. Most embedding models tokenize ~4 chars per token. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Chunker")
    int32 TargetChars = 1200;

    /** Overlap between consecutive chunks in characters; helps preserve context across boundaries. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Chunker")
    int32 OverlapChars = 200;

    /** Hard maximum chunk length (chars) before forcing a split mid-paragraph. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Chunker")
    int32 MaxChars = 2000;

    /** Skip chunks shorter than this (in chars) — usually empty or whitespace-only fragments. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Chunker")
    int32 MinChars = 32;
};

/**
 * Deterministic, model-free corpus chunker. Splits on paragraph boundaries first,
 * then uses a sliding character window with overlap for paragraphs that exceed
 * Params.MaxChars. Char counts are approximate to token counts at ~4:1 for most
 * English embedding models.
 */
class LLAMACORE_API FLlamaCorpusChunker
{
public:
    /** Split a single source document into chunks. The Source string is copied into each chunk's metadata. */
    static void ChunkText(const FString& Text, const FString& Source,
                          const FLlamaChunkerParams& Params, TArray<FLlamaChunk>& OutChunks);

    /** Convenience: chunk a UTF-8 file from disk. Returns false if the file can't be read. */
    static bool ChunkFile(const FString& FilePath, const FLlamaChunkerParams& Params, TArray<FLlamaChunk>& OutChunks);
};
