// Copyright 2025-current Getnamo.

#pragma once

#include "CoreMinimal.h"
#include "CorpusChunker.generated.h"

/** Which retriever produced a particular chunk in retrieval results. */
UENUM(BlueprintType)
enum class ERagRetrievalSource : uint8
{
    Vector  UMETA(DisplayName = "Vector"),
    BM25    UMETA(DisplayName = "BM25"),
    Hybrid  UMETA(DisplayName = "Hybrid")
};

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

    /** Raw retrieval score. Semantics depend on SourceRetriever:
     *   - Vector : L2 distance      (lower = better)
     *   - BM25   : BM25 score       (higher = better)
     *   - Hybrid : RRF fused score  (higher = better)
     *  0.0 when this chunk was constructed outside a retrieval call (e.g. fresh from
     *  the chunker during ingest). */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Chunk")
    float RetrievalScore = 0.f;

    /** Normalized 0..1 confidence relative to the top-1 result of THIS query.
     *  Top-1 is always 1.0 by construction. Useful for cutoff-style filtering;
     *  NOT an absolute similarity. For an absolute signal use RetrievalScore +
     *  SourceRetriever and interpret per the table above. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Chunk")
    float Confidence = 0.f;

    /** Which retriever produced this chunk. Defaults to Vector for chunks
     *  constructed outside a retrieval call. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Chunk")
    ERagRetrievalSource SourceRetriever = ERagRetrievalSource::Vector;
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
