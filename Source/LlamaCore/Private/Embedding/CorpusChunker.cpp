// Copyright 2025-current Getnamo.

#include "Embedding/CorpusChunker.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"

namespace
{
    // Returns true if `Text[Index]` starts a paragraph break (>= 2 newlines).
    static int32 ParagraphBreakLength(const FString& Text, int32 Index)
    {
        const int32 Len = Text.Len();
        int32 Cursor = Index;
        int32 NewlineCount = 0;
        while (Cursor < Len)
        {
            const TCHAR Ch = Text[Cursor];
            if (Ch == TEXT('\n')) { ++NewlineCount; ++Cursor; }
            else if (Ch == TEXT('\r')) { ++Cursor; }
            else if (Ch == TEXT(' ') || Ch == TEXT('\t')) { ++Cursor; }
            else break;
        }
        return NewlineCount >= 2 ? (Cursor - Index) : 0;
    }

    // Find the last sentence boundary (. ? ! followed by space/newline/end) within [Start, End].
    // Returns End if none found.
    static int32 LastSentenceBoundary(const FString& Text, int32 Start, int32 End)
    {
        for (int32 i = End - 1; i > Start; --i)
        {
            const TCHAR Ch = Text[i];
            if (Ch == TEXT('.') || Ch == TEXT('!') || Ch == TEXT('?'))
            {
                if (i + 1 >= Text.Len()) { return i + 1; }
                const TCHAR Next = Text[i + 1];
                if (Next == TEXT(' ') || Next == TEXT('\n') || Next == TEXT('\r') || Next == TEXT('\t'))
                {
                    return i + 1;
                }
            }
        }
        return End;
    }

    static FString SubstrTrimmed(const FString& Text, int32 Start, int32 End)
    {
        FString Slice = Text.Mid(Start, End - Start);
        Slice.TrimStartAndEndInline();
        return Slice;
    }
}

void FLlamaCorpusChunker::ChunkText(const FString& Text, const FString& Source,
                                    const FLlamaChunkerParams& Params, TArray<FLlamaChunk>& OutChunks)
{
    OutChunks.Reset();

    const int32 Len = Text.Len();
    if (Len == 0) { return; }

    // 1. Walk paragraphs; emit each whole if it fits, else slide-window with overlap.
    int32 ParaStart = 0;
    while (ParaStart < Len)
    {
        // Find next paragraph break or end-of-text.
        int32 ParaEnd = ParaStart;
        while (ParaEnd < Len)
        {
            const int32 BreakLen = ParagraphBreakLength(Text, ParaEnd);
            if (BreakLen > 0) { break; }
            ++ParaEnd;
        }

        if (ParaEnd - ParaStart <= Params.MaxChars)
        {
            FLlamaChunk Chunk;
            Chunk.Text = SubstrTrimmed(Text, ParaStart, ParaEnd);
            Chunk.StartChar = ParaStart;
            Chunk.EndChar = ParaEnd;
            Chunk.Source = Source;
            if (Chunk.Text.Len() >= Params.MinChars)
            {
                OutChunks.Add(MoveTemp(Chunk));
            }
        }
        else
        {
            // Slide a window across this oversized paragraph.
            int32 WindowStart = ParaStart;
            while (WindowStart < ParaEnd)
            {
                int32 WindowEnd = FMath::Min(WindowStart + Params.TargetChars, ParaEnd);

                // Snap to last sentence boundary if the snap point exists in the back half of the window.
                if (WindowEnd < ParaEnd)
                {
                    const int32 Snap = LastSentenceBoundary(Text, WindowStart + Params.TargetChars / 2, WindowEnd);
                    if (Snap > WindowStart && Snap <= WindowEnd) { WindowEnd = Snap; }
                }

                FLlamaChunk Chunk;
                Chunk.Text = SubstrTrimmed(Text, WindowStart, WindowEnd);
                Chunk.StartChar = WindowStart;
                Chunk.EndChar = WindowEnd;
                Chunk.Source = Source;
                if (Chunk.Text.Len() >= Params.MinChars)
                {
                    OutChunks.Add(MoveTemp(Chunk));
                }

                if (WindowEnd >= ParaEnd) { break; }
                WindowStart = FMath::Max(WindowStart + 1, WindowEnd - Params.OverlapChars);
            }
        }

        // Skip the paragraph break.
        const int32 BreakLen = ParagraphBreakLength(Text, ParaEnd);
        ParaStart = ParaEnd + (BreakLen > 0 ? BreakLen : 1);
    }
}

bool FLlamaCorpusChunker::ChunkFile(const FString& FilePath, const FLlamaChunkerParams& Params, TArray<FLlamaChunk>& OutChunks)
{
    FString Body;
    if (!FFileHelper::LoadFileToString(Body, *FilePath))
    {
        return false;
    }
    ChunkText(Body, FPaths::GetCleanFilename(FilePath), Params, OutChunks);
    return true;
}
