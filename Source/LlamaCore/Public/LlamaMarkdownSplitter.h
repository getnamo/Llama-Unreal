// Copyright 2025-current Getnamo.

#pragma once

#include "CoreMinimal.h"
#include "LlamaDataTypes.h"

/**
 * Streaming markdown splitter — feed it characters as they arrive, periodically Collect()
 * pending partials. Used by both the local FLlamaNative pipeline and the remote SSE pipeline
 * so OnMarkdownPartialGenerated semantics are identical regardless of backend.
 *
 * Single-threaded by design: callers own the lifetime and only invoke methods from one thread
 * at a time (BG thread for native, GT for remote — both are fine).
 */
struct FLlamaMarkdownSplitter
{
    EMarkdownStreamState CurrentState = EMarkdownStreamState::Text;
    bool bAtLineStart = true;
    int32 PendingStars = 0;
    bool bClosingDelimiter = false;
    bool bConsumingHeadingPrefix = false;

    FString CurrentSegmentText;
    EMarkdownStreamState CurrentSegmentState = EMarkdownStreamState::Text;
    TArray<TPair<FString, EMarkdownStreamState>> PendingSegments;

    /** When non-empty, we're tentatively matching a `<think>` (outside Thinking state) or
     *  `</think>` (inside Thinking state). On full match, we transition state and drop the tag.
     *  On mismatch, the buffered chars are flushed to CurrentSegmentText as literal content,
     *  and the breaking char is reprocessed normally. */
    FString TagMatchBuffer;

    void Reset()
    {
        CurrentState = EMarkdownStreamState::Text;
        bAtLineStart = true;
        PendingStars = 0;
        bClosingDelimiter = false;
        bConsumingHeadingPrefix = false;
        CurrentSegmentText.Empty();
        CurrentSegmentState = EMarkdownStreamState::Text;
        PendingSegments.Empty();
        TagMatchBuffer.Empty();
    }

    void FinalizeCurrentSegment()
    {
        if (!CurrentSegmentText.IsEmpty())
        {
            PendingSegments.Add(TPair<FString, EMarkdownStreamState>(MoveTemp(CurrentSegmentText), CurrentSegmentState));
            CurrentSegmentText.Empty();
        }
        CurrentSegmentState = CurrentState;
    }

    /** Returns the tag we're currently trying to match given CurrentState. */
    static const TCHAR* ExpectedThinkingTag(EMarkdownStreamState State)
    {
        return (State == EMarkdownStreamState::Thinking) ? TEXT("</think>") : TEXT("<think>");
    }

    void ProcessChar(TCHAR Ch, const FLLMMarkdownStreamParams& Cfg)
    {
        // Thinking-tag detection takes precedence. Tag chars never reach the markdown sub-parser.
        if (!TagMatchBuffer.IsEmpty())
        {
            const TCHAR* Expected = ExpectedThinkingTag(CurrentState);
            const int32 ExpectedLen = FCString::Strlen(Expected);
            const int32 NextLen = TagMatchBuffer.Len() + 1;

            // Still a valid prefix of Expected?
            const bool bStillMatches = (NextLen <= ExpectedLen) && (Ch == Expected[TagMatchBuffer.Len()]);
            if (bStillMatches)
            {
                TagMatchBuffer.AppendChar(Ch);
                if (TagMatchBuffer.Len() == ExpectedLen)
                {
                    // Full tag matched — transition state, drop the tag chars entirely.
                    if (CurrentState == EMarkdownStreamState::Thinking)
                    {
                        FinalizeCurrentSegment();
                        CurrentState = EMarkdownStreamState::Text;
                        CurrentSegmentState = CurrentState;
                    }
                    else
                    {
                        FinalizeCurrentSegment();
                        CurrentState = EMarkdownStreamState::Thinking;
                        CurrentSegmentState = CurrentState;
                    }
                    TagMatchBuffer.Empty();
                }
                return;
            }

            // Mismatch — flush buffered prefix as literal text in the current segment, then fall
            // through to process Ch from a clean state. (The buffered chars are emitted raw —
            // no markdown re-parsing of the buffer; they are extremely unlikely to be markdown-meaningful.)
            CurrentSegmentText += TagMatchBuffer;
            TagMatchBuffer.Empty();
            // continue to normal processing below
        }

        if (Ch == TEXT('<'))
        {
            // Begin tentative tag match. We may either complete <think> / </think> or fall back to literal.
            TagMatchBuffer.AppendChar(Ch);
            return;
        }

        // Inside Thinking state, all non-tag chars are plain text. Skip markdown parsing.
        if (CurrentState == EMarkdownStreamState::Thinking)
        {
            CurrentSegmentText += Ch;
            return;
        }

        // Resolve pending stars first
        if (PendingStars > 0)
        {
            if (Ch == TEXT('*'))
            {
                if (bClosingDelimiter)
                {
                    // Closing ** -> exit Bold
                    FinalizeCurrentSegment();
                    CurrentState = EMarkdownStreamState::Text;
                    CurrentSegmentState = CurrentState;
                }
                else
                {
                    // Opening ** -> enter Bold
                    FinalizeCurrentSegment();
                    CurrentState = EMarkdownStreamState::Bold;
                    CurrentSegmentState = CurrentState;
                }
                PendingStars = 0;
                bClosingDelimiter = false;
                bAtLineStart = false;
                return;
            }
            else
            {
                if (bClosingDelimiter)
                {
                    CurrentSegmentText += TEXT('*');
                }
                else
                {
                    FinalizeCurrentSegment();
                    CurrentState = EMarkdownStreamState::Italic;
                    CurrentSegmentState = CurrentState;
                }
                PendingStars = 0;
                bClosingDelimiter = false;
                // Fall through to process Ch normally
            }
        }

        if (bConsumingHeadingPrefix)
        {
            if (Ch == TEXT('#'))
            {
                return;
            }
            if (Ch == TEXT(' '))
            {
                bConsumingHeadingPrefix = false;
                return;
            }
            bConsumingHeadingPrefix = false;
        }

        if (Ch == TEXT('*'))
        {
            if (CurrentState == EMarkdownStreamState::Italic)
            {
                bool bIsSingleWord = !CurrentSegmentText.Contains(TEXT(" "));
                bool bIsEmphasis = Cfg.bSingleWordItalicAsEmphasis && bIsSingleWord;

                if (bIsEmphasis && Cfg.bCollectEmphasisInText)
                {
                    FString EmphasisWord = MoveTemp(CurrentSegmentText);
                    CurrentSegmentText.Empty();
                    CurrentState = EMarkdownStreamState::Text;
                    CurrentSegmentState = EMarkdownStreamState::Text;

                    if (PendingSegments.Num() > 0 && PendingSegments.Last().Value == EMarkdownStreamState::Text)
                    {
                        PendingSegments.Last().Key += EmphasisWord;
                    }
                    else
                    {
                        CurrentSegmentText = EmphasisWord;
                    }
                }
                else
                {
                    if (bIsEmphasis)
                    {
                        CurrentSegmentState = EMarkdownStreamState::Emphasis;
                    }
                    FinalizeCurrentSegment();
                    CurrentState = EMarkdownStreamState::Text;
                    CurrentSegmentState = CurrentState;
                }
                bAtLineStart = false;
                return;
            }
            else if (CurrentState == EMarkdownStreamState::Bold)
            {
                PendingStars = 1;
                bClosingDelimiter = true;
                return;
            }
            else
            {
                PendingStars = 1;
                bClosingDelimiter = false;
                return;
            }
        }

        if (Ch == TEXT('\n'))
        {
            if (CurrentState == EMarkdownStreamState::Heading || CurrentState == EMarkdownStreamState::Quote)
            {
                FinalizeCurrentSegment();
                CurrentState = EMarkdownStreamState::Text;
                CurrentSegmentState = CurrentState;
            }
            bAtLineStart = true;
            CurrentSegmentText += Ch;
            return;
        }

        if (bAtLineStart && CurrentState == EMarkdownStreamState::Text)
        {
            if (Ch == TEXT('#'))
            {
                FinalizeCurrentSegment();
                CurrentState = EMarkdownStreamState::Heading;
                CurrentSegmentState = CurrentState;
                bConsumingHeadingPrefix = true;
                bAtLineStart = false;
                return;
            }
            if (Ch == TEXT('>'))
            {
                FinalizeCurrentSegment();
                CurrentState = EMarkdownStreamState::Quote;
                CurrentSegmentState = CurrentState;
                bAtLineStart = false;
                return;
            }
        }

        if (CurrentState == EMarkdownStreamState::Quote && bAtLineStart && Ch == TEXT(' '))
        {
            bAtLineStart = false;
            return;
        }

        bAtLineStart = false;
        CurrentSegmentText += Ch;
    }

    /** Drain pending segments into OutPartials. Merges consecutive same-state segments and
     *  optionally trims whitespace per Cfg. Safe to call mid-stream and at finalization. */
    void Collect(TArray<TPair<FString, EMarkdownStreamState>>& OutPartials, const FLLMMarkdownStreamParams& Cfg)
    {
        const bool bTrim = Cfg.bTrimMarkdownPartialWhitespace;

        // Flush any partial tag-match buffer as literal text (the stream ended mid-tag, so it
        // wasn't actually a tag — emit what we have).
        if (!TagMatchBuffer.IsEmpty())
        {
            CurrentSegmentText += TagMatchBuffer;
            TagMatchBuffer.Empty();
        }

        if (!CurrentSegmentText.IsEmpty())
        {
            PendingSegments.Add(TPair<FString, EMarkdownStreamState>(MoveTemp(CurrentSegmentText), CurrentSegmentState));
            CurrentSegmentText.Empty();
        }

        for (auto& Seg : PendingSegments)
        {
            if (OutPartials.Num() > 0 && OutPartials.Last().Value == Seg.Value)
            {
                OutPartials.Last().Key += Seg.Key;
            }
            else if (!Seg.Key.IsEmpty())
            {
                OutPartials.Add(MoveTemp(Seg));
            }
        }
        PendingSegments.Empty();

        if (bTrim)
        {
            for (int32 i = OutPartials.Num() - 1; i >= 0; --i)
            {
                OutPartials[i].Key.TrimStartAndEndInline();
                if (OutPartials[i].Key.IsEmpty())
                {
                    OutPartials.RemoveAt(i);
                }
            }
        }
    }
};
