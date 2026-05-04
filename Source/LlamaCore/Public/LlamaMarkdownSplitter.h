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

    void ProcessChar(TCHAR Ch, const FLLMMarkdownStreamParams& Cfg)
    {
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
