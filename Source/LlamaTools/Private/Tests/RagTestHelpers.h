// Copyright 2025-current Getnamo.

#pragma once

#include "CoreMinimal.h"
#include "Misc/Paths.h"
#include "HAL/PlatformProcess.h"
#include "HAL/PlatformTime.h"
#include "Containers/Ticker.h"
#include "Templates/Function.h"

// Shared helpers for the model-gated RAG integration tests.
//
// These live in a header with `inline` functions (single named namespace) rather
// than being copy-pasted into each test .cpp as anonymous-namespace statics.
// UE's unity build concatenates multiple .cpp files into one translation unit;
// two anonymous-namespace functions with the same name in that combined TU are an
// ODR violation ("function already has a body"). Adaptive unity hides this locally
// when a file is recently edited (it gets compiled standalone), but a fresh project
// co-compiles all test files and trips the collision. A single inline definition in
// a shared header is unity-safe: `#pragma once` gives one definition per TU, and
// `inline` permits identical definitions across TUs.
namespace LlamaRagTestHelpers
{
    /** First existing embedding GGUF under Saved/Models, or empty if none present. */
    inline FString FindEmbeddingModel()
    {
        const FString Root = FPaths::ProjectSavedDir() / TEXT("Models");
        const TArray<FString> Candidates = {
            TEXT("bge-small-en-v1.5-q4_k_m.gguf"),
            TEXT("nomic-embed-text-v1.5.Q4_K_M.gguf"),
            TEXT("multilingual-e5-large-instruct-q8_0.gguf"),
        };
        for (const FString& F : Candidates)
        {
            const FString Full = Root / F;
            if (FPaths::FileExists(Full)) { return FPaths::ConvertRelativePathToFull(Full); }
        }
        return FString();
    }

    /** First existing chat/answer GGUF under Saved/Models, or empty if none present. */
    inline FString FindChatModel()
    {
        const FString Root = FPaths::ProjectSavedDir() / TEXT("Models");
        const TArray<FString> Candidates = {
            TEXT("google_gemma-3-4b-it-Q4_K_L.gguf"),
            TEXT("gemma-4-E2B-it-Q6_K.gguf"),
            TEXT("Qwen2.5-Omni-7B-Q4_K_M.gguf"),
            TEXT("Qwen3.5-9B-Q4_K_M.gguf"),
        };
        for (const FString& F : Candidates)
        {
            const FString Full = Root / F;
            if (FPaths::FileExists(Full)) { return FPaths::ConvertRelativePathToFull(Full); }
        }
        return FString();
    }

    /** The in-tree RagDocs corpus directory, or empty if it doesn't exist. */
    inline FString FindCorpusDir()
    {
        const FString Candidate = FPaths::ConvertRelativePathToFull(
            FPaths::ProjectDir() / TEXT("Notes") / TEXT("RagDocs"));
        return FPaths::DirectoryExists(Candidate) ? Candidate : FString();
    }

    /** Pumps the core ticker until Predicate() is true or TimeoutSec elapses. */
    inline bool WaitFor(double TimeoutSec, TFunctionRef<bool()> Predicate)
    {
        const double Deadline = FPlatformTime::Seconds() + TimeoutSec;
        while (FPlatformTime::Seconds() < Deadline)
        {
            FTSTicker::GetCoreTicker().Tick(0.016f);
            if (Predicate()) { return true; }
            FPlatformProcess::Sleep(0.016f);
        }
        return Predicate();
    }
}
