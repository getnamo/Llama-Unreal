// Copyright 2025-current Getnamo.

#pragma once

#include <string>
#include <functional>
#include "CoreMinimal.h"

// Forward-declare whisper context to avoid pulling whisper.h into downstream public headers.
// The full definition is only needed in WhisperInternal.cpp.
struct whisper_context;

/**
 * Pure C++ wrapper around the whisper.cpp C API. Contains no Unreal types in function
 * signatures — all UE integration is handled by FWhisperNative which owns this object.
 *
 * All methods must be called from the same background thread (the whisper BG thread owned
 * by FWhisperNative). The whisper_context is NOT thread-safe for concurrent calls.
 */
class FWhisperInternal
{
public:
	// ---------------------------------------------------------------------------
	// Callbacks (set by FWhisperNative; fired on the BG thread)
	// ---------------------------------------------------------------------------

	/** Called with the concatenated segment text after a successful whisper_full() call.
	 *  bIsFinal is always true for blocking one-shot calls. */
	TFunction<void(const std::string& Text, bool bIsFinal)> OnTranscriptionResult;

	/** Called on any error during model load or inference. */
	TFunction<void(const std::string& ErrorMessage)> OnError;

	// ---------------------------------------------------------------------------
	// Model lifecycle (BG thread only)
	// ---------------------------------------------------------------------------

	/** Load a whisper model from a .bin file on disk. Returns true on success. */
	bool LoadModel(const std::string& ModelPath, bool bUseGPU, int32 Threads);

	/** Unload and free the current whisper context. Safe to call when no model is loaded. */
	void UnloadModel();

	/** Returns true if a model is currently loaded and ready. */
	bool IsModelLoaded() const;

	// ---------------------------------------------------------------------------
	// Inference (BG thread only — blocks until whisper_full() completes)
	// ---------------------------------------------------------------------------

	/**
	 * Transcribe PCM audio data. Blocks until inference completes.
	 *
	 * @param PCMSamples   Float32 mono PCM samples at 16 kHz.
	 * @param NumSamples   Number of float samples (not bytes).
	 * @param Language     BCP-47 language code, or "auto" for auto-detection.
	 * @param bTranslate   Translate output to English.
	 * @param MaxContext   Max prior context tokens (0 = no limit).
	 * @param BestOf       Best-of candidates for greedy sampling.
	 * @param BeamSize     Beam size (used only when bUseBeamSearch is true).
	 * @param bUseBeamSearch If true, use beam search instead of greedy decoding.
	 * @return true if whisper_full() succeeded.
	 */
	bool TranscribeAudio(const float* PCMSamples, int32 NumSamples,
	                     const std::string& Language, bool bTranslate,
	                     int32 MaxContext, int32 BestOf, int32 BeamSize,
	                     bool bUseBeamSearch);

	FWhisperInternal();
	~FWhisperInternal();

private:
	whisper_context* Context = nullptr;
	int32           CachedThreads = 4;

	/** Extract all segments from the last whisper_full() call and emit via OnTranscriptionResult. */
	void ExtractAndEmitSegments(bool bIsFinal);
};
