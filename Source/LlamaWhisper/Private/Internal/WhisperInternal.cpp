// Copyright 2025-current Getnamo.

#include "Internal/WhisperInternal.h"

// whisper.h is included via PublicIncludePaths (ThirdParty/WhisperCpp/include/)
#include "whisper.h"

FWhisperInternal::FWhisperInternal()
{
}

FWhisperInternal::~FWhisperInternal()
{
	UnloadModel();
}

bool FWhisperInternal::LoadModel(const std::string& ModelPath, bool bUseGPU, int32 Threads)
{
	UnloadModel();

	CachedThreads = Threads;

	whisper_context_params CParams = whisper_context_default_params();
	CParams.use_gpu    = bUseGPU;
	CParams.flash_attn = false; // disabled by default for stability across backends

	Context = whisper_init_from_file_with_params(ModelPath.c_str(), CParams);

	if (!Context)
	{
		if (OnError)
		{
			OnError("FWhisperInternal: Failed to load model from: " + ModelPath);
		}
		return false;
	}

	return true;
}

void FWhisperInternal::UnloadModel()
{
	if (Context)
	{
		whisper_free(Context);
		Context = nullptr;
	}
}

bool FWhisperInternal::IsModelLoaded() const
{
	return Context != nullptr;
}

bool FWhisperInternal::TranscribeAudio(const float* PCMSamples, int32 NumSamples,
                                        const std::string& Language, bool bTranslate,
                                        int32 MaxContext, int32 BestOf, int32 BeamSize,
                                        bool bUseBeamSearch)
{
	if (!Context)
	{
		if (OnError)
		{
			OnError("FWhisperInternal: TranscribeAudio called but no model is loaded.");
		}
		return false;
	}

	if (NumSamples <= 0 || PCMSamples == nullptr)
	{
		if (OnError)
		{
			OnError("FWhisperInternal: TranscribeAudio called with empty or null audio data.");
		}
		return false;
	}

	const whisper_sampling_strategy Strategy = bUseBeamSearch
		? WHISPER_SAMPLING_BEAM_SEARCH
		: WHISPER_SAMPLING_GREEDY;

	whisper_full_params WParams = whisper_full_default_params(Strategy);

	WParams.n_threads       = CachedThreads;
	WParams.n_max_text_ctx  = (MaxContext > 0) ? MaxContext : -1;
	WParams.translate       = bTranslate;
	WParams.language        = Language.c_str();  // Pointer is valid for the duration of this call
	WParams.detect_language = Language.empty() || Language == "auto";
	WParams.no_context      = false;
	WParams.single_segment  = false;
	WParams.print_progress  = false;
	WParams.print_realtime  = false;
	WParams.print_special   = false;
	WParams.print_timestamps = false;

	if (bUseBeamSearch)
	{
		WParams.beam_search.beam_size = BeamSize;
	}
	else
	{
		WParams.greedy.best_of = BestOf;
	}

	const int Result = whisper_full(Context, WParams, PCMSamples, NumSamples);

	if (Result != 0)
	{
		if (OnError)
		{
			OnError("FWhisperInternal: whisper_full() failed with code: " + std::to_string(Result));
		}
		return false;
	}

	ExtractAndEmitSegments(/*bIsFinal=*/true);
	return true;
}

void FWhisperInternal::ExtractAndEmitSegments(bool bIsFinal)
{
	if (!Context)
	{
		return;
	}

	const int SegCount = whisper_full_n_segments(Context);
	if (SegCount <= 0)
	{
		return;
	}

	std::string Combined;
	Combined.reserve(256);

	for (int i = 0; i < SegCount; ++i)
	{
		const char* SegText = whisper_full_get_segment_text(Context, i);
		if (SegText)
		{
			Combined += SegText;
		}
	}

	if (!Combined.empty() && OnTranscriptionResult)
	{
		OnTranscriptionResult(Combined, bIsFinal);
	}
}
