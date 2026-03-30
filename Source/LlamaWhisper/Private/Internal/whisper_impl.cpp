// Copyright 2025-current Getnamo.
//
// Isolated translation unit for whisper.cpp source compilation.
//
// This file exists solely to include the whisper.cpp implementation source in a clean
// compiler context, free from Unreal Engine's header pre-pollution and macro conflicts.
//
// The parent Build.cs sets bUseUnity = false so that this file is compiled independently
// without any shared precompiled header. whisper.cpp includes raw C/C++ standard library
// headers that conflict with Unreal's unity-build environment.
//
// Include path resolution (set in LlamaWhisper.Build.cs):
//   "whisper.h"       -> ThirdParty/WhisperCpp/include/whisper.h
//   "whisper-arch.h"  -> ThirdParty/WhisperCpp/src/whisper-arch.h   (PrivateIncludePaths)
//   "ggml.h" etc.     -> ThirdParty/LlamaCpp/Include/                (PublicIncludePaths)
//
// NOTE ON GGML VERSION COMPATIBILITY:
//   whisper.cpp v1.8.0 uses ggml ~0.9.4 (Sept 2025).
//   The LlamaCpp ThirdParty uses ggml from llama.cpp b5215 (Dec 2024).
//   Both share the same ggml-org codebase. Header API compatibility has been verified;
//   if link errors appear, rebuild the ggml libraries alongside whisper.cpp source.

// Unreal defines check() as an assertion macro. While whisper.cpp does not call check()
// directly (it uses WHISPER_ASSERT / GGML_ASSERT), guard it anyway for safety.
#ifdef check
#	pragma push_macro("check")
#	undef check
#	define WHISPER_IMPL_RESTORE_CHECK
#endif

// ---------------------------------------------------------------------------
// Missing defines: bridge between our ggml (llama.cpp b5215, Dec 2024) and
// whisper.cpp v1.8.0 which targets a slightly newer ggml epoch.
// ---------------------------------------------------------------------------

// GGML_KQ_MASK_PAD: KV cache mask alignment padding, added to ggml after b5215.
// whisper.cpp uses it for attention mask tensor sizing. Value 32 matches the
// newer ggml default and is the only value whisper.cpp uses it for.
#ifndef GGML_KQ_MASK_PAD
#	define GGML_KQ_MASK_PAD 32
#endif

// WHISPER_VERSION: normally injected by CMake into a generated whisper-version.h.
// We define it manually since we compile the source directly.
#ifndef WHISPER_VERSION
#	define WHISPER_VERSION "1.8.4"
#endif

// Suppress MSVC warnings that are benign in third-party code
#if defined(_MSC_VER)
#	pragma warning(push)
#	pragma warning(disable: 4244)  // possible loss of data (float/double conversions)
#	pragma warning(disable: 4267)  // size_t -> int conversion
#	pragma warning(disable: 4305)  // double -> float truncation
#	pragma warning(disable: 4996)  // deprecated C runtime functions (e.g. fopen)
#	pragma warning(disable: 4018)  // signed/unsigned mismatch
#	pragma warning(disable: 4100)  // unreferenced formal parameter
#	pragma warning(disable: 4456)  // declaration hides previous local declaration
#	pragma warning(disable: 4457)  // declaration hides function parameter
#	pragma warning(disable: 4458)  // declaration hides class member
#	pragma warning(disable: 4459)  // declaration hides global declaration
#endif

// Include the whisper.cpp implementation source.
// Path is relative to this file's directory: Private/Internal/ -> ../../../../ThirdParty/WhisperCpp/src/
#include "../../../../ThirdParty/WhisperCpp/src/whisper.cpp"

#if defined(_MSC_VER)
#	pragma warning(pop)
#endif

#ifdef WHISPER_IMPL_RESTORE_CHECK
#	pragma pop_macro("check")
#	undef WHISPER_IMPL_RESTORE_CHECK
#endif
