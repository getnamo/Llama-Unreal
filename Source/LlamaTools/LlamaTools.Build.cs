// Copyright 2025-current Getnamo.

using System;
using System.IO;
using UnrealBuildTool;

public class LlamaTools : ModuleRules
{
	private string ThirdPartyPath
	{
		get { return Path.GetFullPath(Path.Combine(ModuleDirectory, "../../ThirdParty")); }
	}

	private string LlamaCppIncludePath
	{
		get { return Path.GetFullPath(Path.Combine(ThirdPartyPath, "LlamaCpp/Include")); }
	}

	// hnswlib is RAG-only and header-only. The include directory lives at the
	// plugin top-level ThirdParty/ alongside LlamaCpp and WhisperCpp.
	private string HnswLibIncludePath
	{
		get { return Path.GetFullPath(Path.Combine(ThirdPartyPath, "hnswlib/include")); }
	}

	public LlamaTools(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

		// hnswlib + RagStore use std::vector / std::string paths that may throw on
		// OOM, and FLlamaInternal helpers (e.g. SafeTokenize) cross over via std
		// containers. Match LlamaCore's exception posture.
		bEnableExceptions = true;

		PublicDependencyModuleNames.AddRange(
			new string[]
			{
				"Core",
				"CoreUObject",
				"Engine",
				"LlamaCore",    // RagStore consumes FLlamaDualBackend / FLlamaNative / FLLMModelParams
			}
		);

		PrivateDependencyModuleNames.AddRange(
			new string[]
			{
			}
		);

		// hnswlib (vector ANN backend) — header-only, RAG-exclusive.
		PublicIncludePaths.Add(HnswLibIncludePath);

		// llama.cpp headers — RagStore.h transitively pulls llama.h via
		// LlamaDualBackend.h. ggml libs themselves are NOT relinked here;
		// LlamaCore already publishes them via PublicAdditionalLibraries
		// and we inherit through PublicDependencyModuleNames.
		PublicIncludePaths.Add(LlamaCppIncludePath);
	}
}
