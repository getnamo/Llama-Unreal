// Copyright 2025-current Getnamo.

using System;
using System.IO;
using UnrealBuildTool;

public class LlamaWhisper : ModuleRules
{
	private string ThirdPartyPath
	{
		get { return Path.GetFullPath(Path.Combine(ModuleDirectory, "../../ThirdParty")); }
	}

	private string LlamaCppIncludePath
	{
		get { return Path.GetFullPath(Path.Combine(ThirdPartyPath, "LlamaCpp/Include")); }
	}

	private string LlamaCppLibPath
	{
		get { return Path.GetFullPath(Path.Combine(ThirdPartyPath, "LlamaCpp/Lib")); }
	}

	private string WhisperCppIncludePath
	{
		get { return Path.GetFullPath(Path.Combine(ThirdPartyPath, "WhisperCpp/include")); }
	}

	private string WhisperCppSrcPath
	{
		get { return Path.GetFullPath(Path.Combine(ThirdPartyPath, "WhisperCpp/src")); }
	}

	public LlamaWhisper(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

		// Required: prevents PCH contamination of whisper_impl.cpp which includes whisper.cpp source directly.
		// whisper.cpp uses standard C++ headers that conflict with Unreal's unity build environment.
		bUseUnity = false;

		// whisper.cpp uses try/catch internally; enable C++ exception handling (/EHsc) for this module.
		bEnableExceptions = true;

		PublicDependencyModuleNames.AddRange(
			new string[]
			{
				"Core",
				"LlamaCore",    // Reuse FLLMThreadTask, FLlamaPaths, FLlamaString
			}
		);

		PrivateDependencyModuleNames.AddRange(
			new string[]
			{
				"CoreUObject",
				"Engine",
				"AudioCaptureCore",    // Audio::FAudioCapture for microphone input
			}
		);

		// whisper.h public include (needed by WhisperInternal.h consumers)
		PublicIncludePaths.Add(WhisperCppIncludePath);

		// whisper-arch.h is included by whisper.cpp itself during compilation
		PrivateIncludePaths.Add(WhisperCppSrcPath);

		// ggml headers — shared with LlamaCore, all ggml backends are already present
		PublicIncludePaths.Add(LlamaCppIncludePath);

		if (Target.Platform == UnrealTargetPlatform.Win64)
		{
			string Win64LibPath = Path.Combine(LlamaCppLibPath, "Win64");

			// Link against existing ggml import libs.
			// DO NOT add RuntimeDependencies for the DLLs — LlamaCore already stages them,
			// adding them again would cause packaging warnings about duplicate files.
			PublicAdditionalLibraries.Add(Path.Combine(Win64LibPath, "ggml.lib"));
			PublicAdditionalLibraries.Add(Path.Combine(Win64LibPath, "ggml-base.lib"));
			PublicAdditionalLibraries.Add(Path.Combine(Win64LibPath, "ggml-cpu.lib"));

			string VulkanLib = Path.Combine(Win64LibPath, "ggml-vulkan.lib");
			if (File.Exists(VulkanLib))
			{
				PublicAdditionalLibraries.Add(VulkanLib);
			}
		}
		else if (Target.Platform == UnrealTargetPlatform.Linux)
		{
			// Linux is not yet tested for whisper compilation inline.
			// ggml shared objects are already staged by LlamaCore.
		}
	}
}
