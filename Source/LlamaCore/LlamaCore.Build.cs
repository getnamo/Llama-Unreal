// Copyright (c) 2022 Mika Pi

using UnrealBuildTool;
using System.IO;
using EpicGames.Core;

public class LlamaCore : ModuleRules
{
	private string PluginBinariesPath
	{
		get { return Path.GetFullPath(Path.Combine(ModuleDirectory, "../../Binaries")); }
	}

	private string PluginLibPath
	{
		get { return Path.GetFullPath(Path.Combine(ModuleDirectory, "../../ThirdParty/LlamaCpp")); }
	}

	private void LinkDyLib(string DyLib)
	{
		string MacPlatform = "Mac";
		PublicAdditionalLibraries.Add(Path.Combine(PluginLibPath, MacPlatform, DyLib));
		PublicDelayLoadDLLs.Add(Path.Combine(PluginLibPath, MacPlatform, DyLib));
		RuntimeDependencies.Add(Path.Combine(PluginLibPath, MacPlatform, DyLib));
	}

	public LlamaCore(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = ModuleRules.PCHUsageMode.UseExplicitOrSharedPCHs;

		PublicIncludePaths.AddRange(
			new string[] {
				// ... add public include paths required here ...
			}
			);


		PrivateIncludePaths.AddRange(
			new string[] {
			}
			);


		PublicDependencyModuleNames.AddRange(
			new string[]
			{
				"Core",
				// ... add other public dependencies that you statically link with here ...
			}
			);


		PrivateDependencyModuleNames.AddRange(
			new string[]
			{
				"CoreUObject",
				"Engine",
				"Slate",
				"SlateCore",
				// ... add private dependencies that you statically link with here ...
			}
			);

		if (Target.bBuildEditor)
		{
			PrivateDependencyModuleNames.AddRange(
				new string[]
				{
					"UnrealEd"
				}
			);
		}

		DynamicallyLoadedModuleNames.AddRange(
			new string[]
			{
				// ... add any modules that your module loads dynamically here ...
			}
		);

		PublicIncludePaths.Add(Path.Combine(PluginDirectory, "Includes"));

		if (Target.Platform == UnrealTargetPlatform.Linux)
		{
			
			PublicAdditionalLibraries.Add(Path.Combine(PluginDirectory, "Libraries", "Linux", "libllama.so"));
		} 
		else if (Target.Platform == UnrealTargetPlatform.Win64)
		{
			//toggle this on for cuda build
			bool bUseCuda = false;
			if (bUseCuda)
			{
				//NB: Creates cuda runtime .dll dependencies, proper import path not defined yet
				//These are usually found in NVIDIA GPU Computing Toolkit\CUDA\v12.2\lib\x64
				PublicAdditionalLibraries.Add(Path.Combine(PluginLibPath, "Win64/Cuda", "cudart.lib"));
				PublicAdditionalLibraries.Add(Path.Combine(PluginLibPath, "Win64/Cuda", "cublas.lib"));
				PublicAdditionalLibraries.Add(Path.Combine(PluginLibPath, "Win64/Cuda", "cuda.lib"));

				PublicAdditionalLibraries.Add(Path.Combine(PluginLibPath, "Win64/Cuda", "llama.lib"));
				PublicAdditionalLibraries.Add(Path.Combine(PluginLibPath, "Win64/Cuda", "ggml_static.lib"));
			}
			else
			{
				//We do not use shared dll atm
				//PublicAdditionalLibraries.Add(Path.Combine(PluginLibPath, "Win64", "ggml_shared.lib"));

				PublicAdditionalLibraries.Add(Path.Combine(PluginLibPath, "Win64", "llama.lib"));
				PublicAdditionalLibraries.Add(Path.Combine(PluginLibPath, "Win64", "ggml_static.lib"));
			}

			//string WinLibDLLPath = Path.Combine(PluginLibPath, "Win64");

			//We do not use shared dll atm
			//RuntimeDependencies.Add("$(BinaryOutputDir)/llama.dll", Path.Combine(WinLibDLLPath, "llama.dll));
			//RuntimeDependencies.Add("$(BinaryOutputDir)/ggml_shared.dll", Path.Combine(WinLibDLLPath, "ggml_shared.dll"));
		}
		else if (Target.Platform == UnrealTargetPlatform.Mac)
		{
			PublicAdditionalLibraries.Add(Path.Combine(PluginDirectory, "Libraries", "Mac", "libggml_static.a"));
			
			//Dylibs act as both, so include them, add as lib and add as runtime dep
			LinkDyLib("libllama.dylib");
			LinkDyLib("libggml_shared.dylib");
		}
		else if (Target.Platform == UnrealTargetPlatform.Android)
		{
			//Built against NDK 25.1.8937393, API 26
			PublicAdditionalLibraries.Add(Path.Combine(PluginDirectory, "Libraries", "Android", "libggml_static.a"));
			PublicAdditionalLibraries.Add(Path.Combine(PluginDirectory, "Libraries", "Android", "libllama.a"));
		}
	}
}
