// Copyright (c) 2022 Mika Pi

using System;
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
			//toggle this off for cpu build if cude is setup
			bool bUseCuda = true;

			//assumes previous installation of llama, defaults to preinstalled location
			string llama = Environment.GetEnvironmentVariable("LLAMA_PATH");
			if (string.IsNullOrEmpty(llama)) { llama = "Win64/Cuda"; }

			string cuda = Environment.GetEnvironmentVariable("CUDA_PATH") + "/lib/x64";

			if (!string.IsNullOrEmpty(cuda) && bUseCuda)
			{
                PublicAdditionalLibraries.Add(Path.Combine(PluginLibPath, cuda, "cudart.lib"));
                PublicAdditionalLibraries.Add(Path.Combine(PluginLibPath, cuda, "cublas.lib"));
                PublicAdditionalLibraries.Add(Path.Combine(PluginLibPath, cuda, "cuda.lib"));
            }

            PublicAdditionalLibraries.Add(Path.Combine(PluginLibPath, llama, "llama.lib"));
            PublicAdditionalLibraries.Add(Path.Combine(PluginLibPath, llama, "ggml_static.lib"));

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
