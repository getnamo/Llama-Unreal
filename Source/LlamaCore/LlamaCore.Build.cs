// Copyright (c) 2022 Mika Pi

using System;
using System.IO;
using UnrealBuildTool;
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
			//toggle this off to stop searching for cuda related setup in build.cs. Will default to CPU build if cuda path fails
			bool bTryToUseCuda = true;

			//First try to load env path llama builds
			bool bCudaFound = false;

			//Check cuda lib status first
			if(bTryToUseCuda)
			{			
				//First try to load cuda in plugin path, these won't exist unless you're in cuda branch
				string CudaPath =  Path.Combine(PluginLibPath, "Win64", "Cuda");

				//test to see if we contain cuda.lib locally
				bCudaFound = File.Exists(Path.Combine(CudaPath, "cuda.lib"));

				if(!bCudaFound)
				{
					//local cuda not found, try environment path
					CudaPath = Environment.GetEnvironmentVariable("CUDA_PATH") + "/lib/x64";
					bCudaFound = !string.IsNullOrEmpty(CudaPath);
				}

				if (bCudaFound)
				{
					PublicAdditionalLibraries.Add(Path.Combine(CudaPath, "cudart.lib"));
					PublicAdditionalLibraries.Add(Path.Combine(CudaPath, "cublas.lib"));
					PublicAdditionalLibraries.Add(Path.Combine(CudaPath, "cuda.lib"));

					System.Console.WriteLine("Llama-Unreal building using cuda at path " + CudaPath);
				}
			}

			string LlamaPath = Environment.GetEnvironmentVariable("LLAMA_PATH");
			bool bUsingLlamaEnvPath = !string.IsNullOrEmpty(LlamaPath);

			if (!bUsingLlamaEnvPath) 
			{
				if(bCudaFound)
				{
					LlamaPath = Path.Combine(PluginLibPath, "Win64", "Cuda");
				}
				else
				{
					LlamaPath = Path.Combine(PluginLibPath, "Win64", "Cpu");
				} 
			}

			PublicAdditionalLibraries.Add(Path.Combine(LlamaPath, "llama.lib"));
            PublicAdditionalLibraries.Add(Path.Combine(LlamaPath, "ggml_static.lib"));

			System.Console.WriteLine("Llama-Unreal building using llama.lib at path " + LlamaPath);

			//We do not use shared dll atm
			//string WinLibDLLPath = Path.Combine(PluginLibPath, "Win64");

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
