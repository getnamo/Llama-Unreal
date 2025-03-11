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
			//We default to vulkan build, turn this off if you want to build with CUDA/cpu only
			bool bVulkanBuild = true;

			//Toggle this off if your CUDA_PATH is not compatible with the build version or
			//you definitely only want CPU build			
			bool bTryToUseCuda = true;

			//First try to load env path llama builds
			bool bCudaFound = false;

			//Check cuda lib status first
			if(bTryToUseCuda && !bVulkanBuild)
			{
				//Almost every dev setup has a CUDA_PATH so try to load cuda in plugin path first;
				//these won't exist unless you're in plugin 'cuda' branch.
				string CudaPath =  Path.Combine(PluginLibPath, "Win64", "Cuda");

				//Test to see if we have a cuda.lib
				bCudaFound = File.Exists(Path.Combine(CudaPath, "cuda.lib"));

				if(!bCudaFound)
				{
					//local cuda not found, try environment path
					CudaPath = Path.Combine(Environment.GetEnvironmentVariable("CUDA_PATH"), "lib", "x64");
					bCudaFound = !string.IsNullOrEmpty(CudaPath);
				}

				if (bCudaFound)
				{
					// PublicAdditionalLibraries.Add(Path.Combine(CudaPath, "cudart.lib"));
					// PublicAdditionalLibraries.Add(Path.Combine(CudaPath, "cublas.lib"));
					// PublicAdditionalLibraries.Add(Path.Combine(CudaPath, "cuda.lib"));

					System.Console.WriteLine("Llama-Unreal building using cuda at path " + CudaPath);
				}
			}

			//If you specify LLAMA_PATH, it will take precedence over local path
			string LlamaPath = Environment.GetEnvironmentVariable("LLAMA_PATH");
			bool bUsingLlamaEnvPath = !string.IsNullOrEmpty(LlamaPath);

			if (!bUsingLlamaEnvPath) 
			{
				LlamaPath = Path.Combine(PluginLibPath, "Win64", "Base");
			}

			PublicAdditionalLibraries.Add(Path.Combine(LlamaPath, "llama.lib"));
			PublicAdditionalLibraries.Add(Path.Combine(LlamaPath, "ggml.lib"));
			PublicAdditionalLibraries.Add(Path.Combine(LlamaPath, "ggml-base.lib"));
			PublicAdditionalLibraries.Add(Path.Combine(LlamaPath, "ggml-cpu.lib"));

			PublicAdditionalLibraries.Add(Path.Combine(LlamaPath, "common.lib"));

			RuntimeDependencies.Add("$(BinaryOutputDir)/ggml.dll", Path.Combine(LlamaPath, "ggml.dll"));
			RuntimeDependencies.Add("$(BinaryOutputDir)/ggml-base.dll", Path.Combine(LlamaPath, "ggml-base.dll"));
			RuntimeDependencies.Add("$(BinaryOutputDir)/ggml-cpu.dll", Path.Combine(LlamaPath, "ggml-cpu.dll"));
			RuntimeDependencies.Add("$(BinaryOutputDir)/llama.dll", Path.Combine(LlamaPath, "llama.dll"));

			System.Console.WriteLine("Llama-Unreal building using llama.lib at path " + LlamaPath);

			if(bVulkanBuild)
			{
				string VulkanPath = Path.Combine(PluginLibPath, "Win64", "Vulkan");
				PublicAdditionalLibraries.Add(Path.Combine(VulkanPath, "ggml-vulkan.lib"));
				RuntimeDependencies.Add("$(BinaryOutputDir)/ggml-vulkan.dll", Path.Combine(VulkanPath, "ggml-vulkan.dll"));
				System.Console.WriteLine("Llama-Unreal building using ggml-vulkan.lib at path " + VulkanPath);
			}
			else if(bCudaFound)
			{
				string CUDAPath = Path.Combine(PluginLibPath, "Win64", "Cuda");
				PublicAdditionalLibraries.Add(Path.Combine(CUDAPath, "ggml-cuda.lib"));
				RuntimeDependencies.Add("$(BinaryOutputDir)/ggml-cuda.dll", Path.Combine(CUDAPath, "ggml-cuda.dll"));

				System.Console.WriteLine("Llama-Unreal building using ggml-cuda.lib at path " + CUDAPath);
			}
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
