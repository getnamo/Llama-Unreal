// Copyright 2025-current Getnamo, 2022-23 Mika Pi.

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

	private string LlamaCppLibPath
	{
		get { return Path.GetFullPath(Path.Combine(ModuleDirectory, "../../ThirdParty/LlamaCpp/Lib")); }
	}

	private string LlamaCppBinariesPath
	{
		get { return Path.GetFullPath(Path.Combine(ModuleDirectory, "../../ThirdParty/LlamaCpp/Binaries")); }
	}

	private string LlamaCppIncludePath
	{
		get { return Path.GetFullPath(Path.Combine(ModuleDirectory, "../../ThirdParty/LlamaCpp/Include")); }
	}

	private void LinkDyLib(string DyLib)
	{
		string MacPlatform = "Mac";
		PublicAdditionalLibraries.Add(Path.Combine(LlamaCppLibPath, MacPlatform, DyLib));
		PublicDelayLoadDLLs.Add(Path.Combine(LlamaCppLibPath, MacPlatform, DyLib));
		RuntimeDependencies.Add(Path.Combine(LlamaCppLibPath, MacPlatform, DyLib));
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

		//Includes
		PublicIncludePaths.Add(LlamaCppIncludePath);

		if (Target.Platform == UnrealTargetPlatform.Linux)
		{
			//NB: Currently not working for b4879

			PublicAdditionalLibraries.Add(Path.Combine(LlamaCppLibPath, "Linux", "libllama.so"));
		} 
		else if (Target.Platform == UnrealTargetPlatform.Win64)
		{
			string Win64LibPath = Path.Combine(LlamaCppLibPath, "Win64");
			string CudaPath;

			//We default to vulkan build, turn this off if you want to build with CUDA/cpu only
			bool bTryToUseVulkan = true;
			bool bVulkanGGMLFound = false;

			//Toggle this off if you don't want to include the cuda backend	
			bool bTryToUseCuda = false;
			bool bCudaGGMLFound = false;
			bool bCudaFound = false;

			if(bTryToUseVulkan)
			{
				bVulkanGGMLFound = File.Exists(Path.Combine(Win64LibPath, "ggml-vulkan.lib"));
			}
			if(bTryToUseCuda)
			{
				bCudaGGMLFound = File.Exists(Path.Combine(Win64LibPath, "ggml-cuda.lib"));

				if(bCudaGGMLFound)
				{
					//Almost every dev setup has a CUDA_PATH so try to load cuda in plugin path first;
					//these won't exist unless you're in plugin 'cuda' branch.
					CudaPath = Win64LibPath;

					//Test to see if we have a cuda.lib
					bCudaFound = File.Exists(Path.Combine(Win64LibPath, "cuda.lib"));

					if (!bCudaFound)
					{
						//local cuda not found, try environment path
						CudaPath = Path.Combine(Environment.GetEnvironmentVariable("CUDA_PATH"), "lib", "x64");
						bCudaFound = !string.IsNullOrEmpty(CudaPath);
					}

					if (bCudaFound)
					{
						System.Console.WriteLine("Llama-Unreal building using CUDA dependencies at path " + CudaPath);
					}
				}
			}

			//If you specify LLAMA_PATH, it will take precedence over local path for libs
			string LlamaLibPath = Environment.GetEnvironmentVariable("LLAMA_PATH");
			string LlamaDllPath = LlamaLibPath;
			bool bUsingLlamaEnvPath = !string.IsNullOrEmpty(LlamaLibPath);

			if (!bUsingLlamaEnvPath) 
			{
				LlamaLibPath = Win64LibPath;
				LlamaDllPath = Path.Combine(LlamaCppBinariesPath, "Win64");
			}

			PublicAdditionalLibraries.Add(Path.Combine(LlamaLibPath, "llama.lib"));
			PublicAdditionalLibraries.Add(Path.Combine(LlamaLibPath, "ggml.lib"));
			PublicAdditionalLibraries.Add(Path.Combine(LlamaLibPath, "ggml-base.lib"));
			PublicAdditionalLibraries.Add(Path.Combine(LlamaLibPath, "ggml-cpu.lib"));

			PublicAdditionalLibraries.Add(Path.Combine(LlamaLibPath, "common.lib"));

			RuntimeDependencies.Add("$(BinaryOutputDir)/ggml.dll", Path.Combine(LlamaDllPath, "ggml.dll"));
			RuntimeDependencies.Add("$(BinaryOutputDir)/ggml-base.dll", Path.Combine(LlamaDllPath, "ggml-base.dll"));
			RuntimeDependencies.Add("$(BinaryOutputDir)/ggml-cpu.dll", Path.Combine(LlamaDllPath, "ggml-cpu.dll"));
			RuntimeDependencies.Add("$(BinaryOutputDir)/llama.dll", Path.Combine(LlamaDllPath, "llama.dll"));

			System.Console.WriteLine("Llama-Unreal building using llama.lib at path " + LlamaLibPath);

			if(bVulkanGGMLFound)
			{
				PublicAdditionalLibraries.Add(Path.Combine(Win64LibPath, "ggml-vulkan.lib"));
				RuntimeDependencies.Add("$(BinaryOutputDir)/ggml-vulkan.dll", Path.Combine(LlamaDllPath, "ggml-vulkan.dll"));
				//PublicDelayLoadDLLs.Add("ggml-vulkan.dll");
				System.Console.WriteLine("Llama-Unreal building using ggml-vulkan.lib at path " + Win64LibPath);
			}
			if(bCudaGGMLFound)
			{
				PublicAdditionalLibraries.Add(Path.Combine(Win64LibPath, "ggml-cuda.lib"));
				RuntimeDependencies.Add("$(BinaryOutputDir)/ggml-cuda.dll", Path.Combine(LlamaDllPath, "ggml-cuda.dll"));
				RuntimeDependencies.Add("$(BinaryOutputDir)/cublas64_12.dll", Path.Combine(LlamaDllPath, "cublas64_12.dll"));
				RuntimeDependencies.Add("$(BinaryOutputDir)/cublasLt64_12.dll", Path.Combine(LlamaDllPath, "cublasLt64_12.dll"));
				RuntimeDependencies.Add("$(BinaryOutputDir)/cudart64_12.dll", Path.Combine(LlamaDllPath, "cudart64_12.dll"));
				//PublicDelayLoadDLLs.Add("ggml-cuda.dll");
				System.Console.WriteLine("Llama-Unreal building using ggml-cuda.lib at path " + Win64LibPath);
			}
		}
		else if (Target.Platform == UnrealTargetPlatform.Mac)
		{
			//NB: Currently not working for b4879

			PublicAdditionalLibraries.Add(Path.Combine(PluginDirectory, "Libraries", "Mac", "libggml_static.a"));
			
			//Dylibs act as both, so include them, add as lib and add as runtime dep
			LinkDyLib("libllama.dylib");
			LinkDyLib("libggml_shared.dylib");
		}
		else if (Target.Platform == UnrealTargetPlatform.Android)
		{
			//NB: Currently not working for b4879

			//Built against NDK 25.1.8937393, API 26
			PublicAdditionalLibraries.Add(Path.Combine(PluginDirectory, "Libraries", "Android", "libggml_static.a"));
			PublicAdditionalLibraries.Add(Path.Combine(PluginDirectory, "Libraries", "Android", "libllama.a"));
		}
	}
}
