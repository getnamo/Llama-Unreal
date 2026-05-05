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

	private string HnswLibIncludePath
	{
		get { return Path.GetFullPath(Path.Combine(ModuleDirectory, "../../ThirdParty/HnswLib/Include")); }
	}

	// DLL filename prefixes this plugin owns — used to identify stale copies for cleanup.
	// "ggml" covers all ggml-base/cpu/vulkan variants; cuda/cublas entries omitted (Vulkan backend).
	private static readonly string[] ManagedDllPrefixes = new[]
	{
		"ggml", "llama.", "mtmd."
	};

	private static bool IsManagedDll(string FileName)
	{
		string Lower = FileName.ToLowerInvariant();
		if (!Lower.EndsWith(".dll")) return false;
		foreach (string Prefix in ManagedDllPrefixes)
		{
			if (Lower.StartsWith(Prefix)) return true;
		}
		return false;
	}

	private void CleanStaleDLLs(string SourceDllDir, string TargetDllDir)
	{
		if (!Directory.Exists(TargetDllDir)) return;
		if (!Directory.Exists(SourceDllDir)) return;

		var ExpectedDlls = new System.Collections.Generic.HashSet<string>(StringComparer.OrdinalIgnoreCase);
		foreach (string Path_ in Directory.EnumerateFiles(SourceDllDir, "*.dll"))
		{
			ExpectedDlls.Add(Path.GetFileName(Path_));
		}

		foreach (string DllPath in Directory.EnumerateFiles(TargetDllDir, "*.dll"))
		{
			string DllName = Path.GetFileName(DllPath);
			if (!IsManagedDll(DllName)) continue;
			if (ExpectedDlls.Contains(DllName)) continue;

			try
			{
				File.Delete(DllPath);
				System.Console.WriteLine("Llama-Unreal: removed stale DLL " + DllPath);
			}
			catch (Exception Ex)
			{
				System.Console.WriteLine("Llama-Unreal: could not remove stale DLL " + DllPath + " (" + Ex.Message + ")");
			}
		}
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
				"AudioCaptureCore",    // Audio::FAudioCapture for microphone capture pipeline
				"Media",               // IMediaCaptureSupport, FMediaCaptureDeviceInfo
				"MediaUtils",          // MediaCaptureSupport::EnumerateVideoCaptureDevices
				"MediaAssets",         // UMediaPlayer, UMediaTexture for video capture pipeline
				"HTTP",                // FHttpModule, IHttpRequest for FLlamaDualBackend remote routing
				"Json",                // FJsonObject, request/response serialization for remote
				"JsonUtilities",       // FJsonObjectConverter helpers
				"ImageWrapper",        // PNG encode for remote multimodal image parts
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
		PublicIncludePaths.Add(HnswLibIncludePath);

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

			//Remove stale DLLs from prior builds so old backend variants aren't loaded at runtime.
			//Skipped when LLAMA_PATH is set — consumer is managing their own DLL set.
			if (!bUsingLlamaEnvPath)
			{
				string PluginWin64Binaries = Path.Combine(PluginBinariesPath, "Win64");
				CleanStaleDLLs(LlamaDllPath, PluginWin64Binaries);

				if (Target.ProjectFile != null)
				{
					string ProjectWin64Binaries = Path.Combine(
						Target.ProjectFile.Directory.FullName, "Binaries", "Win64");
					CleanStaleDLLs(LlamaDllPath, ProjectWin64Binaries);
				}
			}

			PublicAdditionalLibraries.Add(Path.Combine(LlamaLibPath, "llama.lib"));
			PublicAdditionalLibraries.Add(Path.Combine(LlamaLibPath, "ggml.lib"));
			PublicAdditionalLibraries.Add(Path.Combine(LlamaLibPath, "ggml-base.lib"));
			PublicAdditionalLibraries.Add(Path.Combine(LlamaLibPath, "ggml-cpu.lib"));

			PublicAdditionalLibraries.Add(Path.Combine(LlamaLibPath, "common.lib"));
			PublicAdditionalLibraries.Add(Path.Combine(LlamaLibPath, "mtmd.lib"));

			// Stage every DLL found in the source dir. $(BinaryOutputDir) resolves correctly
			// for both editor (Project/Binaries/Win64) and packaged builds — UBT handles the rest.
			// Dynamic enumeration means new backend DLLs are picked up without manual additions.
			if (Directory.Exists(LlamaDllPath))
			{
				foreach (string SrcDll in Directory.EnumerateFiles(LlamaDllPath, "*.dll"))
				{
					string DllName = Path.GetFileName(SrcDll);
					RuntimeDependencies.Add("$(BinaryOutputDir)/" + DllName, SrcDll);
				}
			}

			if(bVulkanGGMLFound)
			{
				PublicAdditionalLibraries.Add(Path.Combine(Win64LibPath, "ggml-vulkan.lib"));
			}
			if(bCudaGGMLFound)
			{
				PublicAdditionalLibraries.Add(Path.Combine(Win64LibPath, "ggml-cuda.lib"));
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
