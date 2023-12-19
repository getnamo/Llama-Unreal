// Copyright (c) 2022 Mika Pi

using UnrealBuildTool;
using System.IO;

public class UELlama : ModuleRules
{
	private string PluginBinariesPath
	{
		get { return Path.GetFullPath(Path.Combine(ModuleDirectory, "../../Binaries")); }
	}

	private string PluginLibPath
	{
		get { return Path.GetFullPath(Path.Combine(PluginDirectory, "Libraries")); }
	}

	private void LinkDyLib(string DyLib)
	{
		string MacPlatform = "Mac";
		PublicAdditionalLibraries.Add(Path.Combine(PluginLibPath, MacPlatform, DyLib));
		PublicDelayLoadDLLs.Add(Path.Combine(PluginLibPath, MacPlatform, DyLib));
		RuntimeDependencies.Add(Path.Combine(PluginLibPath, MacPlatform, DyLib));
	}

	public UELlama(ReadOnlyTargetRules Target) : base(Target)
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


		if (Target.Platform == UnrealTargetPlatform.Linux)
		{
			PublicAdditionalLibraries.Add(Path.Combine(PluginDirectory, "Libraries", "Linux", "libllama.so"));
			PublicIncludePaths.Add(Path.Combine(PluginDirectory, "Includes"));
		} 
		else if (Target.Platform == UnrealTargetPlatform.Win64)
		{
			PublicAdditionalLibraries.Add(Path.Combine(PluginDirectory, "Libraries", "Win64", "llama.lib"));
			PublicIncludePaths.Add(Path.Combine(PluginDirectory, "Includes"));

			string LlamaDLLPath = Path.Combine(PluginLibPath, "Win64", "llama.dll");
			RuntimeDependencies.Add(LlamaDLLPath);
		}
		else if (Target.Platform == UnrealTargetPlatform.Mac)
		{
			PublicIncludePaths.Add(Path.Combine(PluginDirectory, "Includes"));
			PublicAdditionalLibraries.Add(Path.Combine(PluginDirectory, "Libraries", "Mac", "libggml_static.a"));
			
			//Dylibs act as both, so include them, add as lib and add as runtime dep
			LinkDyLib("libllama.dylib");
			LinkDyLib("libggml_shared.dylib");
		}
		else if (Target.Platform == UnrealTargetPlatform.Android)
		{
			//Built against NDK 26.1.10909125, API 23
			PublicAdditionalLibraries.Add(Path.Combine(PluginDirectory, "Libraries", "Android", "libggml_static.a"));
			PublicAdditionalLibraries.Add(Path.Combine(PluginDirectory, "Libraries", "Android", "libllama.a"));
			PublicIncludePaths.Add(Path.Combine(PluginDirectory, "Includes"));
		}
	}
}
