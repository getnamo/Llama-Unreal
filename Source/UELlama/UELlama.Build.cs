// Copyright (c) 2022 Mika Pi

using UnrealBuildTool;
using System.IO;

public class UELlama : ModuleRules
{
	private string PluginBinariesPath
	{
		get { return Path.GetFullPath(Path.Combine(ModuleDirectory, "../../Binaries/Win64")); }
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
			PublicAdditionalLibraries.Add(Path.Combine(PluginDirectory, "Libraries", "libllama.so"));
			PublicIncludePaths.Add(Path.Combine(PluginDirectory, "Includes"));
		} 
		else if (Target.Platform == UnrealTargetPlatform.Win64)
		{
			PublicAdditionalLibraries.Add(Path.Combine(PluginDirectory, "Libraries", "llama.lib"));
			PublicIncludePaths.Add(Path.Combine(PluginDirectory, "Includes"));

			string LlamaDLLPath = Path.Combine(PluginBinariesPath, "llama.dll");
			RuntimeDependencies.Add(LlamaDLLPath);
		}
	}
}
