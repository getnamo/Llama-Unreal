// Copyright 2025-current Getnamo.

#pragma once

#include <string>
#include <vector>
#include "CoreMinimal.h"

DECLARE_LOG_CATEGORY_EXTERN(LlamaLog, Log, All);

class FLlamaPaths
{
public:
	static FString ModelsRelativeRootPath();
	static FString ParsePathIntoFullPath(const FString& InRelativeOrAbsolutePath);

	//Utility function for debugging model location and file enumeration
	static TArray<FString> DebugListDirectoryContent(const FString& InPath);
};

class FLlamaString
{
public:
	static FString ToUE(const std::string& String);
	static std::string ToStd(const FString& String);

	//Simple utility functions to find the last sentence
	static bool IsSentenceEndingPunctuation(const TCHAR Char);
	static FString GetLastSentence(const FString& InputString);

	static void AppendToCharVector(std::vector<char>& VectorHistory, const std::string& Text);
};