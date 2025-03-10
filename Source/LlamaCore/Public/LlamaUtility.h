#pragma once
#include "CoreMinimal.h"

class FLlamaPaths
{
public:

	//Static helpers
	static FString ModelsRelativeRootPath();
	static FString ParsePathIntoFullPath(const FString& InRelativeOrAbsolutePath);
};

class FLlamaString
{
public:
	static FString ToUE(const std::string& String);
	static std::string ToStd(const FString& String);
};