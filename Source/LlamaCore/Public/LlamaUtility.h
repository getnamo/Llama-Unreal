#pragma once
#include "CoreMinimal.h"

DECLARE_LOG_CATEGORY_EXTERN(LlamaLog, Log, All);

class FLlamaPaths
{
public:
	static FString ModelsRelativeRootPath();
	static FString ParsePathIntoFullPath(const FString& InRelativeOrAbsolutePath);
};

class FLlamaString
{
public:
	static FString ToUE(const std::string& String);
	static std::string ToStd(const FString& String);
};