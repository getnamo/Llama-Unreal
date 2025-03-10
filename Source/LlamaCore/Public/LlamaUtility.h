#pragma once
#include "CoreMinimal.h"

class FLlamaPaths
{
public:

	//Static helpers
	static FString ModelsRelativeRootPath();
	static FString ParsePathIntoFullPath(const FString& InRelativeOrAbsolutePath);
};