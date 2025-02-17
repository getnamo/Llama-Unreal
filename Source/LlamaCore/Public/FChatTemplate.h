//Easy user-specified chat template, or use common templates. Don't specify if you wish to load GGUF template.
#pragma once
#include "FChatTemplate.generated.h"

USTRUCT(BlueprintType)
struct FChatTemplate
{
	GENERATED_USTRUCT_BODY();

	//Role: System
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Chat Template")
	FString System;

	//Role: User
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Chat Template")
	FString User;

	//Role: Assistant
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Chat Template")
	FString Assistant;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Chat Template")
	FString CommonSuffix;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Chat Template")
	FString Delimiter;

	FChatTemplate()
	{
		System = TEXT("");
		User = TEXT("");
		Assistant = TEXT("");
		CommonSuffix = TEXT("");
		Delimiter = TEXT("");
	}
	bool IsEmptyTemplate()
	{
		return (
			System == TEXT("") &&
			User == TEXT("") &&
			Assistant == TEXT("") &&
			CommonSuffix == TEXT("") && 
			Delimiter == TEXT(""));
	}
};