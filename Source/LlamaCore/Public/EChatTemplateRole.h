#pragma once

UENUM(BlueprintType)
enum class EChatTemplateRole : uint8
{
	User,
	Assistant,
	System,
	Unknown = 255
};