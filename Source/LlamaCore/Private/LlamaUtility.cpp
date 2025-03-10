#include "LlamaUtility.h"
#include "Misc/Paths.h"

DEFINE_LOG_CATEGORY(LlamaLog);

FString FLlamaPaths::ModelsRelativeRootPath()
{
    FString AbsoluteFilePath;

#if PLATFORM_ANDROID
    //This is the path we're allowed to sample on android
    AbsoluteFilePath = FPaths::Combine(FPaths::Combine(FString(FAndroidMisc::GamePersistentDownloadDir()), "Models/"));
#else

    AbsoluteFilePath = FPaths::ConvertRelativePathToFull(FPaths::Combine(FPaths::ProjectSavedDir(), "Models/"));

#endif

    return AbsoluteFilePath;
}

FString FLlamaPaths::ParsePathIntoFullPath(const FString& InRelativeOrAbsolutePath)
{
    FString FinalPath;

    //Is it a relative path?
    if (InRelativeOrAbsolutePath.StartsWith(TEXT(".")))
    {
        //relative path
        //UE_LOG(LogTemp, Log, TEXT("model returning relative path"));
        FinalPath = FPaths::ConvertRelativePathToFull(FLlamaPaths::ModelsRelativeRootPath() + InRelativeOrAbsolutePath);
    }
    else
    {
        //Already an absolute path
        //UE_LOG(LogTemp, Log, TEXT("model returning absolute path"));
        FinalPath = FPaths::ConvertRelativePathToFull(InRelativeOrAbsolutePath);
    }

    return FinalPath;
}

FString FLlamaString::ToUE(const std::string& String)
{
    return FString(UTF8_TO_TCHAR(String.c_str()));
}

std::string FLlamaString::ToStd(const FString& String)
{
    return std::string(TCHAR_TO_UTF8(*String));
}


