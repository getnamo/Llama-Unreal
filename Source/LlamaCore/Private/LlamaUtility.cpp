// Copyright 2025-current Getnamo.

#include "LlamaUtility.h"
#include "Misc/Paths.h"

DEFINE_LOG_CATEGORY(LlamaLog);

//FLlamaPaths
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

TArray<FString> FLlamaPaths::DebugListDirectoryContent(const FString& InPath)
{
    TArray<FString> Entries;

    FString FullPathDirectory;

    if (InPath.Contains(TEXT("<ProjectDir>")))
    {
        FString Remainder = InPath.Replace(TEXT("<ProjectDir>"), TEXT(""));

        FullPathDirectory = FPaths::ProjectDir() + Remainder;
    }
    else if (InPath.Contains(TEXT("<Content>")))
    {
        FString Remainder = InPath.Replace(TEXT("<Content>"), TEXT(""));

        FullPathDirectory = FPaths::ProjectContentDir() + Remainder;
    }
    else if (InPath.Contains(TEXT("<External>")))
    {
        FString Remainder = InPath.Replace(TEXT("<Content>"), TEXT(""));

#if PLATFORM_ANDROID
        FString ExternalStoragePath = FString(FAndroidMisc::GamePersistentDownloadDir());
        FullPathDirectory = ExternalStoragePath + Remainder;
#else
        UE_LOG(LogTemp, Warning, TEXT("Externals not valid in this context!"));
        //FullPathDirectory = FLlamaNative::ParsePathIntoFullPath(Remainder);
#endif
    }
    else
    {
        //FullPathDirectory = FLlamaNative::ParsePathIntoFullPath(InPath);
    }

    IFileManager& FileManager = IFileManager::Get();

    FullPathDirectory = FPaths::ConvertRelativePathToFull(FullPathDirectory);

    FullPathDirectory = FileManager.ConvertToAbsolutePathForExternalAppForRead(*FullPathDirectory);

    Entries.Add(FullPathDirectory);

    UE_LOG(LogTemp, Log, TEXT("Listing contents of <%s>"), *FullPathDirectory);

    // Find directories
    TArray<FString> Directories;
    FString FinalPath = FullPathDirectory / TEXT("*");
    FileManager.FindFiles(Directories, *FinalPath, false, true);
    for (FString Entry : Directories)
    {
        FString FullPath = FullPathDirectory / Entry;
        if (FileManager.DirectoryExists(*FullPath)) // Filter for directories
        {
            UE_LOG(LogTemp, Log, TEXT("Found directory: %s"), *Entry);
            Entries.Add(Entry);
        }
    }

    // Find files
    TArray<FString> Files;
    FileManager.FindFiles(Files, *FullPathDirectory, TEXT("*.*")); // Find all entries
    for (FString Entry : Files)
    {
        FString FullPath = FullPathDirectory / Entry;
        if (!FileManager.DirectoryExists(*FullPath)) // Filter out directories
        {
            UE_LOG(LogTemp, Log, TEXT("Found file: %s"), *Entry);
            Entries.Add(Entry);
        }
    }

    return Entries;
}


//FLlamaString
FString FLlamaString::ToUE(const std::string& String)
{
    return FString(UTF8_TO_TCHAR(String.c_str()));
}

std::string FLlamaString::ToStd(const FString& String)
{
    return std::string(TCHAR_TO_UTF8(*String));
}

bool FLlamaString::IsSentenceEndingPunctuation(const TCHAR Char)
{
    return Char == TEXT('.') || Char == TEXT('!') || Char == TEXT('?');
}

FString FLlamaString::GetLastSentence(const FString& InputString)
{
    int32 LastPunctuationIndex = INDEX_NONE;
    int32 PrecedingPunctuationIndex = INDEX_NONE;

    // Find the last sentence-ending punctuation
    for (int32 i = InputString.Len() - 1; i >= 0; --i)
    {
        if (IsSentenceEndingPunctuation(InputString[i]))
        {
            LastPunctuationIndex = i;
            break;
        }
    }

    // If no punctuation found, return the entire string
    if (LastPunctuationIndex == INDEX_NONE)
    {
        return InputString;
    }

    // Find the preceding sentence-ending punctuation
    for (int32 i = LastPunctuationIndex - 1; i >= 0; --i)
    {
        if (IsSentenceEndingPunctuation(InputString[i]))
        {
            PrecedingPunctuationIndex = i;
            break;
        }
    }

    // Extract the last sentence
    int32 StartIndex = PrecedingPunctuationIndex == INDEX_NONE ? 0 : PrecedingPunctuationIndex + 1;
    return InputString.Mid(StartIndex, LastPunctuationIndex - StartIndex + 1).TrimStartAndEnd();
}

void FLlamaString::AppendToCharVector(std::vector<char>& VectorHistory, const std::string& Text)
{
    VectorHistory.insert(VectorHistory.end(), Text.begin(), Text.end());
}


