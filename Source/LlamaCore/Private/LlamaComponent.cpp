// 2023 (c) Mika Pi, Modifications Getnamo

#include "LlamaComponent.h"
#include <atomic>
#include <deque>
#include <thread>
#include <functional>
#include <mutex>
#include "HAL/PlatformTime.h"
#include "Misc/Paths.h"
#include "HAL/FileManager.h"
#include "common/common.h"
//#include "common/gguf.h"

#if PLATFORM_ANDROID
#include "Android/AndroidPlatformFile.h"
#endif

#define GGML_CUDA_DMMV_X 64
#define GGML_CUDA_F16
#define GGML_CUDA_MMV_Y 2
#define GGML_USE_CUBLAS
#define GGML_USE_K_QUANTS
#define K_QUANTS_PER_ITERATION 2


using namespace std;

ULlamaComponent::ULlamaComponent(const FObjectInitializer &ObjectInitializer)
    : UActorComponent(ObjectInitializer), Llama(make_unique<Internal::FLlama>())
{
    PrimaryComponentTick.bCanEverTick = true;
    PrimaryComponentTick.bStartWithTickEnabled = true;

    TokenCallbackInternal = [this](FString NewToken, int32 NewContextLength)
    {
        if (bSyncPromptHistory)
        {
            ModelState.PromptHistory.Append(NewToken);

            //Track partials - Sentences
            if (ModelParams.Advanced.bEmitPartials)
            {
                bool bSplitFound = false;
                //Check new token for separators
                for (const FString& Separator : ModelParams.Advanced.PartialsSeparators)
                {
                    if (NewToken.Contains(Separator))
                    {
                        bSplitFound = true;
                    }
                }

                if (bSplitFound)
                {
                    //Sync Chat history on partial period
                    ModelState.ChatHistory = GetStructuredHistory();

                    //Don't update it to an unknown role (means we haven't properly set it
                    if (LastRoleFromStructuredHistory() != EChatTemplateRole::Unknown)
                    {
                        ModelState.LastRole = LastRoleFromStructuredHistory();
                    }
                    //Grab partial from last message
                    
                    if(ModelState.ChatHistory.History.Num() > 0)
                    {
                        const FStructuredChatMessage &Message = ModelState.ChatHistory.History.Last();
                        //Confirm it's from the assistant
                        if (Message.Role == EChatTemplateRole::Assistant)
                        {
                            //Look for period preceding this one
                            FString Sentence = GetLastSentence(Message.Content);

                            if (!Sentence.IsEmpty())
                            {
                                OnPartialParsed.Broadcast(Sentence);
                            }
                        }
                    }
                }

            }
        }
        ModelState.ContextLength = NewContextLength;
        OnNewTokenGenerated.Broadcast(std::move(NewToken));
    };

    Llama->OnTokenCb = TokenCallbackInternal;

    Llama->OnEosCb = [this](bool StopTokenCausedEos, float TokensPerSecond)
    {
        ModelState.LastTokensPerSecond = TokensPerSecond;

        if (ModelParams.Advanced.bSyncStructuredChatHistory)
        {
            ModelState.ChatHistory = GetStructuredHistory();
            ModelState.LastRole = LastRoleFromStructuredHistory();
        }
        OnEndOfStream.Broadcast(StopTokenCausedEos, TokensPerSecond);
    };
    Llama->OnStartEvalCb = [this]()
    {
        OnStartEval.Broadcast();
    };
    Llama->OnContextResetCb = [this]()
    {
        if (bSyncPromptHistory) 
        {
            ModelState.PromptHistory.Empty();
        }
        OnContextReset.Broadcast();
    };
    Llama->OnErrorCb = [this](FString ErrorMessage)
    {
        OnError.Broadcast(ErrorMessage);
    };

    //NB this list should be static...
    //For now just add Chat ML
    FChatTemplate Template;
    Template.System = TEXT("<|im_start|>system");
    Template.User = TEXT("<|im_start|>user");
    Template.Assistant = TEXT("<|im_start|>assistant");
    Template.CommonSuffix = TEXT("<|im_end|>");
    Template.Delimiter = TEXT("\n");

    CommonChatTemplates.Add(TEXT("ChatML"), Template);

    //Temp hack default to ChatML
    ModelParams.ChatTemplate = Template;


    //All sentence ending formatting.
    ModelParams.Advanced.PartialsSeparators.Add(TEXT("."));
    ModelParams.Advanced.PartialsSeparators.Add(TEXT("?"));
    ModelParams.Advanced.PartialsSeparators.Add(TEXT("!"));
}

ULlamaComponent::~ULlamaComponent() = default;

void ULlamaComponent::Activate(bool bReset)
{
    Super::Activate(bReset);

    //Check our role
    if (ModelParams.ModelRole != EChatTemplateRole::Unknown)
    {
    }

    //if it hasn't been started, this will start it
    Llama->StartStopThread(true);
    Llama->bShouldLog = bDebugLogModelOutput;
    Llama->Activate(bReset, ModelParams);
}

void ULlamaComponent::Deactivate()
{
    Llama->Deactivate();
    Super::Deactivate();
}

void ULlamaComponent::TickComponent(float DeltaTime,
                                    ELevelTick TickType,
                                    FActorComponentTickFunction* ThisTickFunction)
{
    Super::TickComponent(DeltaTime, TickType, ThisTickFunction);
    Llama->Process();
}

void ULlamaComponent::InsertPrompt(const FString& Prompt)
{
    Llama->InsertPrompt(Prompt);
}

void ULlamaComponent::UserImpersonateText(const FString& Text, EChatTemplateRole Role, bool bIsEos)
{
    FString CombinedText = Text;

    //Check last role, ensure we give ourselves an assistant role if we haven't yet.
    if (ModelState.LastRole != Role)
    {
        CombinedText = GetRolePrefix(Role) + Text;

        //Modify the role
        ModelState.LastRole = Role;
    }

    //If this was the last text in the stream, auto-wrap suffix
    if (bIsEos)
    {
        CombinedText += ModelParams.ChatTemplate.CommonSuffix + ModelParams.ChatTemplate.Delimiter;
    }

    TokenCallbackInternal(CombinedText, ModelState.ContextLength + CombinedText.Len());
}

FString ULlamaComponent::WrapPromptForRole(const FString& Content, EChatTemplateRole Role, bool AppendModelRolePrefix)
{
    FString FinalInputText = TEXT("");
    if (Role == EChatTemplateRole::User)
    {
        FinalInputText = ModelParams.ChatTemplate.User + ModelParams.ChatTemplate.Delimiter + Content + ModelParams.ChatTemplate.CommonSuffix + ModelParams.ChatTemplate.Delimiter;
    }
    else if (Role == EChatTemplateRole::Assistant)
    {
        FinalInputText = ModelParams.ChatTemplate.Assistant + ModelParams.ChatTemplate.Delimiter + Content + ModelParams.ChatTemplate.CommonSuffix + ModelParams.ChatTemplate.Delimiter;
    }
    else if (Role == EChatTemplateRole::System)
    {
        FinalInputText = ModelParams.ChatTemplate.System + ModelParams.ChatTemplate.Delimiter + Content + ModelParams.ChatTemplate.CommonSuffix + ModelParams.ChatTemplate.Delimiter;
    }
    else
    {
        return Content;
    }

    if (AppendModelRolePrefix) 
    {
        //Preset role reply
        FinalInputText += GetRolePrefix(EChatTemplateRole::Assistant);
    }

    return FinalInputText;
}

FString ULlamaComponent::GetRolePrefix(EChatTemplateRole Role)
{
    FString Prefix = TEXT("");

    if (Role != EChatTemplateRole::Unknown)
    {
        if (Role == EChatTemplateRole::Assistant)
        {
            Prefix += ModelParams.ChatTemplate.Assistant + ModelParams.ChatTemplate.Delimiter;
        }
        else if (Role == EChatTemplateRole::User)
        {
            Prefix += ModelParams.ChatTemplate.User + ModelParams.ChatTemplate.Delimiter;
        }
        else if (Role == EChatTemplateRole::System)
        {
            Prefix += ModelParams.ChatTemplate.System + ModelParams.ChatTemplate.Delimiter;
        }
    }
    return Prefix;
}

void ULlamaComponent::InsertPromptTemplated(const FString& Content, EChatTemplateRole Role)
{
    Llama->InsertPrompt(WrapPromptForRole(Content, Role, true));
}

void ULlamaComponent::StartStopQThread(bool bShouldRun)
{
    Llama->StartStopThread(bShouldRun);
}

void ULlamaComponent::StopGenerating()
{
    Llama->StopGenerating();
}

void ULlamaComponent::ResumeGenerating()
{
    Llama->ResumeGenerating();
}

void ULlamaComponent::SyncParamsToLlama()
{
    Llama->UpdateParams(ModelParams);
}

FString ULlamaComponent::GetTemplateStrippedPrompt()
{
    FString CleanPrompt;
    
    CleanPrompt = ModelState.PromptHistory.Replace(*ModelParams.ChatTemplate.User, TEXT(""));
    CleanPrompt = CleanPrompt.Replace(*ModelParams.ChatTemplate.Assistant, TEXT(""));
    CleanPrompt = CleanPrompt.Replace(*ModelParams.ChatTemplate.System, TEXT(""));
    CleanPrompt = CleanPrompt.Replace(*ModelParams.ChatTemplate.CommonSuffix, TEXT(""));

    return CleanPrompt;
}

FStructuredChatMessage ULlamaComponent::FirstChatMessageInHistory(const FString& History, FString& Remainder)
{
    FStructuredChatMessage Message;
    Message.Role = EChatTemplateRole::Unknown;

    int32 StartIndex = INDEX_NONE;
    FString StartRole = TEXT("");
    int32 StartSystem = History.Find(ModelParams.ChatTemplate.System, ESearchCase::CaseSensitive, ESearchDir::FromStart, -1);
    int32 StartAssistant = History.Find(ModelParams.ChatTemplate.Assistant, ESearchCase::CaseSensitive, ESearchDir::FromStart, -1);
    int32 StartUser = History.Find(ModelParams.ChatTemplate.User, ESearchCase::CaseSensitive, ESearchDir::FromStart, -1);

    //Early exit
    if (StartSystem == INDEX_NONE &&
        StartAssistant == INDEX_NONE &&
        StartUser == INDEX_NONE)
    {
        //Failed end find
        Remainder = TEXT("");
        return Message;
    }

    //so they aren't the lowest (-1)
    if (StartSystem == INDEX_NONE)
    {
        StartSystem = INT32_MAX;
    }
    if (StartAssistant == INDEX_NONE)
    {
        StartAssistant = INT32_MAX;
    }
    if (StartUser == INDEX_NONE)
    {
        StartUser = INT32_MAX;
    }
    
    if (StartSystem <= StartAssistant &&
        StartSystem <= StartUser)
    {
        StartIndex = StartSystem;
        StartRole = ModelParams.ChatTemplate.System;
        Message.Role = EChatTemplateRole::System;
    }

    else if (
        StartUser <= StartAssistant &&
        StartUser <= StartSystem)
    {
        StartIndex = StartUser;
        StartRole = ModelParams.ChatTemplate.User;
        Message.Role = EChatTemplateRole::User;
    }

    else if (
        StartAssistant <= StartUser &&
        StartAssistant <= StartSystem)
    {
        StartIndex = StartAssistant;
        StartRole = ModelParams.ChatTemplate.Assistant;
        Message.Role = EChatTemplateRole::Assistant;
    }

    //Look for system role first
    if (StartIndex != INDEX_NONE)
    {
        const FString& CommonSuffix = ModelParams.ChatTemplate.CommonSuffix;

        StartIndex = StartIndex + StartRole.Len();

        int32 EndIndex = History.Find(CommonSuffix, ESearchCase::CaseSensitive, ESearchDir::FromStart, StartIndex);

        if (EndIndex != INDEX_NONE)
        {
            int32 Count = EndIndex - StartIndex;
            Message.Content = History.Mid(StartIndex, Count).TrimStartAndEnd();

            EndIndex = EndIndex + CommonSuffix.Len();

            Remainder = History.RightChop(EndIndex);
        }
        else
        {
            //No ending, assume all content belongs to this bit
            Message.Content = History.RightChop(StartIndex).TrimStartAndEnd();
            Remainder = TEXT("");
        }
    }
    return Message;
}

FStructuredChatHistory ULlamaComponent::GetStructuredHistory()
{
    FString WorkingHistory = ModelState.PromptHistory;
    FStructuredChatHistory Chat;


    while (!WorkingHistory.IsEmpty())
    {
        FStructuredChatMessage Message = FirstChatMessageInHistory(WorkingHistory, WorkingHistory);

        //Only add proper role messages
        if (Message.Role != EChatTemplateRole::Unknown)
        {
            Chat.History.Add(Message);
        }
    }
    return Chat;
}



TArray<FString> ULlamaComponent::DebugListDirectoryContent(const FString& InPath)
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
        FullPathDirectory = Internal::FLlama::ParsePathIntoFullPath(Remainder);
#endif
    }
    else
    {
        FullPathDirectory = Internal::FLlama::ParsePathIntoFullPath(InPath);
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

//Simple utility functions to find the last sentence
bool ULlamaComponent::IsSentenceEndingPunctuation(const TCHAR Char)
{
    return Char == TEXT('.') || Char == TEXT('!') || Char == TEXT('?');
}

FString ULlamaComponent::GetLastSentence(const FString& InputString)
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

EChatTemplateRole ULlamaComponent::LastRoleFromStructuredHistory()
{
    if (ModelState.ChatHistory.History.Num() > 0)
    {
        return ModelState.ChatHistory.History.Last().Role;
    }
    else
    {
        return EChatTemplateRole::Unknown;
    }
}

