// Copyright 2025-current Getnamo.

#include "Remote/LlamaRemoteComponent.h"
#include "Remote/LlamaRemoteClient.h"
#include "LlamaNative.h"
#include "LlamaUtility.h"
#include "Engine/Texture2D.h"
#include "TextureResource.h"
#include "IImageWrapper.h"
#include "IImageWrapperModule.h"
#include "Modules/ModuleManager.h"
#include "Misc/FileHelper.h"
#include "HAL/PlatformFileManager.h"
#include "Interfaces/IHttpRequest.h"

ULlamaRemoteComponent::ULlamaRemoteComponent(const FObjectInitializer& ObjectInitializer)
    : Super(ObjectInitializer)
{
    // Default to remote routing (preserves existing component contract). The base-class rollback
    // dual-path consults the virtual ShouldBypassNativeKV() predicate (which we override above),
    // so no need to mirror this into ModelParams.bImpersonationMode.
    bUseRemote = true;
    Client = MakeUnique<FLlamaRemoteClient>();
}

ULlamaRemoteComponent::~ULlamaRemoteComponent() = default;

void ULlamaRemoteComponent::BeginDestroy()
{
    if (ActiveStream.IsValid())
    {
        FLlamaRemoteClient::CancelStream(ActiveStream);
        ActiveStream.Reset();
    }
    Super::BeginDestroy();
}

// ---------------- Backend toggle ----------------

void ULlamaRemoteComponent::SetUseRemote(bool bNewUseRemote)
{
    if (bNewUseRemote == bUseRemote) return;

    // Cancel any in-flight inference on the *outgoing* backend before flipping.
    StopGeneration();

    // If leaving remote with a server-side slot, fire-and-forget erase so the server frees its KV prefix.
    if (bUseRemote && AssignedSlotId >= 0 && Client)
    {
        Client->EraseSlot(Endpoint.BaseUrl, AssignedSlotId, Endpoint.ConnectTimeoutSeconds, nullptr);
        AssignedSlotId = -1;
    }

    bUseRemote = bNewUseRemote;
    // No need to sync ModelParams.bImpersonationMode — the base class consults
    // ShouldBypassNativeKV() (overridden above to read bUseRemote).

    // Auto-load the destination only if its config looks valid AND it isn't already loaded.
    if (bUseRemote)
    {
        const bool bConfigOk = !Endpoint.BaseUrl.IsEmpty();
        if (!bConfigOk)
        {
            UE_LOG(LogTemp, Warning, TEXT("SetUseRemote(true): Endpoint.BaseUrl empty, skipping auto-load."));
        }
        else if (!bRemoteModelLoaded)
        {
            LoadModel(/*bForceReload=*/false);
        }
    }
    else
    {
        const bool bConfigOk = !ModelParams.PathToModel.IsEmpty();
        if (!bConfigOk)
        {
            UE_LOG(LogTemp, Warning, TEXT("SetUseRemote(false): ModelParams.PathToModel empty, skipping auto-load."));
        }
        else if (!LlamaNative || !ModelState.bModelIsLoaded)
        {
            Super::LoadModel(/*bForceReload=*/false);
        }
    }

    // Mark history sync pending; actual replay happens lazily on next Insert*. Back-to-back toggles cost nothing.
    bPendingHistorySync = (ModelState.ChatHistory.History.Num() > 0);
}

void ULlamaRemoteComponent::FlushPendingHistorySyncIfNeeded()
{
    if (!bPendingHistorySync) return;

    if (bUseRemote)
    {
        // Remote sends messages[] from ModelState.ChatHistory on every request anyway. Slot was reset
        // in SetUseRemote, so cache_prompt:true rebuilds the server prefix on the next call.
        bPendingHistorySync = false;
        return;
    }

    // Local: need the model loaded to rebuild KV. If not loaded yet, keep flag set; will retry next call.
    if (!LlamaNative || !ModelState.bModelIsLoaded) return;

    LlamaNative->RebuildContextFromHistory(ModelState.ChatHistory);
    bPendingHistorySync = false;
}

void ULlamaRemoteComponent::LoadModel(bool bForceReload)
{
    if (!bUseRemote)
    {
        Super::LoadModel(bForceReload);
        return;
    }

    TWeakObjectPtr<ULlamaRemoteComponent> Weak(this);

    Client->Health(Endpoint.BaseUrl, Endpoint.ConnectTimeoutSeconds,
        [Weak](bool bOk, const FString& Err)
        {
            ULlamaRemoteComponent* Self = Weak.Get();
            if (!Self) return;
            if (!bOk)
            {
                Self->OnError.Broadcast(FString::Printf(TEXT("remote unreachable: %s"), *Err), 61);
                return;
            }

            Self->Client->FetchProps(Self->Endpoint.BaseUrl, Self->Endpoint.ConnectTimeoutSeconds,
                [Weak](bool bPropsOk, TSharedPtr<FJsonObject> Root)
                {
                    ULlamaRemoteComponent* S = Weak.Get();
                    if (!S) return;

                    S->RemoteModelName = TEXT("remote");
                    S->bRemoteVision = false;
                    S->bRemoteAudio = false;
                    S->bRemoteSupportsThinking = false;
                    S->RemoteAudioSampleRate = 16000;

                    if (bPropsOk && Root.IsValid())
                    {
                        FString ModelPath;
                        if (Root->TryGetStringField(TEXT("model_path"), ModelPath) ||
                            Root->TryGetStringField(TEXT("model"), ModelPath))
                        {
                            S->RemoteModelName = FPaths::GetCleanFilename(ModelPath);
                        }

                        FString ChatTemplate;
                        if (Root->TryGetStringField(TEXT("chat_template"), ChatTemplate))
                        {
                            S->ModelState.ChatTemplateInUse.TemplateSource = ChatTemplate;

                            // Mirror FLlamaInternal's auto-detection: Qwen3 / DeepSeek-R1 / etc.
                            // expose <think> in their chat template, or branch on enable_thinking.
                            S->bRemoteSupportsThinking =
                                ChatTemplate.Contains(TEXT("<think>")) ||
                                ChatTemplate.Contains(TEXT("enable_thinking"));
                        }

                        const TArray<TSharedPtr<FJsonValue>>* Modalities = nullptr;
                        if (Root->TryGetArrayField(TEXT("modalities"), Modalities))
                        {
                            for (const TSharedPtr<FJsonValue>& V : *Modalities)
                            {
                                const FString M = V->AsString().ToLower();
                                if (M == TEXT("vision") || M == TEXT("image")) S->bRemoteVision = true;
                                if (M == TEXT("audio"))                        S->bRemoteAudio  = true;
                            }
                        }
                        else
                        {
                            // llama-server sometimes exposes `modalities` inside a nested object.
                            const TSharedPtr<FJsonObject>* Caps = nullptr;
                            if (Root->TryGetObjectField(TEXT("modalities"), Caps) && (*Caps).IsValid())
                            {
                                bool bVision = false, bAudio = false;
                                (*Caps)->TryGetBoolField(TEXT("vision"), bVision);
                                (*Caps)->TryGetBoolField(TEXT("audio"),  bAudio);
                                S->bRemoteVision = bVision;
                                S->bRemoteAudio  = bAudio;
                            }
                        }
                    }

                    // Seed system prompt into history if requested (and not already present).
                    if (S->ModelParams.bAutoInsertSystemPromptOnLoad && !S->ModelParams.SystemPrompt.IsEmpty())
                    {
                        const bool bHasSystem = S->ModelState.ChatHistory.History.ContainsByPredicate(
                            [](const FStructuredChatMessage& M){ return M.Role == EChatTemplateRole::System; });
                        if (!bHasSystem)
                        {
                            FStructuredChatMessage Sys;
                            Sys.Role = EChatTemplateRole::System;
                            Sys.Content = S->ModelParams.SystemPrompt;
                            S->ModelState.ChatHistory.History.Insert(Sys, 0);
                        }
                    }

                    S->bRemoteModelLoaded = true;
                    S->ModelState.bModelIsLoaded = true;
                    S->OnModelLoaded.Broadcast(S->RemoteModelName);
                });
        });
}

void ULlamaRemoteComponent::UnloadModel()
{
    if (!bUseRemote)
    {
        Super::UnloadModel();
        return;
    }

    if (ActiveStream.IsValid())
    {
        FLlamaRemoteClient::CancelStream(ActiveStream);
        ActiveStream.Reset();
    }
    bRemoteModelLoaded = false;
    ModelState.bModelIsLoaded = false;
    AssignedSlotId = -1;
}

bool ULlamaRemoteComponent::IsModelLoaded()
{
    return bUseRemote ? bRemoteModelLoaded : Super::IsModelLoaded();
}

void ULlamaRemoteComponent::InsertTemplatedPrompt(const FString& Text, EChatTemplateRole Role, bool bAddAssistantBOS, bool bGenerateReply)
{
    FlushPendingHistorySyncIfNeeded();
    if (!bUseRemote)
    {
        Super::InsertTemplatedPrompt(Text, Role, bAddAssistantBOS, bGenerateReply);
        return;
    }
    AppendUserMessage(Text, Role);
    if (bGenerateReply)
    {
        BeginStreamFromHistory(true);
    }
}

void ULlamaRemoteComponent::InsertTemplatedPromptStruct(const FLlamaChatPrompt& ChatPrompt)
{
    FlushPendingHistorySyncIfNeeded();
    if (!bUseRemote)
    {
        Super::InsertTemplatedPromptStruct(ChatPrompt);
        return;
    }
    InsertTemplatedPrompt(ChatPrompt.Prompt, ChatPrompt.Role, ChatPrompt.bAddAssistantBOS, ChatPrompt.bGenerateReply);
}

void ULlamaRemoteComponent::InsertRawPrompt(const FString& Text, bool bGenerateReply)
{
    FlushPendingHistorySyncIfNeeded();
    if (!bUseRemote)
    {
        Super::InsertRawPrompt(Text, bGenerateReply);
        return;
    }
    // Raw path uses /v1/completions — bypasses server chat template and local history.
    if (!bRemoteModelLoaded)
    {
        OnError.Broadcast(TEXT("remote model not loaded; call LoadModel first"), 62);
        return;
    }
    if (ActiveStream.IsValid())
    {
        OnError.Broadcast(TEXT("a stream is already in flight; call StopGeneration first"), 63);
        return;
    }

    FLlamaRemoteChatRequest Req;
    Req.Endpoint = Endpoint;
    Req.Params = ModelParams;
    Req.SlotId = (Endpoint.PreferredSlotId >= 0) ? Endpoint.PreferredSlotId : AssignedSlotId;
    Req.bUseRawCompletion = true;
    Req.RawPrompt = Text;

    if (!bGenerateReply) return;

    TWeakObjectPtr<ULlamaRemoteComponent> Weak(this);
    PartialBuffer.Reset();

    ActiveStream = Client->StreamChat(Req,
        [Weak](const FString& Delta)
        {
            if (ULlamaRemoteComponent* Self = Weak.Get()) Self->HandleIncomingDelta(Delta);
        },
        [Weak](const FString& Final, int32 SlotId, float Tps)
        {
            ULlamaRemoteComponent* Self = Weak.Get();
            if (!Self) return;
            if (SlotId >= 0) Self->AssignedSlotId = SlotId;
            Self->FlushPendingPartial(true);
            Self->ModelState.LastTokenGenerationSpeed = Tps;
            Self->OnResponseGenerated.Broadcast(Final);
            Self->OnEndOfStream.Broadcast(true, Tps);
            Self->ActiveStream.Reset();
        },
        [Weak](const FString& Err, int32 Code)
        {
            ULlamaRemoteComponent* Self = Weak.Get();
            if (!Self) return;
            Self->OnError.Broadcast(Err, Code);
            Self->ActiveStream.Reset();
        });
}

void ULlamaRemoteComponent::StopGeneration()
{
    if (!bUseRemote)
    {
        Super::StopGeneration();
        return;
    }
    if (ActiveStream.IsValid())
    {
        FLlamaRemoteClient::CancelStream(ActiveStream);
        ActiveStream.Reset();
        FlushPendingPartial(true);
        OnEndOfStream.Broadcast(false, ModelState.LastTokenGenerationSpeed);
    }
}

void ULlamaRemoteComponent::ResumeGeneration()
{
    if (!bUseRemote)
    {
        Super::ResumeGeneration();
        return;
    }
    // Resuming a truncated remote stream would require server-side resume; not supported by OpenAI API.
    // Re-issue generation from current history instead.
    if (!ActiveStream.IsValid())
    {
        BeginStreamFromHistory(false);
    }
}

void ULlamaRemoteComponent::ResetContextHistory(bool bKeepSystemPrompt)
{
    if (!bUseRemote)
    {
        Super::ResetContextHistory(bKeepSystemPrompt);
        return;
    }
    if (ActiveStream.IsValid())
    {
        FLlamaRemoteClient::CancelStream(ActiveStream);
        ActiveStream.Reset();
    }

    if (bKeepSystemPrompt)
    {
        ModelState.ChatHistory.History.RemoveAll(
            [](const FStructuredChatMessage& M){ return M.Role != EChatTemplateRole::System; });
    }
    else
    {
        ModelState.ChatHistory.History.Reset();
    }
    ModelState.ContextHistory.Reset();
    ModelState.ContextUsed = 0;

    // Best-effort server-side slot erase. Failure is silently ignored (server may be stateless OpenAI).
    if (AssignedSlotId >= 0)
    {
        const int32 OldSlot = AssignedSlotId;
        AssignedSlotId = -1;
        Client->EraseSlot(Endpoint.BaseUrl, OldSlot, Endpoint.ConnectTimeoutSeconds, nullptr);
    }

    OnContextReset.Broadcast();
}

// ---------------- Multimodal ----------------

void ULlamaRemoteComponent::InsertTemplateImagePrompt(UTexture2D* Image, const FString& Text, EChatTemplateRole Role, bool bAddAssistantBOS, bool bGenerateReply)
{
    FlushPendingHistorySyncIfNeeded();
    if (!bUseRemote)
    {
        Super::InsertTemplateImagePrompt(Image, Text, Role, bAddAssistantBOS, bGenerateReply);
        return;
    }
    if (!Image)
    {
        OnError.Broadcast(TEXT("Invalid or null texture passed to InsertTemplateImagePrompt"), 52);
        return;
    }

    FLlamaRemoteMediaBlob Blob;
    EncodeTextureToPng(Image, Blob.Bytes, Blob.Mime);
    if (Blob.Bytes.Num() == 0)
    {
        OnError.Broadcast(TEXT("Failed to PNG-encode texture for remote image prompt"), 53);
        return;
    }
    Blob.bIsImage = true;
    PendingUserMedia.Add(MoveTemp(Blob));

    AppendUserMessage(Text, Role);
    if (bGenerateReply) BeginStreamFromHistory(true);
}

void ULlamaRemoteComponent::InsertTemplateImagePromptFromFile(const FString& ImagePath, const FString& Text, EChatTemplateRole Role, bool bAddAssistantBOS, bool bGenerateReply)
{
    FlushPendingHistorySyncIfNeeded();
    if (!bUseRemote)
    {
        Super::InsertTemplateImagePromptFromFile(ImagePath, Text, Role, bAddAssistantBOS, bGenerateReply);
        return;
    }
    FLlamaRemoteMediaBlob Blob;
    Blob.bIsImage = true;
    if (!LoadImageFileAsPng(ImagePath, Blob.Bytes, Blob.Mime))
    {
        OnError.Broadcast(FString::Printf(TEXT("Failed to read/encode image file: %s"), *ImagePath), 54);
        return;
    }
    PendingUserMedia.Add(MoveTemp(Blob));

    AppendUserMessage(Text, Role);
    if (bGenerateReply) BeginStreamFromHistory(true);
}

void ULlamaRemoteComponent::InsertTemplateAudioPrompt(const TArray<float>& PCMAudio, const FString& Text, EChatTemplateRole Role, bool bAddAssistantBOS, bool bGenerateReply)
{
    FlushPendingHistorySyncIfNeeded();
    if (!bUseRemote)
    {
        Super::InsertTemplateAudioPrompt(PCMAudio, Text, Role, bAddAssistantBOS, bGenerateReply);
        return;
    }
    FLlamaRemoteMediaBlob Blob;
    Blob.bIsImage = false;
    Blob.Mime = TEXT("audio/wav");
    EncodePcmFloatToWav(PCMAudio, RemoteAudioSampleRate, Blob.Bytes);
    PendingUserMedia.Add(MoveTemp(Blob));

    AppendUserMessage(Text, Role);
    if (bGenerateReply) BeginStreamFromHistory(true);
}

void ULlamaRemoteComponent::InsertMultimodalPrompt(const FLlamaMultimodalPrompt& Prompt)
{
    FlushPendingHistorySyncIfNeeded();
    if (!bUseRemote)
    {
        Super::InsertMultimodalPrompt(Prompt);
        return;
    }
    for (const FLlamaMediaEntry& Entry : Prompt.MediaEntries)
    {
        FLlamaRemoteMediaBlob Blob;
        if (Entry.MediaType == ELlamaMediaType::Image)
        {
            Blob.bIsImage = true;
            if (!Entry.FilePath.IsEmpty())
            {
                if (!LoadImageFileAsPng(Entry.FilePath, Blob.Bytes, Blob.Mime))
                {
                    OnError.Broadcast(FString::Printf(TEXT("Failed to read image file: %s"), *Entry.FilePath), 54);
                    return;
                }
            }
            else if (Entry.ImageRGBData.Num() > 0 && Entry.ImageWidth > 0 && Entry.ImageHeight > 0)
            {
                const int32 PixelCount = Entry.ImageWidth * Entry.ImageHeight;
                TArray<uint8> Rgba;
                Rgba.SetNumUninitialized(PixelCount * 4);
                const uint8* Src = Entry.ImageRGBData.GetData();
                uint8* Dst = Rgba.GetData();
                for (int32 i = 0; i < PixelCount; ++i)
                {
                    Dst[i * 4 + 0] = Src[i * 3 + 0];
                    Dst[i * 4 + 1] = Src[i * 3 + 1];
                    Dst[i * 4 + 2] = Src[i * 3 + 2];
                    Dst[i * 4 + 3] = 255;
                }
                IImageWrapperModule& IW = FModuleManager::LoadModuleChecked<IImageWrapperModule>(FName("ImageWrapper"));
                TSharedPtr<IImageWrapper> PNG = IW.CreateImageWrapper(EImageFormat::PNG);
                if (PNG.IsValid() &&
                    PNG->SetRaw(Rgba.GetData(), Rgba.Num(),
                                Entry.ImageWidth, Entry.ImageHeight, ERGBFormat::RGBA, 8))
                {
                    TArray64<uint8> Compressed = PNG->GetCompressed(100);
                    Blob.Bytes.Append(Compressed.GetData(), Compressed.Num());
                    Blob.Mime = TEXT("image/png");
                }
            }
            if (Blob.Bytes.Num() == 0)
            {
                OnError.Broadcast(TEXT("Empty image entry in multimodal prompt"), 55);
                return;
            }
        }
        else
        {
            Blob.bIsImage = false;
            Blob.Mime = TEXT("audio/wav");
            if (!Entry.FilePath.IsEmpty())
            {
                if (!FFileHelper::LoadFileToArray(Blob.Bytes, *Entry.FilePath))
                {
                    OnError.Broadcast(FString::Printf(TEXT("Failed to read audio file: %s"), *Entry.FilePath), 57);
                    return;
                }
                // Assume file is already a WAV/MP3/etc; use its extension as format hint.
                const FString Ext = FPaths::GetExtension(Entry.FilePath).ToLower();
                if (!Ext.IsEmpty()) Blob.Mime = FString::Printf(TEXT("audio/%s"), *Ext);
            }
            else
            {
                EncodePcmFloatToWav(Entry.AudioPCMData, RemoteAudioSampleRate, Blob.Bytes);
            }
        }
        PendingUserMedia.Add(MoveTemp(Blob));
    }

    AppendUserMessage(Prompt.Prompt, Prompt.Role);
    if (Prompt.bGenerateReply) BeginStreamFromHistory(true);
}

bool ULlamaRemoteComponent::IsMultimodalLoaded() const
{
    return bUseRemote ? (bRemoteVision || bRemoteAudio) : Super::IsMultimodalLoaded();
}
bool ULlamaRemoteComponent::SupportsVision() const
{
    return bUseRemote ? bRemoteVision : Super::SupportsVision();
}
bool ULlamaRemoteComponent::SupportsAudio() const
{
    return bUseRemote ? bRemoteAudio : Super::SupportsAudio();
}
int32 ULlamaRemoteComponent::GetAudioSampleRate() const
{
    return bUseRemote ? RemoteAudioSampleRate : Super::GetAudioSampleRate();
}

// ---------------- Internals ----------------

void ULlamaRemoteComponent::AppendUserMessage(const FString& Content, EChatTemplateRole Role)
{
    FStructuredChatMessage Msg;
    Msg.Role = Role;
    Msg.Content = Content;
    ModelState.ChatHistory.History.Add(MoveTemp(Msg));
    ModelState.LastRole = Role;
    OnPromptProcessed.Broadcast(0, Role, 0.f);
}

void ULlamaRemoteComponent::BeginStreamFromHistory(bool bAttachPendingMedia)
{
    if (!bRemoteModelLoaded)
    {
        OnError.Broadcast(TEXT("remote model not loaded; call LoadModel first"), 62);
        return;
    }
    if (ActiveStream.IsValid())
    {
        OnError.Broadcast(TEXT("a stream is already in flight; call StopGeneration first"), 63);
        return;
    }

    FLlamaRemoteChatRequest Req;
    Req.Endpoint = Endpoint;
    Req.Params = ModelParams;
    Req.Messages = ModelState.ChatHistory.History;
    Req.SlotId = (Endpoint.PreferredSlotId >= 0) ? Endpoint.PreferredSlotId : AssignedSlotId;
    if (bAttachPendingMedia && PendingUserMedia.Num() > 0)
    {
        Req.UserMedia = MoveTemp(PendingUserMedia);
        PendingUserMedia.Reset();
    }

    // Pre-append an empty Assistant message so in-flight impersonation tokens have a home.
    FStructuredChatMessage Assistant;
    Assistant.Role = EChatTemplateRole::Assistant;
    Assistant.Content = FString();
    ModelState.ChatHistory.History.Add(Assistant);

    TWeakObjectPtr<ULlamaRemoteComponent> Weak(this);
    PartialBuffer.Reset();
    MdSplitter.Reset();

    ActiveStream = Client->StreamChat(Req,
        [Weak](const FString& Delta)
        {
            if (ULlamaRemoteComponent* Self = Weak.Get()) Self->HandleIncomingDelta(Delta);
        },
        [Weak](const FString& Final, int32 SlotId, float Tps)
        {
            ULlamaRemoteComponent* Self = Weak.Get();
            if (!Self) return;
            if (SlotId >= 0) Self->AssignedSlotId = SlotId;
            Self->FlushPendingPartial(true);

            // Flush any remaining markdown segments accumulated by the splitter.
            if (Self->ModelParams.Advanced.Markdown.bSplitMarkdown)
            {
                TArray<TPair<FString, EMarkdownStreamState>> Final_;
                Self->MdSplitter.Collect(Final_, Self->ModelParams.Advanced.Markdown);
                Self->MdSplitter.Reset();
                for (const auto& P : Final_)
                {
                    if (!P.Key.IsEmpty())
                    {
                        Self->OnMarkdownPartialGenerated.Broadcast(P.Key, P.Value);
                    }
                }
            }

            // Replace the placeholder assistant message's content with the **raw** final text
            // (history retains thinking blocks for context, mirroring local FLlamaInternal::Generate
            // which pushes the unstripped Response into Messages on line 828).
            if (Self->ModelState.ChatHistory.History.Num() > 0)
            {
                FStructuredChatMessage& Last = Self->ModelState.ChatHistory.History.Last();
                if (Last.Role == EChatTemplateRole::Assistant)
                {
                    Last.Content = Final;
                }
            }

            // Apply bStripThinkingFromResponse to the broadcast value only — same behavior as
            // FLlamaInternal::Generate (cpp:836-850): drop everything up to and including </think>
            // plus any trailing newlines.
            FString Emitted = Final;
            if (Self->ModelParams.Advanced.Thinking.bStripThinkingFromResponse &&
                Self->bRemoteSupportsThinking)
            {
                const FString CloseTag = TEXT("</think>");
                int32 ClosePos = INDEX_NONE;
                if (Emitted.FindLastChar(TEXT('>'), ClosePos))
                {
                    // Scan for substring; FString::Find is fine but FindLastChar gave us a fast bail.
                    ClosePos = Emitted.Find(CloseTag, ESearchCase::CaseSensitive, ESearchDir::FromStart);
                }
                if (ClosePos != INDEX_NONE)
                {
                    int32 ContentStart = ClosePos + CloseTag.Len();
                    while (ContentStart < Emitted.Len() &&
                           (Emitted[ContentStart] == TEXT('\n') || Emitted[ContentStart] == TEXT('\r')))
                    {
                        ++ContentStart;
                    }
                    Emitted.RightChopInline(ContentStart, EAllowShrinking::No);
                }
            }

            Self->ModelState.LastTokenGenerationSpeed = Tps;
            Self->OnResponseGenerated.Broadcast(Emitted);
            Self->OnEndOfStream.Broadcast(true, Tps);
            Self->ActiveStream.Reset();
        },
        [Weak](const FString& Err, int32 Code)
        {
            ULlamaRemoteComponent* Self = Weak.Get();
            if (!Self) return;

            // Drop the placeholder assistant message we optimistically appended.
            if (Self->ModelState.ChatHistory.History.Num() > 0 &&
                Self->ModelState.ChatHistory.History.Last().Role == EChatTemplateRole::Assistant &&
                Self->ModelState.ChatHistory.History.Last().Content.IsEmpty())
            {
                Self->ModelState.ChatHistory.History.Pop();
            }
            Self->OnError.Broadcast(Err, Code);
            Self->ActiveStream.Reset();
        });
}

void ULlamaRemoteComponent::HandleIncomingDelta(const FString& Delta)
{
    if (Delta.IsEmpty()) return;

    OnTokenGenerated.Broadcast(Delta);

    // Mirror into the in-progress Assistant message so GetStructuredChatHistory stays live.
    if (ModelState.ChatHistory.History.Num() > 0)
    {
        FStructuredChatMessage& Last = ModelState.ChatHistory.History.Last();
        if (Last.Role == EChatTemplateRole::Assistant)
        {
            Last.Content += Delta;
        }
    }

    // Markdown stream splitting — same emission cadence as native (per separator hit).
    if (ModelParams.Advanced.Markdown.bSplitMarkdown)
    {
        for (int32 i = 0; i < Delta.Len(); ++i)
        {
            MdSplitter.ProcessChar(Delta[i], ModelParams.Advanced.Markdown);
        }

        bool bSplitFound = false;
        for (const FString& Sep : ModelParams.Advanced.Output.PartialsSeparators)
        {
            if (!Sep.IsEmpty() && Delta.Contains(Sep)) { bSplitFound = true; break; }
        }
        if (bSplitFound)
        {
            TArray<TPair<FString, EMarkdownStreamState>> MdPartials;
            MdSplitter.Collect(MdPartials, ModelParams.Advanced.Markdown);
            for (const auto& P : MdPartials)
            {
                if (!P.Key.IsEmpty())
                {
                    OnMarkdownPartialGenerated.Broadcast(P.Key, P.Value);
                }
            }
        }
    }

    if (!ModelParams.Advanced.Output.bEmitPartials) return;

    PartialBuffer += Delta;
    const TArray<FString>& Seps = ModelParams.Advanced.Output.PartialsSeparators;
    if (Seps.Num() == 0) return;

    // Emit on every separator hit, leaving the tail in the buffer.
    while (true)
    {
        int32 BestCut = INDEX_NONE;
        for (const FString& Sep : Seps)
        {
            if (Sep.IsEmpty()) continue;
            int32 Idx = INDEX_NONE;
            if (PartialBuffer.FindLastChar(Sep[0], Idx))
            {
                if (Idx > BestCut) BestCut = Idx;
            }
        }
        if (BestCut == INDEX_NONE) break;

        FString Chunk = PartialBuffer.Left(BestCut + 1);
        PartialBuffer.RightChopInline(BestCut + 1, EAllowShrinking::No);
        Chunk.TrimStartAndEndInline();
        if (!Chunk.IsEmpty())
        {
            OnPartialGenerated.Broadcast(Chunk);
        }
        break; // one emit per delta is enough; next delta will re-scan
    }
}

void ULlamaRemoteComponent::FlushPendingPartial(bool bForceEmitRemainder)
{
    if (bForceEmitRemainder && !PartialBuffer.IsEmpty())
    {
        FString Tail = PartialBuffer;
        PartialBuffer.Reset();
        Tail.TrimStartAndEndInline();
        if (!Tail.IsEmpty() && ModelParams.Advanced.Output.bEmitPartials)
        {
            OnPartialGenerated.Broadcast(Tail);
        }
    }
}

// ---------------- Encoding helpers ----------------

void ULlamaRemoteComponent::EncodeTextureToPng(UTexture2D* Image, TArray<uint8>& OutPng, FString& OutMime)
{
    OutPng.Reset();
    OutMime = TEXT("image/png");

    if (!Image || !Image->GetPlatformData() || Image->GetPlatformData()->Mips.Num() == 0) return;

    FTexturePlatformData* PD = Image->GetPlatformData();
    const int32 W = PD->SizeX;
    const int32 H = PD->SizeY;

    const void* Raw = PD->Mips[0].BulkData.LockReadOnly();
    if (!Raw) return;

    TArray<uint8> Bgra;
    Bgra.SetNumUninitialized(W * H * 4);
    FMemory::Memcpy(Bgra.GetData(), Raw, Bgra.Num());
    PD->Mips[0].BulkData.Unlock();

    IImageWrapperModule& IW = FModuleManager::LoadModuleChecked<IImageWrapperModule>(FName("ImageWrapper"));
    TSharedPtr<IImageWrapper> PNG = IW.CreateImageWrapper(EImageFormat::PNG);
    if (PNG.IsValid() && PNG->SetRaw(Bgra.GetData(), Bgra.Num(), W, H, ERGBFormat::BGRA, 8))
    {
        TArray64<uint8> Compressed = PNG->GetCompressed(100);
        OutPng.Append(Compressed.GetData(), Compressed.Num());
    }
}

bool ULlamaRemoteComponent::LoadImageFileAsPng(const FString& Path, TArray<uint8>& OutPng, FString& OutMime)
{
    OutPng.Reset();
    OutMime.Reset();

    TArray<uint8> FileBytes;
    if (!FFileHelper::LoadFileToArray(FileBytes, *Path)) return false;

    const FString Ext = FPaths::GetExtension(Path).ToLower();
    // Pass-through: common formats the server can decode directly (most vision models accept png/jpeg/webp).
    if (Ext == TEXT("png"))  { OutPng = MoveTemp(FileBytes); OutMime = TEXT("image/png");  return true; }
    if (Ext == TEXT("jpg") || Ext == TEXT("jpeg")) { OutPng = MoveTemp(FileBytes); OutMime = TEXT("image/jpeg"); return true; }
    if (Ext == TEXT("webp")) { OutPng = MoveTemp(FileBytes); OutMime = TEXT("image/webp"); return true; }
    if (Ext == TEXT("gif"))  { OutPng = MoveTemp(FileBytes); OutMime = TEXT("image/gif");  return true; }

    // Unknown format: try to decode and re-encode as PNG via ImageWrapper.
    IImageWrapperModule& IW = FModuleManager::LoadModuleChecked<IImageWrapperModule>(FName("ImageWrapper"));
    const EImageFormat Fmt = IW.DetectImageFormat(FileBytes.GetData(), FileBytes.Num());
    if (Fmt == EImageFormat::Invalid) return false;

    TSharedPtr<IImageWrapper> In = IW.CreateImageWrapper(Fmt);
    if (!In.IsValid() || !In->SetCompressed(FileBytes.GetData(), FileBytes.Num())) return false;
    TArray64<uint8> Raw;
    if (!In->GetRaw(ERGBFormat::RGBA, 8, Raw)) return false;

    TSharedPtr<IImageWrapper> Out = IW.CreateImageWrapper(EImageFormat::PNG);
    if (!Out.IsValid() || !Out->SetRaw(Raw.GetData(), Raw.Num(), In->GetWidth(), In->GetHeight(), ERGBFormat::RGBA, 8))
    {
        return false;
    }
    TArray64<uint8> Compressed = Out->GetCompressed(100);
    OutPng.Append(Compressed.GetData(), Compressed.Num());
    OutMime = TEXT("image/png");
    return OutPng.Num() > 0;
}

void ULlamaRemoteComponent::EncodePcmFloatToWav(const TArray<float>& Pcm, int32 SampleRate, TArray<uint8>& OutWav)
{
    OutWav.Reset();
    const int32 NumSamples = Pcm.Num();
    const int32 NumChannels = 1;
    const int32 BitsPerSample = 16;
    const int32 ByteRate = SampleRate * NumChannels * (BitsPerSample / 8);
    const int32 DataBytes = NumSamples * (BitsPerSample / 8);

    OutWav.Reserve(44 + DataBytes);

    auto Push32 = [&](uint32 V) { for (int i=0;i<4;++i) OutWav.Add(uint8((V >> (8*i)) & 0xFF)); };
    auto Push16 = [&](uint16 V) { OutWav.Add(uint8(V & 0xFF)); OutWav.Add(uint8((V >> 8) & 0xFF)); };
    auto PushStr = [&](const char* S) { while (*S) OutWav.Add(uint8(*S++)); };

    PushStr("RIFF");
    Push32(uint32(36 + DataBytes));
    PushStr("WAVE");
    PushStr("fmt ");
    Push32(16);                  // PCM fmt chunk size
    Push16(1);                   // PCM format
    Push16(uint16(NumChannels));
    Push32(uint32(SampleRate));
    Push32(uint32(ByteRate));
    Push16(uint16(NumChannels * (BitsPerSample / 8)));
    Push16(uint16(BitsPerSample));
    PushStr("data");
    Push32(uint32(DataBytes));

    for (int32 i = 0; i < NumSamples; ++i)
    {
        const float Clamped = FMath::Clamp(Pcm[i], -1.f, 1.f);
        const int16 S = int16(Clamped * 32767.f);
        OutWav.Add(uint8(S & 0xFF));
        OutWav.Add(uint8((S >> 8) & 0xFF));
    }
}
