// Copyright 2025-current Getnamo.

#include "LlamaDualBackend.h"

#include "LlamaNative.h"
#include "LlamaUtility.h"
#include "Remote/LlamaRemoteClient.h"

#include "Engine/Texture2D.h"
#include "TextureResource.h"
#include "IImageWrapper.h"
#include "IImageWrapperModule.h"
#include "Modules/ModuleManager.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"

#include "Dom/JsonObject.h"

// ─── Lifecycle ───────────────────────────────────────────────────────────────

FLlamaDualBackend::FLlamaDualBackend() = default;
// Note: PartialsSeparators are intentionally NOT defaulted here — the host
// (ULlamaComponent / ULlamaSubsystem) populates them on its own ModelParams so the
// per-Insert sync `Backend->ModelParams = ModelParams` carries valid defaults instead
// of clobbering them.

FLlamaDualBackend::~FLlamaDualBackend()
{
    Shutdown();
}

void FLlamaDualBackend::Initialize()
{
    if (!LlamaNative)
    {
        LlamaNative = new FLlamaNative();
        HookNativeCallbacks();
    }
    if (!Client.IsValid())
    {
        Client = MakeUnique<FLlamaRemoteClient>();
    }
}

void FLlamaDualBackend::Shutdown()
{
    if (ActiveStream.IsValid())
    {
        FLlamaRemoteClient::CancelStream(ActiveStream);
        ActiveStream.Reset();
    }
    if (LlamaNative)
    {
        delete LlamaNative;
        LlamaNative = nullptr;
    }
    Client.Reset();
}

void FLlamaDualBackend::OnGameThreadTick(float DeltaTime)
{
    if (LlamaNative)
    {
        LlamaNative->OnGameThreadTick(DeltaTime);
    }
}

void FLlamaDualBackend::HookNativeCallbacks()
{
    if (!LlamaNative) return;

    LlamaNative->OnModelStateChanged = [this](const FLLMModelState& Updated)
    {
        ModelState = Updated;
        if (OnModelStateChanged) OnModelStateChanged(Updated);
    };
    LlamaNative->OnTokenGenerated = [this](const FString& Token)
    {
        if (OnTokenGenerated) OnTokenGenerated(Token);
    };
    LlamaNative->OnPartialGenerated = [this](const FString& Partial)
    {
        if (OnPartialGenerated) OnPartialGenerated(Partial);
    };
    LlamaNative->OnMarkdownPartialGenerated = [this](const FString& Partial, EMarkdownStreamState S)
    {
        if (OnMarkdownPartialGenerated) OnMarkdownPartialGenerated(Partial, S);
    };
    LlamaNative->OnResponseGenerated = [this](const FString& Response)
    {
        if (OnResponseGenerated) OnResponseGenerated(Response);
        if (OnEndOfStream) OnEndOfStream(true, ModelState.LastTokenGenerationSpeed);
    };
    LlamaNative->OnPromptProcessed = [this](int32 Tokens, EChatTemplateRole Role, float Speed)
    {
        if (OnPromptProcessed) OnPromptProcessed(Tokens, Role, Speed);
    };
    LlamaNative->OnError = [this](const FString& Err, int32 Code)
    {
        if (OnError) OnError(Err, Code);
    };
}

// ─── Backend toggle ──────────────────────────────────────────────────────────

void FLlamaDualBackend::SetUseRemote(bool bNew)
{
    if (bNew == bUseRemote) return;

    StopGeneration();

    if (bUseRemote && AssignedSlotId >= 0 && Client.IsValid())
    {
        Client->EraseSlot(Endpoint.BaseUrl, AssignedSlotId, Endpoint.ConnectTimeoutSeconds, nullptr);
        AssignedSlotId = -1;
    }

    bUseRemote = bNew;

    // Auto-load destination if its config looks valid and it isn't already loaded.
    if (bUseRemote)
    {
        if (Endpoint.BaseUrl.IsEmpty())
        {
            UE_LOG(LlamaLog, Warning, TEXT("SetUseRemote(true): Endpoint.BaseUrl empty; skipping auto-load."));
        }
        else if (!bRemoteModelLoaded)
        {
            LoadModel(/*bForceReload=*/false);
        }
    }
    else
    {
        if (ModelParams.PathToModel.IsEmpty())
        {
            UE_LOG(LlamaLog, Warning, TEXT("SetUseRemote(false): ModelParams.PathToModel empty; skipping auto-load."));
        }
        else if (LlamaNative && !ModelState.bModelIsLoaded)
        {
            LoadModel(/*bForceReload=*/false);
        }
    }

    bPendingHistorySync = (ModelState.ChatHistory.History.Num() > 0);
}

void FLlamaDualBackend::FlushPendingHistorySyncIfNeeded()
{
    if (!bPendingHistorySync) return;

    if (bUseRemote)
    {
        bPendingHistorySync = false;
        return;
    }
    if (!LlamaNative || !ModelState.bModelIsLoaded) return;

    LlamaNative->RebuildContextFromHistory(ModelState.ChatHistory);
    bPendingHistorySync = false;
}

// ─── Model load / unload ─────────────────────────────────────────────────────

void FLlamaDualBackend::LoadModel(bool bForceReload)
{
    if (!bUseRemote)
    {
        if (!LlamaNative) return;
        LlamaNative->SetModelParams(ModelParams);
        LlamaNative->LoadModel(bForceReload, [this](const FString& ModelPath, int32 StatusCode)
        {
            if (StatusCode != 0) return; // OnError already fired by FLlamaNative
            if (OnModelLoaded) OnModelLoaded(ModelPath);
        });
        return;
    }

    if (!Client.IsValid())
    {
        if (OnError) OnError(TEXT("Remote client not initialized; call Initialize first"), 60);
        return;
    }

    Client->Health(Endpoint.BaseUrl, Endpoint.ConnectTimeoutSeconds,
        [this](bool bOk, const FString& Err)
        {
            if (!bOk)
            {
                if (OnError) OnError(FString::Printf(TEXT("remote unreachable: %s"), *Err), 61);
                return;
            }
            Client->FetchProps(Endpoint.BaseUrl, Endpoint.ConnectTimeoutSeconds,
                [this](bool bPropsOk, TSharedPtr<FJsonObject> Root)
                {
                    RemoteModelName = TEXT("remote");
                    bRemoteVision = false;
                    bRemoteAudio = false;
                    bRemoteSupportsThinking = false;
                    RemoteAudioSampleRate = 16000;

                    if (bPropsOk && Root.IsValid())
                    {
                        FString ModelPath;
                        if (Root->TryGetStringField(TEXT("model_path"), ModelPath) ||
                            Root->TryGetStringField(TEXT("model"), ModelPath))
                        {
                            RemoteModelName = FPaths::GetCleanFilename(ModelPath);
                        }

                        FString ChatTemplate;
                        if (Root->TryGetStringField(TEXT("chat_template"), ChatTemplate))
                        {
                            ModelState.ChatTemplateInUse.TemplateSource = ChatTemplate;
                            bRemoteSupportsThinking =
                                ChatTemplate.Contains(TEXT("<think>")) ||
                                ChatTemplate.Contains(TEXT("enable_thinking"));
                        }

                        const TArray<TSharedPtr<FJsonValue>>* Modalities = nullptr;
                        if (Root->TryGetArrayField(TEXT("modalities"), Modalities))
                        {
                            for (const TSharedPtr<FJsonValue>& V : *Modalities)
                            {
                                const FString M = V->AsString().ToLower();
                                if (M == TEXT("vision") || M == TEXT("image")) bRemoteVision = true;
                                if (M == TEXT("audio")) bRemoteAudio = true;
                            }
                        }
                        else
                        {
                            const TSharedPtr<FJsonObject>* Caps = nullptr;
                            if (Root->TryGetObjectField(TEXT("modalities"), Caps) && (*Caps).IsValid())
                            {
                                bool bV = false, bA = false;
                                (*Caps)->TryGetBoolField(TEXT("vision"), bV);
                                (*Caps)->TryGetBoolField(TEXT("audio"), bA);
                                bRemoteVision = bV;
                                bRemoteAudio = bA;
                            }
                        }
                    }

                    // Seed system prompt into history if requested and not already present.
                    if (ModelParams.bAutoInsertSystemPromptOnLoad && !ModelParams.SystemPrompt.IsEmpty())
                    {
                        const bool bHasSystem = ModelState.ChatHistory.History.ContainsByPredicate(
                            [](const FStructuredChatMessage& M){ return M.Role == EChatTemplateRole::System; });
                        if (!bHasSystem)
                        {
                            FStructuredChatMessage Sys;
                            Sys.Role = EChatTemplateRole::System;
                            Sys.Content = ModelParams.SystemPrompt;
                            ModelState.ChatHistory.History.Insert(Sys, 0);
                        }
                    }

                    // Note: ModelState.bModelIsLoaded is authoritatively owned by FLlamaNative
                    // (synced via OnModelStateChanged) — don't touch it from the remote path or
                    // a remote→local toggle will see a stale "loaded" flag and skip the auto-load.
                    bRemoteModelLoaded = true;
                    if (OnModelLoaded) OnModelLoaded(RemoteModelName);
                });
        });
}

void FLlamaDualBackend::UnloadModel()
{
    if (!bUseRemote)
    {
        if (!LlamaNative) return;
        LlamaNative->UnloadModel([this](int32 StatusCode)
        {
            if (StatusCode != 0)
            {
                const FString Msg = FString::Printf(TEXT("UnloadModel returned %d"), StatusCode);
                if (OnError) OnError(Msg, StatusCode);
            }
        });
        return;
    }

    if (ActiveStream.IsValid())
    {
        FLlamaRemoteClient::CancelStream(ActiveStream);
        ActiveStream.Reset();
    }
    bRemoteModelLoaded = false;
    AssignedSlotId = -1;
    // Do NOT touch ModelState.bModelIsLoaded — that's the local backend's authoritative flag.
}

bool FLlamaDualBackend::IsModelLoaded() const
{
    return bUseRemote ? bRemoteModelLoaded : ModelState.bModelIsLoaded;
}

// ─── Chat ────────────────────────────────────────────────────────────────────

void FLlamaDualBackend::InsertTemplatedPrompt(const FLlamaChatPrompt& Prompt)
{
    FlushPendingHistorySyncIfNeeded();
    if (!bUseRemote)
    {
        if (!LlamaNative) return;
        LlamaNative->InsertTemplatedPrompt(Prompt);
        return;
    }
    AppendUserMessage(Prompt.Prompt, Prompt.Role);
    if (Prompt.bGenerateReply) BeginStreamFromHistory(true);
}

void FLlamaDualBackend::InsertRawPrompt(const FString& Text, bool bGenerateReply)
{
    FlushPendingHistorySyncIfNeeded();
    if (!bUseRemote)
    {
        if (!LlamaNative) return;
        LlamaNative->InsertRawPrompt(Text, bGenerateReply);
        return;
    }
    if (!bRemoteModelLoaded)
    {
        if (OnError) OnError(TEXT("remote model not loaded; call LoadModel first"), 62);
        return;
    }
    if (ActiveStream.IsValid())
    {
        if (OnError) OnError(TEXT("a stream is already in flight; call StopGeneration first"), 63);
        return;
    }

    FLlamaRemoteChatRequest Req;
    Req.Endpoint = Endpoint;
    Req.Params = ModelParams;
    Req.SlotId = (Endpoint.PreferredSlotId >= 0) ? Endpoint.PreferredSlotId : AssignedSlotId;
    Req.bUseRawCompletion = true;
    Req.RawPrompt = Text;

    if (!bGenerateReply) return;

    PartialBuffer.Reset();

    ActiveStream = Client->StreamChat(Req,
        [this](const FString& Delta) { HandleIncomingDelta(Delta); },
        [this](const FString& Final, int32 SlotId, float Tps)
        {
            if (SlotId >= 0) AssignedSlotId = SlotId;
            FlushPendingPartial(true);
            ModelState.LastTokenGenerationSpeed = Tps;
            if (OnResponseGenerated) OnResponseGenerated(Final);
            if (OnEndOfStream) OnEndOfStream(true, Tps);
            ActiveStream.Reset();
        },
        [this](const FString& Err, int32 Code)
        {
            if (OnError) OnError(Err, Code);
            ActiveStream.Reset();
        });
}

void FLlamaDualBackend::StopGeneration()
{
    if (!bUseRemote)
    {
        if (LlamaNative) LlamaNative->StopGeneration();
        return;
    }
    if (ActiveStream.IsValid())
    {
        FLlamaRemoteClient::CancelStream(ActiveStream);
        ActiveStream.Reset();
        FlushPendingPartial(true);
        if (OnEndOfStream) OnEndOfStream(false, ModelState.LastTokenGenerationSpeed);
    }
}

void FLlamaDualBackend::ResumeGeneration()
{
    if (!bUseRemote)
    {
        if (LlamaNative) LlamaNative->ResumeGeneration();
        return;
    }
    if (!ActiveStream.IsValid())
    {
        BeginStreamFromHistory(false);
    }
}

void FLlamaDualBackend::ResetContextHistory(bool bKeepSystemPrompt)
{
    if (!bUseRemote)
    {
        if (LlamaNative) LlamaNative->ResetContextHistory(bKeepSystemPrompt);
        if (OnContextReset) OnContextReset();
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

    if (AssignedSlotId >= 0 && Client.IsValid())
    {
        const int32 OldSlot = AssignedSlotId;
        AssignedSlotId = -1;
        Client->EraseSlot(Endpoint.BaseUrl, OldSlot, Endpoint.ConnectTimeoutSeconds, nullptr);
    }

    if (OnContextReset) OnContextReset();
}

void FLlamaDualBackend::RebuildContextFromHistory(const FStructuredChatHistory& History)
{
    if (!ShouldBypassNativeKV() && LlamaNative)
    {
        LlamaNative->RebuildContextFromHistory(History);
        return;
    }
    ModelState.ChatHistory = History;
    if (ModelState.ChatHistory.History.Num() > 0)
    {
        ModelState.LastRole = ModelState.ChatHistory.History.Last().Role;
    }
    if (OnPromptProcessed) OnPromptProcessed(0, ModelState.LastRole, 0.f);

    // Remote doesn't need a KV rebuild — chat history is sent in every request.
    // Mark the slot stale so cache_prompt rebuilds from scratch on next call.
    if (bUseRemote && AssignedSlotId >= 0 && Client.IsValid())
    {
        const int32 Old = AssignedSlotId;
        AssignedSlotId = -1;
        Client->EraseSlot(Endpoint.BaseUrl, Old, Endpoint.ConnectTimeoutSeconds, nullptr);
    }
}

void FLlamaDualBackend::RemoveLastReply()
{
    if (ShouldBypassNativeKV())
    {
        const int32 Count = ModelState.ChatHistory.History.Num();
        if (Count > 0)
        {
            ModelState.ChatHistory.History.RemoveAt(Count - 1);
        }
        return;
    }
    LlamaNative->RemoveLastReply();
}

void FLlamaDualBackend::RemoveLastUserInput()
{
    if (ShouldBypassNativeKV())
    {
        const int32 Count = ModelState.ChatHistory.History.Num();
        if (Count > 1)
        {
            ModelState.ChatHistory.History.RemoveAt(Count - 1);
            ModelState.ChatHistory.History.RemoveAt(Count - 2);
        }
        return;
    }
    LlamaNative->RemoveLastUserInput();
}

void FLlamaDualBackend::RemoveLastNTokens(int32 N)
{
    // Token-precise rollback only makes sense locally; remote has no per-token control.
    if (LlamaNative && !bUseRemote)
    {
        LlamaNative->RemoveLastNTokens(N);
    }
}

void FLlamaDualBackend::ImpersonateTemplatedPrompt(const FLlamaChatPrompt& Prompt)
{
    if (!LlamaNative) return;
    LlamaNative->SetModelParams(ModelParams);
    LlamaNative->ImpersonateTemplatedPrompt(Prompt);
}

void FLlamaDualBackend::ImpersonateTemplatedToken(const FString& Token, EChatTemplateRole Role, bool bEoS)
{
    if (!LlamaNative) return;
    LlamaNative->ImpersonateTemplatedToken(Token, Role, bEoS);
}

FString FLlamaDualBackend::WrapPromptForRole(const FString& Text, EChatTemplateRole Role, const FString& OverrideTemplate)
{
    if (!LlamaNative) return Text;
    return LlamaNative->WrapPromptForRole(Text, Role, OverrideTemplate);
}

// ─── Multimodal ──────────────────────────────────────────────────────────────

void FLlamaDualBackend::InsertTemplateImagePromptFromTexture(UTexture2D* Image, const FString& Text,
    EChatTemplateRole Role, bool bAddAssistantBOS, bool bGenerateReply)
{
    FlushPendingHistorySyncIfNeeded();

    if (!bUseRemote)
    {
        if (!LlamaNative) { if (OnError) OnError(TEXT("No native backend"), 60); return; }
        if (!Image || !Image->GetPlatformData() || Image->GetPlatformData()->Mips.Num() == 0)
        {
            if (OnError) OnError(TEXT("Invalid texture for image prompt"), 52);
            return;
        }
        if (!LlamaNative->IsMultimodalLoaded())
        {
            if (OnError) OnError(TEXT("Multimodal projector not loaded — set MmprojPath in ModelParams"), 50);
            return;
        }
        if (!LlamaNative->SupportsVision())
        {
            if (OnError) OnError(TEXT("Vision not supported by loaded model"), 55);
            return;
        }

        FTexturePlatformData* PD = Image->GetPlatformData();
        const int32 W = PD->SizeX;
        const int32 H = PD->SizeY;
        const void* Raw = PD->Mips[0].BulkData.LockReadOnly();
        TArray<uint8> RGB;
        RGB.SetNum(W * H * 3);
        const uint8* Src = static_cast<const uint8*>(Raw);
        for (int32 i = 0; i < W * H; ++i)
        {
            RGB[i*3 + 0] = Src[i*4 + 2];
            RGB[i*3 + 1] = Src[i*4 + 1];
            RGB[i*3 + 2] = Src[i*4 + 0];
        }
        PD->Mips[0].BulkData.Unlock();

        FLlamaMultimodalPrompt Prompt;
        Prompt.Prompt = Text.Contains(TEXT("<__media__>")) ? Text : FString::Printf(TEXT("<__media__>\n%s"), *Text);
        Prompt.Role = Role;
        Prompt.bAddAssistantBOS = bAddAssistantBOS;
        Prompt.bGenerateReply = bGenerateReply;
        FLlamaMediaEntry Entry;
        Entry.MediaType = ELlamaMediaType::Image;
        Entry.ImageRGBData = MoveTemp(RGB);
        Entry.ImageWidth = W;
        Entry.ImageHeight = H;
        Prompt.MediaEntries.Add(MoveTemp(Entry));

        LlamaNative->InsertMultimodalPrompt(Prompt);
        return;
    }

    if (!Image)
    {
        if (OnError) OnError(TEXT("Invalid texture for image prompt"), 52);
        return;
    }
    FLlamaRemoteMediaBlob Blob;
    EncodeTextureToPng(Image, Blob.Bytes, Blob.Mime);
    if (Blob.Bytes.Num() == 0)
    {
        if (OnError) OnError(TEXT("Failed to PNG-encode texture"), 53);
        return;
    }
    Blob.bIsImage = true;
    PendingUserMedia.Add(MoveTemp(Blob));

    AppendUserMessage(Text, Role);
    if (bGenerateReply) BeginStreamFromHistory(true);
}

void FLlamaDualBackend::InsertTemplateImagePromptFromFile(const FString& ImagePath, const FString& Text,
    EChatTemplateRole Role, bool bAddAssistantBOS, bool bGenerateReply)
{
    FlushPendingHistorySyncIfNeeded();

    if (!bUseRemote)
    {
        if (!LlamaNative) { if (OnError) OnError(TEXT("No native backend"), 60); return; }
        if (!LlamaNative->IsMultimodalLoaded())
        {
            if (OnError) OnError(TEXT("Multimodal projector not loaded — set MmprojPath in ModelParams"), 50);
            return;
        }
        if (!LlamaNative->SupportsVision())
        {
            if (OnError) OnError(TEXT("Vision not supported by loaded model"), 55);
            return;
        }
        FLlamaMultimodalPrompt Prompt;
        Prompt.Prompt = Text.Contains(TEXT("<__media__>")) ? Text : FString::Printf(TEXT("<__media__>\n%s"), *Text);
        Prompt.Role = Role;
        Prompt.bAddAssistantBOS = bAddAssistantBOS;
        Prompt.bGenerateReply = bGenerateReply;
        FLlamaMediaEntry Entry;
        Entry.MediaType = ELlamaMediaType::Image;
        Entry.FilePath = ImagePath;
        Prompt.MediaEntries.Add(MoveTemp(Entry));
        LlamaNative->InsertMultimodalPrompt(Prompt);
        return;
    }

    FLlamaRemoteMediaBlob Blob;
    Blob.bIsImage = true;
    if (!LoadImageFileAsPng(ImagePath, Blob.Bytes, Blob.Mime))
    {
        if (OnError) OnError(FString::Printf(TEXT("Failed to read/encode image: %s"), *ImagePath), 54);
        return;
    }
    PendingUserMedia.Add(MoveTemp(Blob));

    AppendUserMessage(Text, Role);
    if (bGenerateReply) BeginStreamFromHistory(true);
}

void FLlamaDualBackend::InsertTemplateAudioPrompt(const TArray<float>& PCMAudio, const FString& Text,
    EChatTemplateRole Role, bool bAddAssistantBOS, bool bGenerateReply)
{
    FlushPendingHistorySyncIfNeeded();

    if (!bUseRemote)
    {
        if (!LlamaNative) { if (OnError) OnError(TEXT("No native backend"), 60); return; }
        if (!LlamaNative->IsMultimodalLoaded())
        {
            if (OnError) OnError(TEXT("Multimodal projector not loaded — set MmprojPath in ModelParams"), 50);
            return;
        }
        if (!LlamaNative->SupportsAudio())
        {
            if (OnError) OnError(TEXT("Audio not supported by loaded model"), 56);
            return;
        }

        FLlamaMultimodalPrompt Prompt;
        Prompt.Prompt = Text.Contains(TEXT("<__media__>")) ? Text : FString::Printf(TEXT("<__media__>\n%s"), *Text);
        Prompt.Role = Role;
        Prompt.bAddAssistantBOS = bAddAssistantBOS;
        Prompt.bGenerateReply = bGenerateReply;
        FLlamaMediaEntry Entry;
        Entry.MediaType = ELlamaMediaType::Audio;
        Entry.AudioPCMData = PCMAudio;
        Prompt.MediaEntries.Add(MoveTemp(Entry));
        LlamaNative->InsertMultimodalPrompt(Prompt);
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

void FLlamaDualBackend::InsertMultimodalPrompt(const FLlamaMultimodalPrompt& Prompt)
{
    FlushPendingHistorySyncIfNeeded();

    if (!bUseRemote)
    {
        if (!LlamaNative) { if (OnError) OnError(TEXT("No native backend"), 60); return; }
        if (!LlamaNative->IsMultimodalLoaded())
        {
            if (OnError) OnError(TEXT("Multimodal projector not loaded"), 50);
            return;
        }
        LlamaNative->InsertMultimodalPrompt(Prompt);
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
                    if (OnError) OnError(FString::Printf(TEXT("Failed to read image: %s"), *Entry.FilePath), 54);
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
                    Dst[i*4 + 0] = Src[i*3 + 0];
                    Dst[i*4 + 1] = Src[i*3 + 1];
                    Dst[i*4 + 2] = Src[i*3 + 2];
                    Dst[i*4 + 3] = 255;
                }
                IImageWrapperModule& IW = FModuleManager::LoadModuleChecked<IImageWrapperModule>(FName("ImageWrapper"));
                TSharedPtr<IImageWrapper> PNG = IW.CreateImageWrapper(EImageFormat::PNG);
                if (PNG.IsValid() && PNG->SetRaw(Rgba.GetData(), Rgba.Num(), Entry.ImageWidth, Entry.ImageHeight, ERGBFormat::RGBA, 8))
                {
                    TArray64<uint8> Compressed = PNG->GetCompressed(100);
                    Blob.Bytes.Append(Compressed.GetData(), Compressed.Num());
                    Blob.Mime = TEXT("image/png");
                }
            }
            if (Blob.Bytes.Num() == 0)
            {
                if (OnError) OnError(TEXT("Empty image entry in multimodal prompt"), 55);
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
                    if (OnError) OnError(FString::Printf(TEXT("Failed to read audio: %s"), *Entry.FilePath), 57);
                    return;
                }
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

bool FLlamaDualBackend::IsMultimodalLoaded() const
{
    return bUseRemote ? (bRemoteVision || bRemoteAudio) : (LlamaNative && LlamaNative->IsMultimodalLoaded());
}
bool FLlamaDualBackend::SupportsVision() const
{
    return bUseRemote ? bRemoteVision : (LlamaNative && LlamaNative->SupportsVision());
}
bool FLlamaDualBackend::SupportsAudio() const
{
    return bUseRemote ? bRemoteAudio : (LlamaNative && LlamaNative->SupportsAudio());
}
int32 FLlamaDualBackend::GetAudioSampleRate() const
{
    return bUseRemote ? RemoteAudioSampleRate : (LlamaNative ? LlamaNative->GetAudioSampleRate() : 16000);
}

// ─── Embedding ───────────────────────────────────────────────────────────────

void FLlamaDualBackend::GeneratePromptEmbeddingsForText(const FString& Text)
{
    if (!LlamaNative) return;
    if (!ModelParams.Advanced.bEmbeddingMode)
    {
        UE_LOG(LlamaLog, Warning, TEXT("Model not in embedding mode; cannot generate embeddings."));
        return;
    }
    LlamaNative->GetPromptEmbeddings(Text, [this](const TArray<float>& Embeddings, const FString& Source)
    {
        if (OnEmbeddings) OnEmbeddings(Embeddings, Source);
    });
}

void FLlamaDualBackend::GeneratePromptEmbeddingsForTexts(const TArray<FString>& Texts)
{
    if (!LlamaNative) return;
    if (!ModelParams.Advanced.bEmbeddingMode)
    {
        UE_LOG(LlamaLog, Warning, TEXT("Model not in embedding mode; cannot generate embeddings."));
        return;
    }
    LlamaNative->GetPromptEmbeddingsBatch(Texts,
        [this](const TArray<float>& Embeddings, const FString& Source)
        {
            if (OnEmbeddings) OnEmbeddings(Embeddings, Source);
        },
        [this](const TArray<TArray<float>>& /*All*/, const TArray<FString>& Sources)
        {
            if (OnAllEmbeddingsGenerated) OnAllEmbeddingsGenerated(Sources);
        });
}

int32 FLlamaDualBackend::GetEmbeddingDimension() const
{
    return LlamaNative ? LlamaNative->GetEmbeddingDimension() : 0;
}

void FLlamaDualBackend::EmbedTextsAsync(const TArray<FString>& Texts,
    TFunction<void(const TArray<TArray<float>>&, const TArray<FString>&)> OnDone)
{
    if (!LlamaNative)
    {
        if (OnDone) OnDone(TArray<TArray<float>>(), TArray<FString>());
        return;
    }
    if (!ModelParams.Advanced.bEmbeddingMode)
    {
        UE_LOG(LlamaLog, Warning, TEXT("EmbedTextsAsync: model not in embedding mode."));
        if (OnDone) OnDone(TArray<TArray<float>>(), TArray<FString>());
        return;
    }
    LlamaNative->GetPromptEmbeddingsBatch(Texts, /*per-item*/ nullptr,
        [OnDone = MoveTemp(OnDone)](const TArray<TArray<float>>& All, const TArray<FString>& Sources)
        {
            if (OnDone) OnDone(All, Sources);
        });
}

// ─── Audio consumer ──────────────────────────────────────────────────────────

void FLlamaDualBackend::OnAudioSegment(const FLlamaAudioSegment& Segment)
{
    // Local mode: forward to FLlamaNative which builds a multimodal prompt internally.
    if (!bUseRemote && LlamaNative)
    {
        LlamaNative->AudioPromptTemplate = AudioPromptTemplate;
        LlamaNative->AudioPromptRole = AudioPromptRole;
        LlamaNative->OnAudioSegment(Segment);
        return;
    }

    // Remote mode: encode and dispatch.
    if (bUseRemote)
    {
        FLlamaRemoteMediaBlob Blob;
        Blob.bIsImage = false;
        Blob.Mime = TEXT("audio/wav");
        EncodePcmFloatToWav(Segment.PCMSamples, RemoteAudioSampleRate, Blob.Bytes);
        PendingUserMedia.Add(MoveTemp(Blob));

        const FString PromptText = AudioPromptTemplate.Contains(TEXT("<__media__>"))
            ? AudioPromptTemplate
            : FString::Printf(TEXT("<__media__>\n%s"), *AudioPromptTemplate);

        AppendUserMessage(PromptText, AudioPromptRole);
        BeginStreamFromHistory(true);
    }
}

// ─── Internals ───────────────────────────────────────────────────────────────

void FLlamaDualBackend::AppendUserMessage(const FString& Content, EChatTemplateRole Role)
{
    FStructuredChatMessage Msg;
    Msg.Role = Role;
    Msg.Content = Content;
    ModelState.ChatHistory.History.Add(MoveTemp(Msg));
    ModelState.LastRole = Role;
    if (OnPromptProcessed) OnPromptProcessed(0, Role, 0.f);
}

void FLlamaDualBackend::BeginStreamFromHistory(bool bAttachPendingMedia)
{
    if (!bRemoteModelLoaded)
    {
        if (OnError) OnError(TEXT("remote model not loaded; call LoadModel first"), 62);
        return;
    }
    if (ActiveStream.IsValid())
    {
        if (OnError) OnError(TEXT("a stream is already in flight; call StopGeneration first"), 63);
        return;
    }
    if (!Client.IsValid())
    {
        if (OnError) OnError(TEXT("Remote client not initialized"), 60);
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

    // Pre-append empty Assistant message for live streaming append.
    FStructuredChatMessage Assistant;
    Assistant.Role = EChatTemplateRole::Assistant;
    Assistant.Content = FString();
    ModelState.ChatHistory.History.Add(Assistant);

    PartialBuffer.Reset();
    MdSplitter.Reset();

    ActiveStream = Client->StreamChat(Req,
        [this](const FString& Delta) { HandleIncomingDelta(Delta); },
        [this](const FString& Final, int32 SlotId, float Tps)
        {
            if (SlotId >= 0) AssignedSlotId = SlotId;
            FlushPendingPartial(true);

            if (ModelParams.Advanced.Markdown.bSplitMarkdown)
            {
                TArray<TPair<FString, EMarkdownStreamState>> Final_;
                MdSplitter.Collect(Final_, ModelParams.Advanced.Markdown);
                MdSplitter.Reset();
                for (const auto& P : Final_)
                {
                    if (!P.Key.IsEmpty() && OnMarkdownPartialGenerated)
                    {
                        OnMarkdownPartialGenerated(P.Key, P.Value);
                    }
                }
            }

            if (ModelState.ChatHistory.History.Num() > 0)
            {
                FStructuredChatMessage& Last = ModelState.ChatHistory.History.Last();
                if (Last.Role == EChatTemplateRole::Assistant)
                {
                    Last.Content = Final;
                }
            }

            FString Emitted = Final;
            if (ModelParams.Advanced.Thinking.bStripThinkingFromResponse && bRemoteSupportsThinking)
            {
                const FString CloseTag = TEXT("</think>");
                int32 ClosePos = Emitted.Find(CloseTag, ESearchCase::CaseSensitive, ESearchDir::FromStart);
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

            ModelState.LastTokenGenerationSpeed = Tps;
            if (OnResponseGenerated) OnResponseGenerated(Emitted);
            if (OnEndOfStream) OnEndOfStream(true, Tps);
            ActiveStream.Reset();
        },
        [this](const FString& Err, int32 Code)
        {
            // Drop the optimistic placeholder.
            if (ModelState.ChatHistory.History.Num() > 0 &&
                ModelState.ChatHistory.History.Last().Role == EChatTemplateRole::Assistant &&
                ModelState.ChatHistory.History.Last().Content.IsEmpty())
            {
                ModelState.ChatHistory.History.Pop();
            }
            if (OnError) OnError(Err, Code);
            ActiveStream.Reset();
        });
}

void FLlamaDualBackend::HandleIncomingDelta(const FString& Delta)
{
    if (Delta.IsEmpty()) return;

    if (OnTokenGenerated) OnTokenGenerated(Delta);

    if (ModelState.ChatHistory.History.Num() > 0)
    {
        FStructuredChatMessage& Last = ModelState.ChatHistory.History.Last();
        if (Last.Role == EChatTemplateRole::Assistant)
        {
            Last.Content += Delta;
        }
    }

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
                if (!P.Key.IsEmpty() && OnMarkdownPartialGenerated)
                {
                    OnMarkdownPartialGenerated(P.Key, P.Value);
                }
            }
        }
    }

    if (!ModelParams.Advanced.Output.bEmitPartials) return;

    PartialBuffer += Delta;
    const TArray<FString>& Seps = ModelParams.Advanced.Output.PartialsSeparators;
    if (Seps.Num() == 0) return;

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
        if (!Chunk.IsEmpty() && OnPartialGenerated)
        {
            OnPartialGenerated(Chunk);
        }
        break;
    }
}

void FLlamaDualBackend::FlushPendingPartial(bool bForceEmitRemainder)
{
    if (bForceEmitRemainder && !PartialBuffer.IsEmpty())
    {
        FString Tail = PartialBuffer;
        PartialBuffer.Reset();
        Tail.TrimStartAndEndInline();
        if (!Tail.IsEmpty() && ModelParams.Advanced.Output.bEmitPartials && OnPartialGenerated)
        {
            OnPartialGenerated(Tail);
        }
    }
}

// ─── Encoding helpers ────────────────────────────────────────────────────────

void FLlamaDualBackend::EncodeTextureToPng(UTexture2D* Image, TArray<uint8>& OutPng, FString& OutMime)
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

bool FLlamaDualBackend::LoadImageFileAsPng(const FString& Path, TArray<uint8>& OutPng, FString& OutMime)
{
    OutPng.Reset();
    OutMime.Reset();

    TArray<uint8> FileBytes;
    if (!FFileHelper::LoadFileToArray(FileBytes, *Path)) return false;

    const FString Ext = FPaths::GetExtension(Path).ToLower();
    if (Ext == TEXT("png"))  { OutPng = MoveTemp(FileBytes); OutMime = TEXT("image/png");  return true; }
    if (Ext == TEXT("jpg") || Ext == TEXT("jpeg")) { OutPng = MoveTemp(FileBytes); OutMime = TEXT("image/jpeg"); return true; }
    if (Ext == TEXT("webp")) { OutPng = MoveTemp(FileBytes); OutMime = TEXT("image/webp"); return true; }
    if (Ext == TEXT("gif"))  { OutPng = MoveTemp(FileBytes); OutMime = TEXT("image/gif");  return true; }

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

void FLlamaDualBackend::EncodePcmFloatToWav(const TArray<float>& Pcm, int32 SampleRate, TArray<uint8>& OutWav)
{
    OutWav.Reset();
    const int32 NumSamples  = Pcm.Num();
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
    Push32(16);
    Push16(1);
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
