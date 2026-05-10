# Llama Unreal

[![GitHub release](https://img.shields.io/github/release/getnamo/Llama-Unreal.svg)](https://github.com/getnamo/Llama-Unreal/releases)
[![Github All Releases](https://img.shields.io/github/downloads/getnamo/Llama-Unreal/total.svg)](https://github.com/getnamo/Llama-Unreal/releases)

An Unreal plugin for [llama.cpp](https://github.com/ggml-org/llama.cpp) to support embedding local LLMs in your projects.

Fork is modern re-write from [upstream](https://github.com/mika314/UELlama) to support latest API, including: GPULayers, advanced sampling (MinP, Miro, etc), Jinja templates, chat history, partial rollback & context reset, regeneration, and more. Defaults to Vulkan build on windows for wider hardware support while retaining very similar performance to CUDA backend (~3% diff) in both prompt processing and token generation speed (as tested on b7285 benchmarks). 


[Discord Server](https://discord.gg/qfJUyxaW4s)

# Install & Setup

1. [Download Latest Release](https://github.com/getnamo/Llama-Unreal/releases) Ensure to use the `Llama-Unreal-UEx.x-vx.x.x.7z` link which contains compiled binaries, *not* the Source Code (zip) link.
2. Create new or choose desired unreal project.
3. Browse to your project folder (project root)
4. Copy *Plugins* folder from .7z release into your project root.

#### Windows
5. Plugin should now be ready to use.

#### Other platforms
5. If platform doesn't have a supported release, build llama.cpp for your platform (see below)
6. Ensure project is mixed type (has c++ and blueprints), compile project (plugin gets compiled with it).


# How to use - Basics

Everything is wrapped inside a [`ULlamaComponent`](Source/LlamaCore/Public/LlamaComponent.h) (per-actor lifetime) or [`ULlamaSubsystem`](Source/LlamaCore/Public/LlamaSubsystem.h) (engine-wide singleton, survives level transitions). Both expose the *same* Blueprint surface: chat, multimodal, embeddings, rollback, impersonation, audio capture wiring. Both can run inference locally via the bundled llama.cpp ([`FLlamaNative`](Source/LlamaCore/Public/LlamaNative.h)) **or** route through an OpenAI-compatible HTTP endpoint by flipping `bUseRemote = true` and configuring `Endpoint.BaseUrl` - see [Remote routing](#remote-routing) below. The shared dual-backend core lives in [`FLlamaDualBackend`](Source/LlamaCore/Public/LlamaDualBackend.h).

1) In your component or subsystem, adjust your [`ModelParams`](https://github.com/getnamo/Llama-Unreal/blob/ae243df80150b94219911f8a9f36012373336dd9/Source/LlamaCore/Public/LlamaComponent.h#L62) of type [`FLLMModelParams`](https://github.com/getnamo/Llama-Unreal/blob/ae243df80150b94219911f8a9f36012373336dd9/Source/LlamaCore/Public/LlamaDataTypes.h#L208). The most important settings are:
  - `PathToModel` - where your [*.gguf](https://huggingface.co/docs/hub/en/gguf) is placed. If path begins with a . it's considered relative to Saved/Models path, otherwise it's an absolute path.
  - `SystemPrompt` - this will be autoinserted on load by default
  - `MaxContextLength` - this should match your model, default is 4096
  - `GPULayers` - how many layers to offload to GPU. Specifying more layers than the model needs works fine, e.g. use 99 if you want all of them to be offloaded for various practical model sizes. NB: Typically an 8B model will have about 33 layers. Loading more layers will eat up more VRAM, fitting the entire model inside of your target GPU will greatly increase generation speed.

3) Call [`LoadModel`](https://github.com/getnamo/Llama-Unreal/blob/ae243df80150b94219911f8a9f36012373336dd9/Source/LlamaCore/Public/LlamaComponent.h#L78). Consider listening to the [`OnModelLoaded`](https://github.com/getnamo/Llama-Unreal/blob/ae243df80150b94219911f8a9f36012373336dd9/Source/LlamaCore/Public/LlamaComponent.h#L54) callback to deal with post loading operations.

2) Call [`InsertTemplatedPrompt`](Source/LlamaCore/Public/LlamaComponent.h) with your message and role (typically User) along with whether you want your prompt to generate a response or not. Optionally use [`InsertRawPrompt`](Source/LlamaCore/Public/LlamaComponent.h) if you're doing raw input style without chat formatting. Note that you can safely chain requests and they will queue up one after another, responses will return in order.

  **Assistant prefill / prepend**: `InsertTemplatedPrompt` and the `FLlamaChatPrompt` struct accept an optional `AssistantPrefill` argument. When non-empty (and `bAddAssistantBOS = true`), the text is inserted into the assistant turn after the BOS header but before sampling - the model continues from it without an intervening end-of-turn token. The prefill is treated as if the model produced it: streamed via `OnTokenGenerated` / `OnPartialGenerated`, returned in `OnResponseGenerated`, and stored in chat history. Useful for steering first-token behavior (`"Answer: "`) or for hard-suppressing thinking on a thinking-capable model (`"<think></think>\n\n"`). Currently a local-only feature; a warning is emitted in remote mode.

3) You should receive replies via [`OnResponseGenerated`](https://github.com/getnamo/Llama-Unreal/blob/ae243df80150b94219911f8a9f36012373336dd9/Source/LlamaCore/Public/LlamaComponent.h#L36) when full response has been generated. If you need streaming information, listen to [`OnNewTokenGenerated`](https://github.com/getnamo/Llama-Unreal/blob/ae243df80150b94219911f8a9f36012373336dd9/Source/LlamaCore/Public/LlamaComponent.h#L32) and optionally `OnPartialGenerated` (sentence-level) or `OnMarkdownPartialGenerated` (formatting-aware partials tagged with `EMarkdownStreamState`: Text, Italic, Bold, Heading, Quote, Emphasis, **Thinking**). Markdown emission requires `Advanced.Markdown.bSplitMarkdown = true`. Thinking-capable models (Qwen3, DeepSeek-R1) auto-route content between `<think>...</think>` tags into the `Thinking` category - tag chars are stripped, content is delivered separately so you can route it to a "thinking" UI panel.

`OnPartialGenerated` fires per-sentence using `Advanced.Output.PartialsSeparators`, which by default covers `.` `?` `!` `\n` `…` plus CJK (`。 ？ ！`), Devanagari danda (`।`), and Arabic question mark (`؟`). The matcher reads `Sep[0]` so only single-character entries are effective - replace or extend the array if your content needs different break points (clauses on `;` `:` etc.).

Explore [LlamaComponent.h](https://github.com/getnamo/Llama-Unreal/blob/ae243df80150b94219911f8a9f36012373336dd9/Source/LlamaCore/Public/LlamaComponent.h) for detailed API. Also if you need to modify sampling properties you find them in [`FLLMModelAdvancedParams`](https://github.com/getnamo/Llama-Unreal/blob/ae243df80150b94219911f8a9f36012373336dd9/Source/LlamaCore/Public/LlamaDataTypes.h#L49).

### Restoring history (save-game / external state)

Call `RebuildContextFromHistory(FStructuredChatHistory)` to wipe the model's KV cache and re-ingest a saved conversation. The model's KV state is rebuilt so the next prompt continues correctly. State-only fallback is used when no native backend is available (e.g. running purely remote).


# Remote routing

The plugin is local-first, but every `ULlamaComponent` and `ULlamaSubsystem` can route inference through an OpenAI-compatible HTTP endpoint (e.g. [llama-server](https://github.com/ggml-org/llama.cpp/tree/master/tools/server), LM Studio, Ollama, vLLM, OpenAI itself) by setting `bUseRemote = true`. The shared dual-backend ([`FLlamaDualBackend`](Source/LlamaCore/Public/LlamaDualBackend.h)) keeps a local `FLlamaNative` and a remote HTTP client side-by-side and routes each call to whichever is active. All delegates (`OnTokenGenerated`, `OnResponseGenerated`, `OnPartialGenerated`, `OnMarkdownPartialGenerated`, `OnEndOfStream`) fire on both paths; same chat history, same multimodal entry points, same rollback helpers.

## Setup

1. Set `Endpoint.BaseUrl` to your server (e.g. `http://127.0.0.1:8080`). The server must expose `/v1/chat/completions`, `/health`, and `/props`.
2. Call `SetUseRemote(true)` (or tick the `bUseRemote` checkbox before `LoadModel`).
3. `LoadModel` runs a `/health` probe and `/props` fetch (model id, chat template, modality capabilities). Fires `OnModelLoaded` on success.
4. Use the same API as local: `InsertTemplatedPrompt`, `InsertMultimodalPrompt`, `StopGeneration`, `ResetContextHistory`, `RemoveLastAssistantReply`, etc.

```cpp
Component->Endpoint.BaseUrl = TEXT("http://127.0.0.1:8080");
Component->SetUseRemote(true);
Component->LoadModel();
Component->InsertTemplatedPrompt(TEXT("Hello"), EChatTemplateRole::User);
```

The plugin auto-translates `ModelParams` into the request JSON: `Advanced.Sampling.*` → `temperature` / `top_p` / `top_k` / `min_p` / `repeat_penalty` / `mirostat*` / etc., `StopSequences` → `stop`, `Seed` → `seed`. Llama-server extensions used: `cache_prompt: true` (KV prefix reuse via stateful slots), `id_slot` for slot reuse, `chat_template_kwargs.enable_thinking` for Qwen3-style thinking control.

## Toggling between local and remote at runtime

Both backends coexist. `SetUseRemote(bool)` flips routing live - any active stream is cancelled cleanly, the destination backend auto-loads if its config is valid (`Endpoint.BaseUrl` for remote, `ModelParams.PathToModel` for local), and the chat history is lazily synced on the next prompt insertion.

```cpp
Component->SetUseRemote(false);  // switch to local FLlamaNative for the next prompt
```

`bUseRemote` defaults to `false` (local-first). The remote-side properties (`Endpoint`, etc.) are hidden in the editor until the toggle is on.

**Smart KV sync on remote → local handoff.** When you swap back to local with messages added during the remote excursion, the plugin tracks how many messages the local KV last decoded and hashes that prefix. If the prefix still matches the current chat history, only the new messages are appended via `InsertTemplatedPrompt(bGenerateReply=false)` - no full replay. If the prefix has diverged (e.g. you reset history while remote, or rolled back a turn), it falls back to a full `RebuildContextFromHistory`. Toggle via `bUseIncrementalKVSyncOnToggle` (default `true`); flip to `false` for the bulletproof full-rebuild path on every sync if the smart path ever surfaces a KV bug in your environment.

**Model file is *not* reloaded** on toggle. Once the local backend's model file is in VRAM it stays resident across remote excursions, and `IsModelLoaded()` correctly reports per-backend so the lazy-load on swap-back skips reload when the local is already warm. KV cache rebuild (above) is the only cost. To free local VRAM during long remote sessions, call `UnloadModel()` explicitly before `SetUseRemote(true)`.

**Warm-loading the local backend.** If you want zero handoff cost - i.e. instant local takeover on `SetUseRemote(false)` - set `bPreloadLocalWhenRemote = true` (default `false`). When `LoadModel` runs in remote mode this also kicks off a silent local model load in the background. Trade: full model resident from startup. Useful when you expect to swap mid-session and can't tolerate the multi-second initial load latency.

## Multimodal over HTTP

Image and audio prompts use the same `InsertTemplateImagePrompt` / `InsertTemplateAudioPrompt` / `InsertMultimodalPrompt` calls. The remote path encodes media as data-URLs in OpenAI content parts (`image_url` for images, `input_audio` for audio). The server must have an mmproj loaded and report the modality in `/props`.

## Slot caching & stateless fallback

The first request lets the server auto-assign a slot; subsequent requests reuse `AssignedSlotId` so only the incremental delta is reprocessed. `ResetContextHistory` issues a best-effort `/slots/{id}?action=erase`. If the server doesn't support slot ops (vanilla OpenAI API), the component silently downgrades to stateless mode (full `messages[]` per request).

# Multimodal (Vision & Audio)

The plugin supports multimodal models - LLMs that can process images and/or audio alongside text - using the `mtmd` library bundled with llama.cpp.

## Supported Models

Any vision or audio model available in GGUF format that ships a separate multimodal projector file (`mmproj`). Tested with:
- **Qwen2.5-Omni** (vision + audio)

Models and projectors are available on Hugging Face in their respective GGUF repositories.

## Setup Requirements

### 1. Model Files

You need two GGUF files per multimodal model:

| File | Purpose |
|---|---|
| `model.gguf` | The base language model - same as any text-only LLM |
| `mmproj-model-f16.gguf` (or similar) | The multimodal projector that encodes images/audio into token embeddings |

Place both in your `Saved/Models` folder (or any absolute path).

### 2. ModelParams Configuration

Set `MmprojPath` in `FLLMModelParams` before calling `LoadModel`. Paths beginning with `.` are relative to `Saved/Models`:

```
ModelParams.PathToModel  = "./Qwen2.5-Omni-7B-Q4_K_M.gguf"
ModelParams.MmprojPath   = "./mmproj-Qwen2.5-Omni-7B-Q8_0.gguf"
```

If `MmprojPath` is empty, multimodal is disabled and the model runs as text-only.

### 3. Build Requirements (custom builds only)

If building llama.cpp from source you must also build and include the `mtmd` target:

```
cmake --build . --config Release --target mtmd -j
```

Then copy alongside the other libs/dlls:
- `{build root}/tools/mtmd/Release/mtmd.lib` → `ThirdParty/LlamaCpp/Lib/Win64/`
- `{build root}/bin/Release/mtmd.dll` → `ThirdParty/LlamaCpp/Binaries/Win64/`

And the headers:
- `{llama.cpp root}/tools/mtmd/mtmd.h`
- `{llama.cpp root}/tools/mtmd/mtmd-helper.h`

→ `ThirdParty/LlamaCpp/Include/mtmd/`

## How to Use

### Capability Checks

Before making multimodal calls, verify the projector loaded and the model supports the desired modality:

```
IsMultimodalLoaded()   // projector loaded successfully
SupportsVision()       // model can process images
SupportsAudio()        // model can process audio
GetAudioSampleRate()   // expected PCM sample rate (typically 16000 Hz)
```

Calling a multimodal function without a loaded projector fires `OnError` with code **50** - no crash.

### Image Prompts

**From a UTexture2D** (e.g. a render target or imported asset, must be `PF_B8G8R8A8` format):

```
InsertTemplateImagePrompt(MyTexture, "What is in this image?")
```

**From a file path on disk** (more efficient - avoids GPU readback):

```
InsertTemplateImagePromptFromFile("C:/Images/photo.jpg", "Describe this scene.")
```

Both functions accept `Role`, `bAddAssistantBOS`, and `bGenerateReply` parameters matching the text API.

### Audio Prompts

Audio must be provided as mono float PCM at the model's expected sample rate (use `GetAudioSampleRate()` to check - typically 16 kHz). Use `ULlamaAudioUtils` to convert Unreal `USoundWave` assets:

```
// One-shot convenience: converts SoundWave → 16 kHz mono float PCM
TArray<float> PCM;
ULlamaAudioUtils::SoundWaveToLLMAudio(MySoundWave, PCM, GetAudioSampleRate());

// Then pass to the component
InsertTemplateAudioPrompt(PCM, "Transcribe this audio.")
```

`ULlamaAudioUtils` also exposes lower-level steps if you need finer control:
- `SoundWaveToPCMFloat` - raw PCM decode (returns source sample rate and channel count)
- `PCMFloatToMono` - stereo/multichannel → mono downmix
- `ResamplePCMFloat` - arbitrary sample rate conversion

### Multiple Media in One Message

Use `InsertMultimodalPrompt` with an `FLlamaMultimodalPrompt` struct to place multiple images or audio clips in a single message. Each `<__media__>` marker in the prompt text corresponds to one `FLlamaMediaEntry` in `MediaEntries` (matched in order):

```
FLlamaMultimodalPrompt P;
P.Prompt = "Image A: <__media__>\nImage B: <__media__>\nCompare these two images.";
P.MediaEntries = { EntryA, EntryB };
P.bGenerateReply = true;
InsertMultimodalPrompt(P);
```

If the prompt contains no `<__media__>` markers and `MediaEntries` has exactly one entry, the marker is auto-prepended.

### Multi-Message Composition

Build up context across multiple calls using `bGenerateReply = false`, then trigger generation on the final call - works the same as the text-only API:

```
// Insert image without generating
InsertTemplateImagePrompt(ImageA, "First image:", User, false, false)
InsertTemplateImagePrompt(ImageB, "Second image:", User, false, false)
// Generate on final text-only message
InsertTemplatedPrompt("Now compare those two images.", User)
```

## Error Codes

Multimodal errors are delivered through the existing `OnError` delegate:

| Code | Condition |
|---|---|
| 50 | Multimodal projector not loaded (`MmprojPath` empty or init failed) |
| 51 | `<__media__>` marker count doesn't match `MediaEntries` count |
| 52 | Invalid bitmap (null texture, unsupported pixel format, failed file load) |
| 53 | `mtmd_tokenize` failed |
| 54 | `mtmd_helper_eval_chunks` failed (eval error during image/audio ingestion) |
| 55 | Vision not supported by the loaded mmproj |
| 56 | Audio not supported by the loaded mmproj |

## Known Limitations

- **Context rollback:** `RollbackContextHistoryByMessages` does not correctly account for the variable token count of multimodal messages (image token count depends on resolution). Use `ResetContextHistory` to clear context after multimodal sessions instead.
- **Texture format:** `InsertTemplateImagePrompt` only supports `PF_B8G8R8A8` textures. Use `InsertTemplateImagePromptFromFile` to load other formats directly via the mtmd file decoder.
- **Audio sample rate:** The caller is responsible for providing PCM at the model's expected rate. Use `GetAudioSampleRate()` and `ULlamaAudioUtils::ResamplePCMFloat` to convert if needed.

---

# RAG: Embeddings, Vector DB & Hybrid Search

The plugin ships with a complete local RAG stack: an embedding pipeline, an HNSW vector index, a model-free BM25 inverted index, and an RRF hybrid retriever - all running in-process, no external services.

> **Module note:** the RAG stack lives in its own module, **`LlamaTools`** (sibling to `LlamaCore` and `LlamaWhisper`), under [`Source/LlamaTools/`](Source/LlamaTools/). C++ consumers must add `"LlamaTools"` to their `*.Build.cs` `PublicDependencyModuleNames` to access `URagStore`, `FVectorDatabase`, `FBM25Index`, etc. The embedding *backend* (the raw `GetPromptEmbeddings` / `GetEmbeddingDimension` calls on `ULlamaComponent` and `ULlamaSubsystem`) is in `LlamaCore` so any consumer can compute embeddings without pulling the full RAG stack - `URagStore` is what *consumes* that backend. Removing the `LlamaTools/` directory leaves `LlamaCore` (and `LlamaWhisper`) unaffected.

## Quick start

The `URagStore` (UObject) and `URagStoreComponent` (UActorComponent) are self-contained: they own their own embedding model and, optionally, their own answer model. The minimal flow:

1. Configure two model paths:
   - `EmbeddingModelParams.PathToModel = "./bge-small-en-v1.5-q4_k_m.gguf"` (or any embedding GGUF - `bEmbeddingMode` is force-set at load time).
   - `AnswerModelParams.PathToModel = "./google_gemma-3-4b-it-Q4_K_L.gguf"` (or any chat GGUF you'd run via `ULlamaComponent`).
2. Drop a `URagStoreComponent` on your actor. With `bAutoInitializeOnBeginPlay = true` (default), `BeginPlay` calls `LoadModels()` and auto-`Initialize()`s once the embedder reports its dimension. For non-actor flows, `NewObject<URagStore>()` and call `LoadModels()` + `Initialize()` yourself.
3. Ingest content: `IngestText(text, source)`, `IngestFile(path)`, `IngestDocuments(texts, sources)`, or `IngestDirectory(folder, "txt,md", recursive)`. `OnIngestComplete(int32 Added)` fires when done.
4. **Ask in one call**: bind `OnAskTokenGenerated`/`OnAskPartialGenerated`/`OnAskResponseGenerated` and call `AskDefault("your question")`. The store retrieves top-K chunks, formats them with `SummarizingPromptTemplate` (overridable; ships with a sensible default that uses `{context}` and `{query}` placeholders), and streams the answer through the same `OnAsk*` delegates regardless of which answer pathway is configured. `AnswerPrefill` (default `"Answer: "`) is applied to the assistant turn before sampling - see [Assistant prefill](#how-to-use---basics) - and works around the Gemma3 first-token-EOT quirk; set to `"<think></think>\n\n"` to hard-suppress thinking on a thinking-capable model, or empty for raw generation.
5. Or get chunks directly: `RetrieveAsync(query, params)` returns `TArray<FLlamaChunk>` with `Confidence` (0..1), `RetrievalScore` (raw, retriever-specific), and `SourceRetriever` (`Vector` / `BM25` / `Hybrid`) populated. `Params.MinConfidence` pre-filters the tail.
6. Persist with `SaveToFile(Path)` / `LoadFromFile(Path)`. A single `.rag` file bundles vectors + BM25 index + chunk metadata.

### Power-user paths

- **Share an embedder across multiple stores** to save VRAM: load one `ULlamaComponent` in embedding mode and assign it to each store's `ExternalEmbedder`. The internal embedder is skipped when `ExternalEmbedder` is set.
- **Route answers through an existing chat component** (e.g. an in-game NPC `ULlamaComponent`): leave `AnswerModelParams.PathToModel` empty and assign the component to `AnswerEngine`. The store wires `OnAsk*` relays to its broadcasts and gates on a `bAskInFlight` flag so unrelated chat from the same component doesn't leak into Ask events.
- **Score-aware filtering**: each retrieved chunk carries `Confidence` ∈ [0,1] (top-1 always 1.0; lower = lower-quality match relative to top-1) and the raw `RetrievalScore` (L2 distance for vector, BM25 score, RRF score for hybrid). Set `FRagRetrievalParams::MinConfidence = 0.5` to drop chunks less than half as good as the best, etc. Top-1 always survives the filter so a query never returns blank.

## Components

- **`FVectorDatabase`** ([VectorDatabase.h](Source/LlamaTools/Public/Embedding/VectorDatabase.h)) - HNSW (hnswlib) ANN with L2 metric. Works as cosine when input is L2-normalized, which `GetPromptEmbeddings` does by default. `UVectorDatabase` is the Blueprint-callable wrapper.
- **`FBM25Index`** ([BM25Index.h](Source/LlamaTools/Public/Embedding/BM25Index.h)) - Lexical retrieval with BM25+ IDF; tokenizer is model-free (Unicode-aware lowercase + alphanumeric split + ASCII stopword filter).
- **`FHybridRetriever`** ([HybridRetriever.h](Source/LlamaTools/Public/Embedding/HybridRetriever.h)) - Reciprocal Rank Fusion (k=60) of the dense and sparse ranks; parameter-free across heterogeneous score scales.
- **`FLlamaCorpusChunker`** ([CorpusChunker.h](Source/LlamaTools/Public/Embedding/CorpusChunker.h)) - Deterministic paragraph + sliding-window chunker with sentence-boundary snapping.
- **`URagStore`** ([RagStore.h](Source/LlamaTools/Public/Embedding/RagStore.h)) - Composes the above; self-contained two-model pipeline (embedder + optional answerer); `Ask()` for end-to-end retrieve+answer with streaming.
- **`URagStoreComponent`** ([RagStoreComponent.h](Source/LlamaTools/Public/Embedding/RagStoreComponent.h)) - Actor-component wrapper: same surface, auto-init on BeginPlay, BP-friendly delegate chain.

## Recommended embedding models (GGUF)

| Model | Dim | Size | Use case |
|---|---|---|---|
| `bge-small-en-v1.5-q4_k_m` | 384 | ~33 MB | CI/test fixture; tiny, strong on English MTEB |
| `nomic-embed-text-v1.5-q4_k_m` | 768 | ~85 MB | General-purpose default |
| `Qwen3-Embedding-0.6B-q8_0` | 1024 | ~600 MB | Multilingual, modern |

A fetch script for the test fixture lives at [`Source/LlamaTools/Private/Tests/fetch_models.ps1`](Source/LlamaTools/Private/Tests/fetch_models.ps1).

---

# LlamaWhisper Module

Whisper.cpp embedded into the plugin, using the same ggml backend. 

Exposed via `UWhisperComponent` wrapping `FWhisperNative` which can be optionally embedded in your own class instead. The basic api is the following:

1. Add `UWhisperComponent` to your actor of choice. Model defined in `ModelParams` will load on startup, '.' before any path denotes relative to `Saved/Models`. Grab e.g. `ggml-small.en.bin` from  https://huggingface.co/ggerganov/whisper.cpp/tree/main
2. Model will load on begin play, disable `bAutoLoadModelOnStartup` on the component if you wish to load manually.
3. Choose a VAD mode via `StreamParams.VADMode`:
   - **Disabled** - no VAD; audio buffers from `StartMicrophoneCapture` to `StopMicrophoneCapture` and is dispatched as one chunk. If audio exceeds `MaxSpeechSegmentSec` (default 15s) it is auto-chunked with `NonVADOverlapSec` overlap (default 0.5s) - you may need to de-duplicate words at boundaries manually.
   - **Energy-Based (RMS)** *(default)* - lightweight onset/offset detection using an RMS energy threshold. Configurable via `VADThreshold`, `VADHoldTimeSec`, and `VADPreRollSec`. Fast, zero extra model files, works best in quiet environments.
   - **Silero Neural VAD** - neural VAD using a ggml-converted Silero model. More robust in noisy environments. Requires a separate model file pointed to by `StreamParams.PathToVADModel` (default `./ggml-silero-v6.2.0.bin`). The model loads automatically after the whisper model loads. Silero-specific stream params:
     - `SileroThreshold` (default 0.5) - speech probability threshold per window. Lower values are more sensitive; raise to reduce false positives in noisy environments.
     - `SileroHoldTimeSec` (default 0.2s) - silence duration before speech offset. Shorter than the EnergyBased default (0.8s) because Silero's neural detection is more precise.

   Download Silero VAD models from:
     - v6: https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v6.2.0.bin

   Place the file in your project's `Saved/Models/` folder and set the path with a leading `.` (e.g. `./ggml-silero-v6.2.0.bin`). Bind `OnVADModelLoaded` to react when the Silero model is ready.

   In all VAD modes, start the microphone with `StartMicrophoneCapture` and stop with `StopMicrophoneCapture`. Any in-progress speech at stop time is always flushed and dispatched.
4. Listen to `OnTranscriptionResult` for transcriptions. Bind `OnVADStateChanged` for speech onset/offset events.


# Automation Tests

The plugin ships with two automation buckets matching the module split:

- **`LlamaCore`** - backend-only tests (no RAG, no model files needed): chat-history UTF-8 round-trip, dual-backend prefix-hash and frontier-invalidation logic, path resolver, etc.
- **`LlamaTools`** - RAG stack: `VectorDatabase` (HNSW round-trip + ordering + persistence + dimension guards), `BM25` (query/save/load), `Chunker` determinism, `RAG.HybridRRF`, `RAG.ScoreNormalization*`, plus the model-gated end-to-end tests (`RAG.AskPipeline`, `RAG.IngestDirectoryWalk`, `RAG.QwenThinkingDiagnostic`) that load a real embedder + answer model from `Saved/Models/`.

Run headless with:

```
UnrealEditor-Cmd.exe <project name>.uproject -ExecCmds="Automation RunTests LlamaCore+LlamaTools;Quit" -unattended -nullrhi -nopause -log
```

Use `LlamaCore` or `LlamaTools` alone to run a single bucket. The model-gated `LlamaTools.RAG.*` tests skip cleanly with an `AddInfo` log when the corresponding GGUFs are not present - fetch them with [`Source/LlamaTools/Private/Tests/fetch_models.ps1`](Source/LlamaTools/Private/Tests/fetch_models.ps1) (see the RAG section above).

# Note on speed

If you're running the inference in a high spec game fully loaded into the same GPU that renders the game, expect about ~1/3-1/2 of the performance due to resource contention; e.g. an 8B model running at ~90TPS might have ~40TPS speed in game. You may want to use a smaller model or [apply pressure easing strategies](https://github.com/getnamo/Llama-Unreal/blob/main/Source/LlamaCore/Public/LlamaDataTypes.h#L133) to manage perfectly stable framerates.

# Llama.cpp Build Instructions

To do custom backends or support platforms not currently supported you can follow these build instruction. Note that these build instructions should be run from the cloned llama.cpp root directory, not the plugin root.

SN: curl issues: https://github.com/ggml-org/llama.cpp/issues/9937

### Basic Build Steps
1. clone [Llama.cpp](https://github.com/ggml-org/llama.cpp)
2. build using commands given below e.g. for Vulkan
```
mkdir build
cd build/
cmake .. -DGGML_VULKAN=ON -DGGML_NATIVE=OFF
cmake --build . --config Release -j --verbose
```

also in newer builds consider

```cmake .. -DGGML_VULKAN=ON -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF -DCMAKE_CXX_FLAGS_RELEASE="/Zi"```

to workaround CURL and generate .pdbs for debugging


3. Include: After build 
- Copy `{llama.cpp root}/include`
- Copy `{llama.cpp root}/ggml/include`
- into `{plugin root}/ThirdParty/LlamaCpp/Include`
- Copy `{llama.cpp root}/common/` `common.h` and `sampling.h`
- into `{plugin root}/ThirdParty/LlamaCpp/Include/common`
- *(Multimodal)* Copy `{llama.cpp root}/tools/mtmd/mtmd.h` and `mtmd-helper.h`
- *(Multimodal)* into `{plugin root}/ThirdParty/LlamaCpp/Include/mtmd`

4. Libs: Assuming `{llama.cpp root}/build` as `{build root}`.

- Copy `{build root}/src/Release/llama.lib`,
- Copy `{build root}/common/Release/common.lib`,
- Copy `{build root}/ggml/src/Release/` `ggml.lib`, `ggml-base.lib` & `ggml-cpu.lib`,
- Copy `{build root}/ggml/src/Release/ggml-vulkan/Release/ggml-vulkan.lib`
- *(Multimodal)* Copy `{build root}/tools/mtmd/Release/mtmd.lib`
- into `{plugin root}/ThirdParty/LlamaCpp/Lib/Win64`

5. Dlls:
- Copy `{build root}/bin/Release/` `ggml.dll`, `ggml-base.dll`, `ggml-cpu.dll`, `ggml-vulkan.dll`, & `llama.dll`
- *(Multimodal)* Copy `{build root}/bin/Release/mtmd.dll`
- into `{plugin root}/ThirdParty/LlamaCpp/Binaries/Win64`
6. Build plugin

### Current Version
Current Plugin [Llama.cpp](https://github.com/ggml-org/llama.cpp) was built from git has/tag: [b8586](https://github.com/ggml-org/llama.cpp/releases/tag/b8586)

NB: use `-DGGML_NATIVE=OFF` to ensure wider portability.


### Windows build
With the following build commands for windows.

#### CPU Only

```
mkdir build
cd build/
cmake .. -DGGML_NATIVE=OFF
cmake --build . --config Release -j --verbose
```
#### Vulkan

see https://github.com/ggml-org/llama.cpp/blob/b4762/docs/build.md#git-bash-mingw64

e.g. once [Vulkan SDK](https://vulkan.lunarg.com/sdk/home#windows) has been installed run.

```
mkdir build
cd build/
cmake .. -DGGML_VULKAN=ON -DGGML_NATIVE=OFF
cmake --build . --config Release -j --verbose
```

#### CUDA

ATM CUDA 12.4 runtime is recommended.

- Ensure `bTryToUseCuda = true;` is set in LlamaCore.build.cs to add CUDA libs to build (untested in v0.9 update)

```
mkdir build
cd build
cmake .. -DGGML_CUDA=ON -DGGML_NATIVE=OFF
cmake --build . --config Release -j --verbose
```

### Linux build (cross-compile from Windows)

Cross-compile llama.cpp using UE 5.7's bundled clang toolchain (`v26_clang-20.1.8-rockylinux8`) so the resulting `.so` files have the same glibc/libstdc++ ABI as UBT's Linux output. This produces binaries that run on Rocky 8 / Ubuntu 20.04+ / Debian 11+ (glibc ≥ 2.28).

**Prerequisites (Windows host):**
1. UE 5.7 Linux toolchain - install `v26_clang-20.1.8-rockylinux8` from [Epic's Linux requirements page](https://dev.epicgames.com/documentation/en-us/unreal-engine/linux-development-requirements-for-unreal-engine?application_version=5.7). Sets `LINUX_MULTIARCH_ROOT`. UE 5.7 strictly requires this exact version (declared in `Engine/Config/Linux/Linux_SDK.json`); Note: UBT expects `LINUX_MULTIARCH_ROOT` to point at the **specific toolchain dir** (e.g. `C:\<toolchains root>\v26_clang-20.1.8-rockylinux8\`), not the parent.
2. **UE Linux engine platform component** - in Epic Games Launcher: Library → UE 5.7 → small box icon → Options → enable "Engine Platforms → Linux" → Apply (~1-2 GB download). UBT refuses Linux targets without this even when the toolchain is installed.
3. Vulkan SDK from https://vulkan.lunarg.com/sdk/home (sets `VULKAN_SDK`). Only the headers are needed at build time; `libvulkan.so.1` is `dlopen`'d at runtime on the Linux side.
4. **libvulkan in the cross sysroot** - `find_package(Vulkan)` wants a libvulkan.so to satisfy the linker (ggml-vulkan dlopens at runtime, but CMake still checks). Easiest source: install `libvulkan-dev` in WSL2, then copy the `.so.1.x.y` file into the v26 toolchain at `<toolchain>/x86_64-unknown-linux-gnu/usr/lib64/` and create plain-file copies (NOT symlinks - WSL2 creates Windows junctions that clang.exe can't follow) named `libvulkan.so` and `libvulkan.so.1`.
5. **MSVC** (Visual Studio 2022 Build Tools or Community) - the `vulkan-shaders-gen` sub-build is a *host* (Windows) tool built at CMake time. Without MSVC on PATH, CMake auto-picks mingw gcc which can't compile native Windows binaries. The build helper script invokes `VsDevCmd.bat` to set this up automatically.
6. A modern llama.cpp checkout (b9090 or compatible).

**Critical: libc++ ABI alignment.** UE on Linux compiles against libc++ (clang's `std::__1::vector`), not libstdc++ (`std::__cxx11::vector`). The toolchain file (`cmake/ue57-linux-cross.cmake`) sets `-stdlib=libc++` for the llama.cpp build so its `common_*` helper symbols mangle to match what UE's compile expects. Without this, plugin link fails with `undefined reference to common_batch_add(llama_batch&, ..., std::__1::vector<int>...)`. libc++ is static-only in this toolchain (no `libc++.so`), so each `.so` embeds its own copy - larger binaries (~+2-5 MB each), but ABI-clean.

**One-shot build via the bundled helper:**
```cmd
cd <plugin>\Scripts
build-llamacpp-linux.bat <path\to\llama.cpp>
```
The script copies the cmake toolchain file into the llama.cpp clone, configures, builds, and stages `.so` artifacts under `<plugin>\ThirdParty\LlamaCpp\Lib\Linux\`.

**Manual invocation** (equivalent, for when you want to inspect intermediate state):
```cmd
cd <path\to\llama.cpp>
copy <plugin>\Scripts\..\..\..\<plugin>\cmake\ue57-linux-cross.cmake cmake\ue57-linux-cross.cmake

cmake -S . -B build-linux-vulkan -G Ninja ^
  -DCMAKE_TOOLCHAIN_FILE=cmake/ue57-linux-cross.cmake ^
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON ^
  -DGGML_VULKAN=ON -DGGML_NATIVE=OFF ^
  -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF ^
  -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TOOLS=OFF ^
  -DVulkan_INCLUDE_DIR=%VULKAN_SDK%\Include
cmake --build build-linux-vulkan --config Release -j
```

The toolchain file (`cmake/ue57-linux-cross.cmake` in the llama.cpp clone) bakes `$ORIGIN` rpath into every `.so` so `libllama.so` finds `libggml.so` etc. in the same staged directory at runtime.

**Artifacts to stage** (into `<plugin>\ThirdParty\LlamaCpp\Lib\Linux\`):

7 unique libraries, each staged twice - as `libfoo.so` (linker resolves `-llama` at link time) and as `libfoo.so.0` (loader resolves SONAME references between sibling libs at runtime). 14 files total. The build helper script handles this; copy real files, **not** symlinks (WSL2 emits Windows junctions, the loader doesn't follow them).

| Library | Purpose |
|---|---|
| `libllama.so` | llama API |
| `libggml.so` | ggml dispatcher |
| `libggml-base.so` | ggml core |
| `libggml-cpu.so` | CPU backend |
| `libggml-vulkan.so` | Vulkan backend (optional but recommended) |
| `libllama-common.so` | common helpers (Win64's `llama-common-base.lib` is consolidated into this single shared lib on Linux because `BUILD_SHARED_LIBS=ON` merges static sub-archives into the parent - no separate `libllama-common-base.so`) |
| `libmtmd.so` | multimodal helper |

The `.so` files are linked with `-Wl,-rpath,$ORIGIN` (via `CMAKE_INSTALL_RPATH=$ORIGIN` in the toolchain file), so once staged in the same directory at runtime they find each other without `LD_LIBRARY_PATH`.

On Linux a single `.so` serves as both link target and runtime artifact, so this is the only directory needed - no `Lib/` + `Binaries/` split like Windows.

**Cross-compile the plugin (Linux game target):**
```cmd
"<UE5.7>\Engine\Build\BatchFiles\Build.bat" <ProjectName> Linux Development -Project=<full-path>\<project name>.uproject
```
UBT picks up the toolchain via `LINUX_MULTIARCH_ROOT` and links against the staged `.so` files; `RuntimeDependencies` copies them next to the produced ELF binary.

**Runtime verification (WSL2):**

Set up Vulkan tools inside WSL2 once:
```bash
sudo apt update && sudo apt install -y vulkan-tools mesa-vulkan-drivers libvulkan1
vulkaninfo --summary    # should list a Vulkan device
```

**Sanity check at the lib layer** - confirms ABI/linkage works without UE in the loop:
```bash
# Tiny C program calling llama_backend_init + llama_print_system_info
gcc -o smoke smoke.c -I<plugin>/ThirdParty/LlamaCpp/Include \
    -L<plugin>/ThirdParty/LlamaCpp/Lib/Linux \
    -Wl,-rpath,<plugin>/ThirdParty/LlamaCpp/Lib/Linux -lllama
./smoke   # expect: "system_info: CPU : LLAMAFILE = 1 | REPACK = 1 | ..."
```

**Plugin layer verification** - `ldd` against the cross-built binary confirms every staged `.so` resolves at runtime:
```bash
cd /mnt/c/<path-to-project>/Binaries/Linux
chmod +x <ProjectName>
ldd ./<ProjectName>
# Expect to see libllama.so.0, libggml*.so.0, libllama-common.so.0, libmtmd.so.0,
# libggml-vulkan.so.0 all resolving to ./<lib>.so.0 entries (via $ORIGIN rpath).
```

**End-to-end automation tests** - running the cross-built game binary directly against `Automation RunTests` requires cooked Linux content (asset registry, shader formats), which a bare cross-compile doesn't produce. The proper path is:

1. **Full UAT BuildCookRun**: `RunUAT.bat BuildCookRun -project=<...>.uproject -platform=Linux -clientconfig=Development -build -cook -stage -unattended` produces a complete staged build under `Saved/StagedBuilds/Linux/`. The staged binary can then run `Automation RunTests` cleanly.
2. **Linux editor inside WSL2**: a UE source build native to Linux, paired with `<ProjectName>Editor` for Linux. Heavy, but enables the full automation suite (`LlamaCore.UTF8.*`, `LlamaCore.RAG.*`, `LlamaTools.*` - all currently `EditorContext`-gated).

For the goal of "verify the plugin compiles and loads on Linux", the lib-layer smoke test + `ldd` check are sufficient acceptance evidence. UE-side automation on Linux is a downstream packaging exercise.

### Mac build

```
mkdir build
cd build/
cmake .. -DBUILD_SHARED_LIBS=ON
cmake --build . --config Release -j --verbose
```

### Android build

For Android build see: https://github.com/ggerganov/llama.cpp/blob/master/docs/android.md#cross-compile-using-android-ndk

```
mkdir build-android
cd build-android
export NDK=<your_ndk_directory>
cmake -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-23 -DCMAKE_C_FLAGS=-march=armv8.4a+dotprod ..
$ make
```

Then the .so or .lib file was copied into e.g. `ThirdParty/LlamaCpp/Win64/cpu` directory and all the .h files were copied to the `Includes` directory.
