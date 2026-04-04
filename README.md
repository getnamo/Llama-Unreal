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

Everything is wrapped inside a [`ULlamaComponent`](https://github.com/getnamo/Llama-Unreal/blob/ae243df80150b94219911f8a9f36012373336dd9/Source/LlamaCore/Public/LlamaComponent.h#L17) or [`ULlamaSubsystem`](https://github.com/getnamo/Llama-Unreal/blob/ae243df80150b94219911f8a9f36012373336dd9/Source/LlamaCore/Public/LlamaSubsystem.h#L16) which interfaces in a threadsafe manner to llama.cpp code internally via [`FLlamaNative`](https://github.com/getnamo/Llama-Unreal/blob/ae243df80150b94219911f8a9f36012373336dd9/Source/LlamaCore/Public/LlamaNative.h#L14). All core functionality is available both in C++ and in blueprint.

1) In your component or subsystem, adjust your [`ModelParams`](https://github.com/getnamo/Llama-Unreal/blob/ae243df80150b94219911f8a9f36012373336dd9/Source/LlamaCore/Public/LlamaComponent.h#L62) of type [`FLLMModelParams`](https://github.com/getnamo/Llama-Unreal/blob/ae243df80150b94219911f8a9f36012373336dd9/Source/LlamaCore/Public/LlamaDataTypes.h#L208). The most important settings are:
  - `PathToModel` - where your [*.gguf](https://huggingface.co/docs/hub/en/gguf) is placed. If path begins with a . it's considered relative to Saved/Models path, otherwise it's an absolute path.
  - `SystemPrompt` - this will be autoinserted on load by default
  - `MaxContextLength` - this should match your model, default is 4096
  - `GPULayers` - how many layers to offload to GPU. Specifying more layers than the model needs works fine, e.g. use 99 if you want all of them to be offloaded for various practical model sizes. NB: Typically an 8B model will have about 33 layers. Loading more layers will eat up more VRAM, fitting the entire model inside of your target GPU will greatly increase generation speed.

3) Call [`LoadModel`](https://github.com/getnamo/Llama-Unreal/blob/ae243df80150b94219911f8a9f36012373336dd9/Source/LlamaCore/Public/LlamaComponent.h#L78). Consider listening to the [`OnModelLoaded`](https://github.com/getnamo/Llama-Unreal/blob/ae243df80150b94219911f8a9f36012373336dd9/Source/LlamaCore/Public/LlamaComponent.h#L54) callback to deal with post loading operations.

2) Call [`InsertTemplatedPrompt`](https://github.com/getnamo/Llama-Unreal/blob/ae243df80150b94219911f8a9f36012373336dd9/Source/LlamaCore/Public/LlamaComponent.h#L101) with your message and role (typically User) along with whether you want your prompt to generate a response or not. Optionally use [`InsertRawPrompt`](https://github.com/getnamo/Llama-Unreal/blob/ae243df80150b94219911f8a9f36012373336dd9/Source/LlamaCore/Public/LlamaComponent.h#L108) if you're doing raw input style without chat formatting. Note that you can safely chain requests and they will queue up one after another, responses will return in order.

3) You should receive replies via [`OnResponseGenerated`](https://github.com/getnamo/Llama-Unreal/blob/ae243df80150b94219911f8a9f36012373336dd9/Source/LlamaCore/Public/LlamaComponent.h#L36) when full response has been generated. If you need streaming information, listen to [`OnNewTokenGenerated`](https://github.com/getnamo/Llama-Unreal/blob/ae243df80150b94219911f8a9f36012373336dd9/Source/LlamaCore/Public/LlamaComponent.h#L32) and optionally [`OnPartialGenerated`](https://github.com/getnamo/Llama-Unreal/blob/ae243df80150b94219911f8a9f36012373336dd9/Source/LlamaCore/Public/LlamaComponent.h#L40) which will provide token and sentance level streams respectively.

Explore [LlamaComponent.h](https://github.com/getnamo/Llama-Unreal/blob/ae243df80150b94219911f8a9f36012373336dd9/Source/LlamaCore/Public/LlamaComponent.h) for detailed API. Also if you need to modify sampling properties you find them in [`FLLMModelAdvancedParams`](https://github.com/getnamo/Llama-Unreal/blob/ae243df80150b94219911f8a9f36012373336dd9/Source/LlamaCore/Public/LlamaDataTypes.h#L49).


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

## LlamaWhisper Module

Whisper.cpp embedded into the plugin, using the same ggml backend. 

Exposed via `UWhisperComponent` wrapping `FWhisperNative` which can be optionally embedded in your own class instead. The basic api is the following:

1. Add `UWhisperComponent` to your actor of choice. Model defined in `ModelParams` will load on startup, '.' before any path denotes relative to `Saved/Models`. Grab e.g. `ggml-small.en.bin` from  https://huggingface.co/ggerganov/whisper.cpp/tree/main
2. Model will load on begin play, disable `bAutoLoadModelOnStartup` on the component if you wish to load manually.
3. Choose a VAD mode via `StreamParams.VADMode`:
   - **Disabled** - no VAD; audio buffers from `StartMicrophoneCapture` to `StopMicrophoneCapture` and is dispatched as one chunk. If audio exceeds `MaxSpeechSegmentSec` (default 15s) it is auto-chunked with `NonVADOverlapSec` overlap (default 0.5s) - you may need to de-duplicate words at boundaries manually.
   - **Energy-Based (RMS)** *(default)* - lightweight onset/offset detection using an RMS energy threshold. Configurable via `VADThreshold`, `VADHoldTimeSec`, and `VADPreRollSec`. Fast, zero extra model files, works best in quiet environments.
   - **Silero Neural VAD** - neural VAD using a ggml-converted Silero model. More robust in noisy environments. Requires a separate model file pointed to by `StreamParams.PathToVADModel` (default `./ggml-silero-v6.2.0.bin`). The model loads automatically after the whisper model loads. Silero-specific stream params:
     - `SileroThreshold` (default 0.5) - speech probability threshold per window. Lower values are more sensitive; raise to reduce false positives in noisy environments.
     - `SileroHoldTimeSec` (default 0Z.2s) - silence duration before speech offset. Shorter than the EnergyBased default (0.8s) because Silero's neural detection is more precise.

   Download Silero VAD models from:
     - v6: https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v6.2.0.bin

   Place the file in your project's `Saved/Models/` folder and set the path with a leading `.` (e.g. `./ggml-silero-v6.2.0.bin`). Bind `OnVADModelLoaded` to react when the Silero model is ready.

   In all VAD modes, start the microphone with `StartMicrophoneCapture` and stop with `StopMicrophoneCapture`. Any in-progress speech at stop time is always flushed and dispatched.
4. Listen to `OnTranscriptionResult` for transcriptions. Bind `OnVADStateChanged` for speech onset/offset events.
