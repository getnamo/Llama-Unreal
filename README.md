# Llama Unreal

An Unreal focused API wrapper for [llama.cpp](https://github.com/ggerganov/llama.cpp) to support embedding LLMs into your games locally. Forked from [upstream](https://github.com/mika314/UELlama) to focus on improved API with wider support for builds (CPU, CUDA, Android, Mac).

Early releases, api still pretty unstable YMMV.

NB: currently has [#7 issue](https://github.com/getnamo/Llama-Unreal/issues/7) which may require you to do your own static llama.cpp build until resolved.

[Discord Server](https://discord.gg/qfJUyxaW4s)

# Install & Setup

1. [Download Latest Release](https://github.com/getnamo/Llama-Unreal/releases) Ensure to use the `Llama-Unreal-UEx.x-vx.x.x.7z` link which contains compiled binaries, *not* the Source Code (zip) link.
2. Create new or choose desired unreal project.
3. Browse to your project folder (project root)
4. Copy *Plugins* folder from .7z release into your project root.
5. Plugin should now be ready to use.
NB: You may need to manually copy `ggml.dll` and `llama.dll` to your project binaries folder for it to run correctly. (v0.5.0 issue)

# How to use - Basics

NB: Early days of API, unstable.

Everything is wrapped inside a [`ULlamaComponent`](https://github.com/getnamo/Llama-Unreal/blob/5b149eabccd2832fb630bb08f0d9f0c14325ed82/Source/LlamaCore/Public/LlamaComponent.h#L237) which interfaces internally via [`FLlama`](https://github.com/getnamo/Llama-Unreal/blob/5b149eabccd2832fb630bb08f0d9f0c14325ed82/Source/LlamaCore/Private/LlamaComponent.cpp#L87).

1) Setup your [`ModelParams`](https://github.com/getnamo/Llama-Unreal/blob/5b149eabccd2832fb630bb08f0d9f0c14325ed82/Source/LlamaCore/Public/LlamaComponent.h#L273) of type [`FLLMModelParams`](https://github.com/getnamo/Llama-Unreal/blob/5b149eabccd2832fb630bb08f0d9f0c14325ed82/Source/LlamaCore/Public/LlamaComponent.h#L165) 

2) Call [`InsertPromptTemplated`](https://github.com/getnamo/Llama-Unreal/blob/5b149eabccd2832fb630bb08f0d9f0c14325ed82/Source/LlamaCore/Public/LlamaComponent.h#L307) (or [`InsertPrompt`](https://github.com/getnamo/Llama-Unreal/blob/5b149eabccd2832fb630bb08f0d9f0c14325ed82/Source/LlamaCore/Public/LlamaComponent.h#L290) if you're doing raw input style without formatting. NB: only `ChatML` templating is currently specified for templated input.

3) You should receive replies via [`OnNewTokenGenerated`](https://github.com/getnamo/Llama-Unreal/blob/5b149eabccd2832fb630bb08f0d9f0c14325ed82/Source/LlamaCore/Public/LlamaComponent.h#L252) callback

Explore [LlamaComponent.h](https://github.com/getnamo/Llama-Unreal/blob/main/Source/LlamaCore/Public/LlamaComponent.h) for detailed API.


# Llama.cpp Build Instructions

If you want to do builds for your own use case or replace the llama.cpp backend. Note that these build instructions should be run from the cloned llama.cpp root directory, not the plugin root.

Forked Plugin [Llama.cpp](https://github.com/ggerganov/llama.cpp) was built from git hash: [2f3c1466ff46a2413b0e363a5005c46538186ee6](https://github.com/ggerganov/llama.cpp/tree/2f3c1466ff46a2413b0e363a5005c46538186ee6)


### Windows build
With the following build commands for windows (cpu build only, CUDA ignored, see upstream for GPU version):

#### CPU Only

```
mkdir build
cd build/
cmake ..
cmake --build . --config Release -j --verbose
```

#### CUDA

ATM built for CUDA 12.2 runtime

- Use `cuda` branch if you want cuda enabled.
- We build statically due to dll runtime load bug so you need to copy `cudart.lib` `cublas.lib` and `cuda.lib` into your libraries/win64 path. These are ignored atm.
- Ensure `bTryToUseCuda = true;` is set in LlamaCore.build.cs to add CUDA libs to build.
- NB help wanted: Ideally this needs a variant that build with `-DBUILD_SHARED_LIBS=ON`

```
mkdir build
cd build
cmake .. -DGGML_CUDA=ON
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
