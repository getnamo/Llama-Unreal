# Mesa Turnip Vulkan driver (optional, Android arm64)

Mesa's open-source Vulkan driver for Qualcomm Adreno GPUs ("Turnip"), built as an Android Vulkan HAL module. It isn't part of the plugin's repository or release; download it with:

```
Scripts\download-turnip-android.bat
```

That puts `arm64-v8a/libvulkan_freedreno.so` here.

- Source: https://github.com/nihui/mesa-turnip-android-driver, release 26.3.0 (`mesa-turnip-android-26.3.0.zip`), built from Mesa 26.3.0 (https://gitlab.freedesktop.org/mesa/mesa).
- SHA-256 of the zip: `6a9bef1c83aa6280e2389d33a992f2d80199d1d13be629c7ff12d5f9f17b5803`; of `libvulkan_freedreno.so`: `bd2ec31af4c54a98880529467291c871aaa2d33bd0487b8fc6afb6e0af2bb632`
- License: MIT (Mesa; the packaging repository is MIT too).

`download-turnip-android.bat <version>` fetches another release tag; only 26.3.0's checksum is verified.

## Why

Qualcomm's own Vulkan driver fails to compile some of llama.cpp's Vulkan shaders on Adreno GPUs (`vk::Device::createComputePipeline: ErrorUnknown`; llama.cpp issue 28635), so without Turnip those devices run llama.cpp on the CPU. Turnip compiles them.

## How it's used

When the file exists, `Source/LlamaCore/LlamaCore_APL.xml` packages it into arm64 APKs. At runtime, `Source/LlamaCore/Private/Android/LlamaAndroidVulkan.cpp` loads it for llama.cpp if it finds a GPU it supports, and otherwise uses the system driver. The engine's renderer always uses the system driver.

Turnip mainly speeds up prompt processing (2.5-4x the CPU on an Adreno 750). Token generation is about even with the CPU on a 4B model and slower on very small ones.

- To leave it out, delete the file.
- To choose a driver at runtime, set the environment variable `LLAMA_VULKAN_DRIVER`: `system` for the device's driver, or a driver library name or path.
