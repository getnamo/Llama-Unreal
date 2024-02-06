# Getnamo Fork notes

Cuda variant
- atm built for CUDA 12.2 runtime

# Install & Setup

1. [Download Latest Release](https://github.com/getnamo/Llama-Unreal/releases) Ensure to use the `Llama-Unreal-UEx.x-vx.x.x.7z` link which contains compiled binaries, *not* the Source Code (zip) link.
2. Create new or choose desired unreal project.
3. Browse to your project folder (project root)
4. Copy *Plugins* folder from .7z release into your project root.
5. Plugin should now be ready to use.

# Llama.cpp Build Instructions

If you want to do builds for your own use case or replace the llama.cpp backend. Note that these build instructions should be run from the cloned llama.cpp root directory, not the plugin root.

Forked Plugin [Llama.cpp](https://github.com/ggerganov/llama.cpp) was built from git hash: [b7e7982953f80a656e03feb5cfb17a17a173eb26](https://github.com/ggerganov/llama.cpp/tree/b7e7982953f80a656e03feb5cfb17a17a173eb26)


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

- Use `cuda` branch if you want cuda enabled.
- We build statically due to dll runtime load bug so you need to copy `cudart.lib` `cublas.lib` and `cuda.lib` into your libraries/win64 path. These are ignored atm.
- You also need to update `bUseCuda = true;` in .build.cs to add CUDA libs to build.
- Ideally this needs a variant that build with `-DBUILD_SHARED_LIBS=ON`

```
mkdir build
cd build
cmake .. -DLLAMA_CUBLAS=ON -DLLAMA_CUDA_DMMV_X=64 -DLLAMA_CUDA_MMV_Y=2 -DLLAMA_CUDA_F16=true
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

For Android build see: https://github.com/ggerganov/llama.cpp?tab=readme-ov-file#android

```
mkdir build-android
cd build-android
export NDK=<your_ndk_directory>
cmake -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-23 -DCMAKE_C_FLAGS=-march=armv8.4a+dotprod ..
$ make
```

Then the .so or .lib file was copied into the `Libraries` directory and all the .h files were copied to the `Includes` directory.
