# Getnamo Fork notes

Forked for usability fixes with a CPU focused build to also obviate CUDA issues for larger release compatibility. It's fast enough for decent 7B models.

# Llama.cpp Build Parameters

Forked Plugin [Llama.cpp](https://github.com/ggerganov/llama.cpp) was built from git hash: [708e179e8562c2604240df95a2241dea17fd808b](https://github.com/ggerganov/llama.cpp/tree/708e179e8562c2604240df95a2241dea17fd808b)


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

```
mkdir build
cd build
cmake .. -DLLAMA_CUBLAS=ON
cmake --build . --config Release
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
