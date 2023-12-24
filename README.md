# Getnamo Fork notes

Forked for usability fixes with a CPU focused build to also obviate CUDA issues for larger release compatibility. It's fast enough for decent 7B models.

# Llama.cpp Build Parameters

Forked Plugin [Llama.cpp](https://github.com/ggerganov/llama.cpp) was built from git hash: [708e179e8562c2604240df95a2241dea17fd808b](https://github.com/ggerganov/llama.cpp/tree/708e179e8562c2604240df95a2241dea17fd808b)

With the following build commands for windows (cpu build only, CUDA ignored, see upstream for GPU version):

```
mkdir build
cd build/
cmake ..
cd ..
cmake --build build --config Release -j --verbose
```

For Android build see: https://github.com/ggerganov/llama.cpp?tab=readme-ov-file#android

Then the .so or .lib file was copied into the `Libraries` directory and all the .h files were copied to the `Includes` directory.