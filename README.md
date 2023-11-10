# Getnamo Fork notes

Forked for usability fixes with a CPU focused build to also obviate CUDA issues for larger release compatibility. It's fast enough for decent 7B models.

# Llama.cpp Build Parameters

Forked Plugin [Llama.cpp](https://github.com/ggerganov/llama.cpp) was built from git hash: [a40f2b656fab364ce0aff98dbefe9bd9c3721cc9](https://github.com/ggerganov/llama.cpp/tree/a40f2b656fab364ce0aff98dbefe9bd9c3721cc9)

With the following build commands (cpu build only, CUDA ignored, see upstream for GPU version):

```
mkdir build
cd build/
cmake .. -DBUILD_SHARED_LIBS=ON
cd ..
cmake --build build --config Release -j --verbose
```

Then the .so or .lib file was copied into the `Libraries` directory and all the .h files were copied to the `Includes` directory. In Windows you should put the build/bin/llama.dll into `Binaries/Win64` directory.

You will need to have CUDA 12.2 installed or you will have an error loading the "UELlama" Module, this is because the llama.dll was compiled with that CUDA version, if you want to switch the version you will re-compile the binary.

# Resources
[![🌸 This dude put LLaMA 2 inside UE5 🌸 41 / 100 🌸](https://img.youtube.com/vi/j_r5xWm3Xl8/maxresdefault.jpg)](https://www.youtube.com/watch?v=j_r5xWm3Xl8)
