@echo off
REM Cross-compile llama.cpp (+ llama-unreal-common) for Android as static libraries, then stage them
REM under <plugin>\ThirdParty\LlamaCpp\Lib\Android\<abi>\.
REM
REM Static, like UE's own Android build: everything links into the game's single .so with the NDK's
REM static libc++ (what UE uses), so there are no extra .so files to package and llama.cpp shares the
REM engine's allocator.
REM
REM Prerequisites:
REM   1. Android NDK r27c (27.2.12479018, what UE 5.8 requires), NDKROOT set (UE's SetupAndroid.bat does it).
REM   2. Vulkan SDK (VULKAN_SDK set): glslc for the shaders, and vulkan.hpp (the NDK only ships the C headers).
REM   3. Visual Studio 2022 (MSVC builds the vulkan-shaders-gen host tool) and Ninja on PATH.
REM   4. llama.cpp checkout at v1.0.0 or compatible.
REM
REM Usage:
REM   build-llamacpp-android.bat <path\to\llama.cpp> [abis]
REM     abis: space-separated, default "arm64-v8a x86_64" (x86_64 is for the emulator)

setlocal enabledelayedexpansion

if "%~1"=="" (
    echo ERROR: pass path to llama.cpp checkout, e.g.:
    echo   %~nx0 C:\path\to\llama.cpp
    exit /b 2
)
set "LLAMACPP_DIR=%~1"
set "ABIS=%~2"
if "%ABIS%"=="" set "ABIS=arm64-v8a x86_64"

if "%NDKROOT%"=="" (
    echo ERROR: NDKROOT not set. Install NDK 27.2.12479018 ^(UE's Engine\Extras\Android\SetupAndroid.bat^).
    exit /b 2
)
if "%VULKAN_SDK%"=="" (
    echo ERROR: VULKAN_SDK not set. Install the Vulkan SDK from https://vulkan.lunarg.com/sdk/home
    exit /b 2
)

REM Forward slashes for CMake
set "NDK=%NDKROOT:\=/%"
set "VKSDK=%VULKAN_SDK:\=/%"
set "SYSROOT_LIB=%NDK%/toolchains/llvm/prebuilt/windows-x86_64/sysroot/usr/lib"
REM The NDK toolchain compiles with -g; debug info is ~80% of the archives, so it is stripped when staging
set "STRIP=%NDKROOT%\toolchains\llvm\prebuilt\windows-x86_64\bin\llvm-strip.exe"

REM MSVC for the vulkan-shaders-gen host tool; keep mingw out of the way
call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=amd64 -no_logo || (
    echo ERROR: failed to initialize MSVC env.
    exit /b 1
)
set "PATH=%PATH:C:\ProgramData\mingw64\mingw64\bin;=%"
set "PATH=%PATH:C:\msys64\mingw64\bin;=%"

set "SCRIPT_DIR=%~dp0"
set "PLUGIN_DIR=%SCRIPT_DIR%.."
set "LLU_CMAKE=%PLUGIN_DIR%\ThirdParty\LlamaCpp\LlamaUnrealCommon\llama-unreal-common.cmake"
set "LLU_CMAKE=%LLU_CMAKE:\=/%"

pushd "%LLAMACPP_DIR%"

for %%A in (%ABIS%) do (
    set "ABI=%%A"
    set "TRIPLE="
    if "%%A"=="arm64-v8a" set "TRIPLE=aarch64-linux-android"
    if "%%A"=="x86_64" set "TRIPLE=x86_64-linux-android"
    if not defined TRIPLE (
        echo ERROR: unsupported ABI %%A ^(use arm64-v8a or x86_64^)
        popd
        exit /b 2
    )
    set "BUILD_DIR=build-android-%%A"
    echo.
    echo [build-llamacpp-android] === %%A ===

    cmake -S . -B !BUILD_DIR! -G Ninja ^
        -DCMAKE_TOOLCHAIN_FILE=%NDK%/build/cmake/android.toolchain.cmake ^
        -DANDROID_ABI=%%A ^
        -DANDROID_PLATFORM=android-26 ^
        -DANDROID_STL=c++_static ^
        -DCMAKE_BUILD_TYPE=Release ^
        -DBUILD_SHARED_LIBS=OFF ^
        -DGGML_NATIVE=OFF ^
        -DGGML_OPENMP=OFF ^
        -DGGML_VULKAN=ON ^
        -DLLAMA_CURL=OFF ^
        -DLLAMA_OPENSSL=OFF ^
        -DLLAMA_BUILD_TESTS=OFF ^
        -DLLAMA_BUILD_EXAMPLES=OFF ^
        -DLLAMA_BUILD_SERVER=OFF ^
        -DLLAMA_BUILD_APP=OFF ^
        -DLLAMA_BUILD_TOOLS=ON ^
        -DVulkan_INCLUDE_DIR=%VKSDK%/Include ^
        -DVulkan_LIBRARY=%SYSROOT_LIB%/!TRIPLE!/26/libvulkan.so ^
        -DVulkan_GLSLC_EXECUTABLE=%VKSDK%/Bin/glslc.exe ^
        -DSPIRV-Headers_DIR=%VKSDK%/Lib/cmake ^
        "-DCMAKE_PROJECT_llama.cpp_INCLUDE=%LLU_CMAKE%"
    if errorlevel 1 (
        echo [build-llamacpp-android] CMake configure failed for %%A.
        popd
        exit /b 1
    )

    cmake --build !BUILD_DIR! --config Release -j ^
        --target llama ggml ggml-base ggml-cpu ggml-vulkan llama-common llama-common-base mtmd vendor-hash cpp-httplib llama-unreal-common
    if errorlevel 1 (
        echo [build-llamacpp-android] Build failed for %%A.
        popd
        exit /b 1
    )

    REM Stage the static libraries; old ones are removed first so they do not ship
    set "STAGE_DIR=%PLUGIN_DIR%\ThirdParty\LlamaCpp\Lib\Android\%%A"
    if exist "!STAGE_DIR!" del /q "!STAGE_DIR!\*.a" 2>nul
    if not exist "!STAGE_DIR!" mkdir "!STAGE_DIR!"
    set COPIED=0
    for %%L in (libllama libggml libggml-base libggml-cpu libggml-vulkan libllama-common libllama-common-base libmtmd libvendor-hash libcpp-httplib libllama-unreal-common) do (
        set "FOUND="
        for /f "delims=" %%F in ('dir /s /b "!BUILD_DIR!\%%L.a" 2^>nul') do if not defined FOUND set "FOUND=%%F"
        if defined FOUND (
            copy /Y "!FOUND!" "!STAGE_DIR!\%%L.a" >nul
            "%STRIP%" --strip-debug "!STAGE_DIR!\%%L.a"
            set /a COPIED+=1
        ) else (
            echo   MISSING: %%L.a
        )
    )
    echo [build-llamacpp-android] %%A: staged !COPIED! of 11 libraries in !STAGE_DIR!
    if !COPIED! lss 11 (
        popd
        exit /b 1
    )
)

popd
echo.
echo [build-llamacpp-android] Done.
endlocal
