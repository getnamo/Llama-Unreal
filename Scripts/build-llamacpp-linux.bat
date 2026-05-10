@echo off
REM Cross-compile llama.cpp for Linux (Vulkan) using UE 5.7's bundled clang
REM toolchain (v26_clang-20.1.8-rockylinux8), then stage the resulting .so files
REM under <plugin>\ThirdParty\LlamaCpp\Lib\Linux\.
REM
REM Prerequisites:
REM   1. UE Linux toolchain v26 installed at C:\UnrealToolchains\v26_clang-20.1.8-rockylinux8\
REM      (Epic installer; sets LINUX_MULTIARCH_ROOT — but we override anyway).
REM   2. Vulkan SDK installed (VULKAN_SDK env var set, e.g. C:\VulkanSDK\1.4.304.1).
REM   3. WSL2 Ubuntu with libvulkan-dev:
REM        sudo apt install -y libvulkan-dev libvulkan1
REM   4. libvulkan.so.1 mirrored into the v26 cross sysroot (one-time):
REM        cp /usr/lib/x86_64-linux-gnu/libvulkan.so.1.3.* \
REM           /mnt/c/UnrealToolchains/v26_clang-20.1.8-rockylinux8/x86_64-unknown-linux-gnu/usr/lib64/
REM      then in Windows cmd: copy that file to libvulkan.so and libvulkan.so.1 (NOT symlinks —
REM      WSL2 creates Windows junctions that clang.exe can't follow).
REM   5. Visual Studio 2022 Build Tools installed (provides MSVC cl.exe for the
REM      vulkan-shaders-gen host tool sub-build).
REM   6. llama.cpp checkout at b9090 or compatible.
REM   7. cmake/ue57-linux-cross.cmake copied into the llama.cpp checkout's cmake/ dir.
REM
REM Usage:
REM   build-llamacpp-linux.bat <path\to\llama.cpp>

setlocal enabledelayedexpansion

if "%~1"=="" (
    echo ERROR: pass path to llama.cpp checkout, e.g.:
    echo   %~nx0 C:\path\to\llama.cpp
    exit /b 2
)
set "LLAMACPP_DIR=%~1"

if "%VULKAN_SDK%"=="" (
    echo ERROR: VULKAN_SDK env var not set. Install Vulkan SDK from https://vulkan.lunarg.com/sdk/home
    exit /b 2
)

REM Force the v26 toolchain regardless of LINUX_MULTIARCH_ROOT (which often still
REM points at older v21/v22 from prior installs). Use forward slashes to avoid
REM CMake escape-sequence parser tripping on \U.
set "LLAMA_UE_LINUX_TC=C:/UnrealToolchains/v26_clang-20.1.8-rockylinux8"

if not exist "%LLAMA_UE_LINUX_TC%/x86_64-unknown-linux-gnu/bin/clang.exe" (
    echo ERROR: v26 toolchain not found at %LLAMA_UE_LINUX_TC%.
    echo Install via Epic's Linux toolchain installer.
    exit /b 2
)

REM Initialize MSVC env so the vulkan-shaders-gen sub-build (which needs to compile
REM a HOST Windows binary at build time) finds cl.exe instead of falling back to
REM mingw gcc. Also strip mingw out of PATH for safety.
call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=amd64 || (
    echo ERROR: failed to initialize MSVC env.
    exit /b 1
)
set "PATH=%PATH:C:\ProgramData\mingw64\mingw64\bin;=%"
set "PATH=%PATH:C:\msys64\mingw64\bin;=%"

REM Resolve plugin paths (script lives at <plugin>\Scripts\)
set "SCRIPT_DIR=%~dp0"
set "PLUGIN_DIR=%SCRIPT_DIR%.."
set "STAGE_DIR=%PLUGIN_DIR%\ThirdParty\LlamaCpp\Lib\Linux"

if not exist "%STAGE_DIR%" mkdir "%STAGE_DIR%"

echo [build-llamacpp-linux] llama.cpp:  %LLAMACPP_DIR%
echo [build-llamacpp-linux] Toolchain:  %LLAMA_UE_LINUX_TC%
echo [build-llamacpp-linux] Vulkan SDK: %VULKAN_SDK%
echo [build-llamacpp-linux] Stage to:   %STAGE_DIR%
echo.

pushd "%LLAMACPP_DIR%"

REM Configure
cmake -S . -B build-linux-vulkan -G Ninja ^
    -DCMAKE_TOOLCHAIN_FILE=cmake/ue57-linux-cross.cmake ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DBUILD_SHARED_LIBS=ON ^
    -DGGML_VULKAN=ON ^
    -DGGML_NATIVE=OFF ^
    -DLLAMA_CURL=OFF ^
    -DLLAMA_BUILD_TESTS=OFF ^
    -DLLAMA_BUILD_EXAMPLES=OFF ^
    -DLLAMA_BUILD_SERVER=OFF ^
    -DLLAMA_BUILD_TOOLS=ON ^
    -DVulkan_INCLUDE_DIR=%VULKAN_SDK%/Include ^
    -DVulkan_LIBRARY=%LLAMA_UE_LINUX_TC%/x86_64-unknown-linux-gnu/usr/lib64/libvulkan.so
if errorlevel 1 (
    echo [build-llamacpp-linux] CMake configure failed.
    popd
    exit /b 1
)

REM Build the libs we need. Building all tools/* targets fails on std::filesystem
REM linkage (Rocky 8 GCC 8.5 ABI nit), so we list the specific targets we care about.
cmake --build build-linux-vulkan --config Release -j 8 ^
    --target llama ggml ggml-base ggml-cpu ggml-vulkan llama-common mtmd
if errorlevel 1 (
    echo [build-llamacpp-linux] Build failed.
    popd
    exit /b 1
)

REM Stage. Each lib produces three names — `libfoo.so` symlink, `libfoo.so.0`
REM soname symlink, and `libfoo.so.X.Y.Z` real file. We need the unversioned
REM name (UBT linker resolution) AND the soname (sibling lib runtime resolution).
REM Copy the real file under both names — symlinks don't survive WSL2->Windows
REM filesystem junctions reliably for the loader.
set COPIED=0
set "BIN=build-linux-vulkan\bin"
for %%S in (libllama libggml libggml-base libggml-cpu libggml-vulkan libllama-common libmtmd) do (
    set "REAL="
    REM Find the highest-versioned file (skip the .so.0 symlink/junction).
    for /f "delims=" %%F in ('dir /b /a-d "%BIN%\%%S.so.*" 2^>nul ^| findstr /v /e "\.so\.0"') do set "REAL=%BIN%\%%F"
    if defined REAL (
        copy /Y "!REAL!" "%STAGE_DIR%\%%S.so"   >nul
        copy /Y "!REAL!" "%STAGE_DIR%\%%S.so.0" >nul
        echo   %%S.so {+ .so.0}
        set /a COPIED+=2
    ) else (
        echo   MISSING: %%S
    )
)

popd

if !COPIED! lss 14 (
    echo.
    echo [build-llamacpp-linux] WARNING: only !COPIED! file(s) staged, expected 14.
    exit /b 1
)

echo.
echo [build-llamacpp-linux] Staged !COPIED! files (7 libs x 2 names). Done.
echo.
echo Next: in Epic Games Launcher, enable "Linux" engine platform component for UE 5.7,
echo then build the plugin Linux target:
echo   "%%UE57%%\Engine\Build\BatchFiles\Build.bat" ^<ProjectName^> Linux Development ^^
echo       -Project=^<full-path^>\^<ProjectName^>.uproject
endlocal
