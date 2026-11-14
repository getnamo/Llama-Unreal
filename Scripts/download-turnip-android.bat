@echo off
REM Downloads the Mesa Turnip Vulkan driver for Adreno GPUs into ThirdParty\MesaTurnip\arm64-v8a\.
REM
REM Optional, Android arm64 only. When the driver is there, arm64 builds package it and llama.cpp uses it
REM instead of Qualcomm's driver, which fails to compile some of llama.cpp's Vulkan shaders on Adreno GPUs.
REM See ThirdParty\MesaTurnip\README.md.
REM
REM The driver comes from https://github.com/nihui/mesa-turnip-android-driver (MIT, built from Mesa).
REM
REM Usage:
REM   download-turnip-android.bat [version]
REM     version: a release tag of that repository, default 26.3.0 (the version this plugin was tested with;
REM     only its checksum is verified)

setlocal

REM Windows' own curl and tar (bsdtar reads zip files; a GNU tar earlier on PATH would not)

set "VERSION=%~1"
if "%VERSION%"=="" set "VERSION=26.3.0"
set "KNOWN_VERSION=26.3.0"
set "KNOWN_SHA256=6a9bef1c83aa6280e2389d33a992f2d80199d1d13be629c7ff12d5f9f17b5803"

set "SCRIPT_DIR=%~dp0"
set "DEST_DIR=%SCRIPT_DIR%..\ThirdParty\MesaTurnip\arm64-v8a"
set "WORK_DIR=%TEMP%\llama-unreal-turnip-%VERSION%"
set "ZIP_NAME=mesa-turnip-android-%VERSION%.zip"
set "URL=https://github.com/nihui/mesa-turnip-android-driver/releases/download/%VERSION%/%ZIP_NAME%"

if exist "%WORK_DIR%" rmdir /s /q "%WORK_DIR%"
mkdir "%WORK_DIR%" || exit /b 1

echo [download-turnip-android] %URL%
"%SystemRoot%\System32\curl.exe" -fL --retry 3 -o "%WORK_DIR%\%ZIP_NAME%" "%URL%" || (
    echo ERROR: download failed. Check the version: https://github.com/nihui/mesa-turnip-android-driver/releases
    exit /b 1
)

if "%VERSION%"=="%KNOWN_VERSION%" (
    powershell -NoProfile -Command "if ((Get-FileHash -Algorithm SHA256 '%WORK_DIR%\%ZIP_NAME%').Hash -ne '%KNOWN_SHA256%') { exit 1 }" || (
        echo ERROR: checksum mismatch for %ZIP_NAME%, not installing it.
        exit /b 1
    )
    echo [download-turnip-android] checksum OK
) else (
    echo [download-turnip-android] note: no known checksum for %VERSION%, not verified
)

"%SystemRoot%\System32\tar.exe" -xf "%WORK_DIR%\%ZIP_NAME%" -C "%WORK_DIR%" || exit /b 1
if not exist "%WORK_DIR%\libvulkan_freedreno.so" (
    echo ERROR: libvulkan_freedreno.so not found in %ZIP_NAME%
    exit /b 1
)

if not exist "%DEST_DIR%" mkdir "%DEST_DIR%"
copy /y "%WORK_DIR%\libvulkan_freedreno.so" "%DEST_DIR%\libvulkan_freedreno.so" > nul || exit /b 1
rmdir /s /q "%WORK_DIR%"

echo [download-turnip-android] installed Turnip %VERSION% to %DEST_DIR%\libvulkan_freedreno.so
echo [download-turnip-android] delete that file to stop packaging it
exit /b 0
