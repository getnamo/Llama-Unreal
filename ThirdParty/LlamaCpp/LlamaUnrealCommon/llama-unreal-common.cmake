# Adds the llama-unreal-common shared library (Llama-Unreal's C API over llama-common: chat templates,
# sampling, speculative decoding) to a llama.cpp build, without modifying llama.cpp. Pass this file when
# configuring llama.cpp:
#
#   cmake .. <usual options> "-DCMAKE_PROJECT_llama.cpp_INCLUDE=<plugin>/ThirdParty/LlamaCpp/LlamaUnrealCommon/llama-unreal-common.cmake"
#   cmake --build . --config Release --target ... llama-unreal-common
#
# Optional: -DLLU_BUILD_TEST=ON adds llama-unreal-common-test, a CPU-only check of the library (see test/).
#
# CMake includes this right after llama.cpp's project() call, before llama-common exists, so the targets
# are defined at the end of the top-level directory instead (output dirs, versions and llama-common are set by then).

set(LLU_DIR "${CMAKE_CURRENT_LIST_DIR}")
option(LLU_BUILD_TEST "llama-unreal-common: build the CPU-only test program" OFF)

function(llu_add_targets)
    if (NOT TARGET llama-common)
        message(WARNING "llama-unreal-common: llama-common isn't built (LLAMA_BUILD_COMMON=OFF?), skipping it")
        return()
    endif()

    add_library(llama-unreal-common SHARED
        "${LLU_DIR}/llama-unreal-common.h"
        "${LLU_DIR}/llu-internal.h"
        "${LLU_DIR}/llu-common.cpp"
        "${LLU_DIR}/llu-chat.cpp"
        "${LLU_DIR}/llu-sampling.cpp"
        "${LLU_DIR}/llu-speculative.cpp")
    target_include_directories(llama-unreal-common PUBLIC "${LLU_DIR}")
    target_compile_definitions(llama-unreal-common PRIVATE LLU_BUILD)
    target_link_libraries(llama-unreal-common PUBLIC llama PRIVATE llama-common)
    target_compile_features(llama-unreal-common PRIVATE cxx_std_17)

    # Same file naming/soname scheme as libllama (libllama-unreal-common.so.N.Y.Z on Linux)
    get_target_property(_llu_version llama VERSION)
    get_target_property(_llu_soversion llama SOVERSION)
    if (_llu_version AND _llu_soversion)
        set_target_properties(llama-unreal-common PROPERTIES VERSION "${_llu_version}" SOVERSION "${_llu_soversion}")
    endif()

    if (LLU_BUILD_TEST)
        # Uses only the C APIs (llama.h + llama-unreal-common.h), like the plugin does
        add_executable(llama-unreal-common-test "${LLU_DIR}/test/llu-test.cpp")
        target_link_libraries(llama-unreal-common-test PRIVATE llama-unreal-common llama)
        target_compile_features(llama-unreal-common-test PRIVATE cxx_std_17)
    endif()
endfunction()

cmake_language(DEFER DIRECTORY "${CMAKE_SOURCE_DIR}" CALL llu_add_targets)
