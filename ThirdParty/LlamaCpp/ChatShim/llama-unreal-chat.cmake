# Adds the llama-unreal-chat shared library (Jinja chat templating behind a C API) to a llama.cpp build,
# without modifying llama.cpp. Pass this file when configuring llama.cpp:
#
#   cmake .. <usual options> "-DCMAKE_PROJECT_llama.cpp_INCLUDE=<plugin>/ThirdParty/LlamaCpp/ChatShim/llama-unreal-chat.cmake"
#   cmake --build . --config Release --target ... llama-unreal-chat
#
# CMake includes this right after llama.cpp's project() call, before llama-common exists, so the target is
# defined at the end of the top-level directory instead (output dirs, versions and llama-common are set by then).

set(LLU_CHAT_SHIM_DIR "${CMAKE_CURRENT_LIST_DIR}")

function(llu_add_chat_shim)
    if (NOT TARGET llama-common)
        message(WARNING "llama-unreal-chat: llama-common isn't built (LLAMA_BUILD_COMMON=OFF?), skipping the chat shim")
        return()
    endif()

    add_library(llama-unreal-chat SHARED
        "${LLU_CHAT_SHIM_DIR}/llama-unreal-chat.cpp"
        "${LLU_CHAT_SHIM_DIR}/llama-unreal-chat.h")
    target_include_directories(llama-unreal-chat PUBLIC "${LLU_CHAT_SHIM_DIR}")
    target_compile_definitions(llama-unreal-chat PRIVATE LLU_CHAT_BUILD)
    target_link_libraries(llama-unreal-chat PRIVATE llama-common llama)
    target_compile_features(llama-unreal-chat PRIVATE cxx_std_17)

    # Same file naming/soname scheme as libllama (libllama-unreal-chat.so.N.Y.Z on Linux)
    get_target_property(_llu_version llama VERSION)
    get_target_property(_llu_soversion llama SOVERSION)
    if (_llu_version AND _llu_soversion)
        set_target_properties(llama-unreal-chat PROPERTIES VERSION "${_llu_version}" SOVERSION "${_llu_soversion}")
    endif()
endfunction()

cmake_language(DEFER DIRECTORY "${CMAKE_SOURCE_DIR}" CALL llu_add_chat_shim)
