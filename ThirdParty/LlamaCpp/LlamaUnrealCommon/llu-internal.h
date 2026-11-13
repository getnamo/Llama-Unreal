// Copyright 2025-current Getnamo.
//
// Shared helpers for the llama-unreal-common implementation files (not part of the public API).

#pragma once

#include "llama-unreal-common.h"

#include <algorithm>
#include <cstring>
#include <exception>
#include <string>

namespace llu {

void set_error(const std::string & message);
void clear_error();

// Copies src into dst (capacity length), NUL-terminated (truncated) when terminate is set
inline void copy_out(const std::string & src, char * dst, int32_t length, bool terminate) {
    if (dst == nullptr || length <= 0) {
        return;
    }
    const size_t n = terminate ? std::min(src.size(), (size_t) length - 1) : std::min(src.size(), (size_t) length);
    std::memcpy(dst, src.data(), n);
    if (terminate) {
        dst[n] = '\0';
    }
}

// Runs body, turning exceptions into llu_last_error() and the given fallback return value
template <typename T, typename F>
T guarded(T fallback, const char * what, F && body) {
    clear_error();
    try {
        return body();
    } catch (const std::exception & e) {
        set_error(std::string(what) + ": " + e.what());
    } catch (...) {
        set_error(std::string(what) + ": unknown exception");
    }
    return fallback;
}

template <typename F>
void guarded_void(const char * what, F && body) {
    clear_error();
    try {
        body();
    } catch (const std::exception & e) {
        set_error(std::string(what) + ": " + e.what());
    } catch (...) {
        set_error(std::string(what) + ": unknown exception");
    }
}

// Rejects param structs from a header that doesn't match this build
template <typename P>
bool check_struct_size(const P * params, const char * name) {
    if (params == nullptr) {
        set_error(std::string(name) + " is null");
        return false;
    }
    if (params->struct_size != (int32_t) sizeof(P)) {
        set_error(std::string(name) + " size mismatch (llama-unreal-common.h and the library are from different builds)");
        return false;
    }
    return true;
}

} // namespace llu
