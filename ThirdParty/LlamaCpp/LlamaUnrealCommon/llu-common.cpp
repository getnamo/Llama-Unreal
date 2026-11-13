// Copyright 2025-current Getnamo.

#include "llu-internal.h"

#include "common.h"

namespace llu {

static thread_local std::string last_error;

void set_error(const std::string & message) {
    last_error = message;
}

void clear_error() {
    last_error.clear();
}

} // namespace llu

extern "C" {

const char * llu_last_error(void) {
    return llu::last_error.c_str();
}

void llu_embd_normalize(const float * inp, float * out, int32_t n, int32_t embd_norm) {
    common_embd_normalize(inp, out, n, embd_norm);
}

} // extern "C"
