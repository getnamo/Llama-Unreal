#include "common.h"
#include "llama.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iterator>
#include <iostream>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <cinttypes>

#if defined(__APPLE__) && defined(__MACH__)
#include <sys/types.h>
#include <sys/sysctl.h>
#endif

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <codecvt>
#include <locale>
#include <windows.h>
#include <fcntl.h>
#include <io.h>
#else
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

#define LOG_DISABLE_LOGS 1

int32_t get_num_physical_cores() {
#ifdef __linux__
    // enumerate the set of thread siblings, num entries is num cores
    std::unordered_set<std::string> siblings;
    for (uint32_t cpu=0; cpu < UINT32_MAX; ++cpu) {
        std::ifstream thread_siblings("/sys/devices/system/cpu"
            + std::to_string(cpu) + "/topology/thread_siblings");
        if (!thread_siblings.is_open()) {
            break; // no more cpus
        }
        std::string line;
        if (std::getline(thread_siblings, line)) {
            siblings.insert(line);
        }
    }
    if (!siblings.empty()) {
        return static_cast<int32_t>(siblings.size());
    }
#elif defined(__APPLE__) && defined(__MACH__)
    int32_t num_physical_cores;
    size_t len = sizeof(num_physical_cores);
    int result = sysctlbyname("hw.perflevel0.physicalcpu", &num_physical_cores, &len, NULL, 0);
    if (result == 0) {
        return num_physical_cores;
    }
    result = sysctlbyname("hw.physicalcpu", &num_physical_cores, &len, NULL, 0);
    if (result == 0) {
        return num_physical_cores;
    }
#elif defined(_WIN32)
    //TODO: Implement
#endif
    unsigned int n_threads = std::thread::hardware_concurrency();
    return n_threads > 0 ? (n_threads <= 4 ? n_threads : n_threads / 2) : 4;
}

//
// String parsing
//

std::string parse_samplers_input(std::string input) {
    std::string output = "";
    // since samplers names are written multiple ways
    // make it ready for both system names and input names
    std::unordered_map<std::string, char> samplers_symbols {
        {"top_k",      'k'},
        {"top-k",      'k'},
        {"top_p",      'p'},
        {"top-p",      'p'},
        {"nucleus",    'p'},
        {"typical_p",  'y'},
        {"typical-p",  'y'},
        {"typical",    'y'},
        {"min_p",      'm'},
        {"min-p",      'm'},
        {"tfs_z",      'f'},
        {"tfs-z",      'f'},
        {"tfs",        'f'},
        {"temp",       't'},
        {"temperature",'t'}
    };
    // expected format example: "temp;top_k;tfs_z;typical_p;top_p;min_p"
    size_t separator = input.find(';');
    while (separator != input.npos) {
        std::string name = input.substr(0,separator);
        input = input.substr(separator+1);
        separator = input.find(';');

        if (samplers_symbols.find(name) != samplers_symbols.end()) {
            output += samplers_symbols[name];
        }
    }
    if (samplers_symbols.find(input) != samplers_symbols.end()) {
        output += samplers_symbols[input];
    }
    return output;
}


void llama_batch_clear(struct llama_batch & batch) {
    batch.n_tokens = 0;
}

void llama_batch_add(
                 struct llama_batch & batch,
                        llama_token   id,
                          llama_pos   pos,
    const std::vector<llama_seq_id> & seq_ids,
                               bool   logits) {
    batch.token   [batch.n_tokens] = id;
    batch.pos     [batch.n_tokens] = pos;
    batch.n_seq_id[batch.n_tokens] = seq_ids.size();
    for (size_t i = 0; i < seq_ids.size(); ++i) {
        batch.seq_id[batch.n_tokens][i] = seq_ids[i];
    }
    batch.logits  [batch.n_tokens] = logits;

    batch.n_tokens++;
}

//
// Vocab utils
//

std::vector<llama_token> llama_tokenize(
  const struct llama_context * ctx,
           const std::string & text,
                        bool   add_bos,
                        bool   special) {
    return llama_tokenize(llama_get_model(ctx), text, add_bos, special);
}

std::vector<llama_token> llama_tokenize(
    const struct llama_model * model,
           const std::string & text,
                        bool   add_bos,
                        bool   special) {
    // upper limit for the number of tokens
    int n_tokens = text.length() + add_bos;
    std::vector<llama_token> result(n_tokens);
    n_tokens = llama_tokenize(model, text.data(), text.length(), result.data(), result.size(), add_bos, special);
    if (n_tokens < 0) {
        result.resize(-n_tokens);
        int check = llama_tokenize(model, text.data(), text.length(), result.data(), result.size(), add_bos, special);
        GGML_ASSERT(check == -n_tokens);
    } else {
        result.resize(n_tokens);
    }
    return result;
}

std::string llama_token_to_piece(const struct llama_context * ctx, llama_token token) {
    std::vector<char> result(8, 0);
    const int n_tokens = llama_token_to_piece(llama_get_model(ctx), token, result.data(), result.size());
    if (n_tokens < 0) {
        result.resize(-n_tokens);
        int check = llama_token_to_piece(llama_get_model(ctx), token, result.data(), result.size());
        GGML_ASSERT(check == -n_tokens);
    } else {
        result.resize(n_tokens);
    }

    return std::string(result.data(), result.size());
}

std::string llama_detokenize_spm(llama_context * ctx, const std::vector<llama_token> & tokens) {
    const llama_token bos_id = llama_token_bos(llama_get_model(ctx));

    std::string piece;
    std::string result;

    for (size_t i = 0; i < tokens.size(); ++i) {
        piece = llama_token_to_piece(ctx, tokens[i]);

        // remove the leading space of the first non-BOS token
        if (((tokens[0] == bos_id && i == 1) || (tokens[0] != bos_id && i == 0)) && piece[0] == ' ') {
            piece = piece.substr(1);
        }

        result += piece;
    }

    return result;
}

std::string llama_detokenize_bpe(llama_context * ctx, const std::vector<llama_token> & tokens) {
    std::string piece;
    std::string result;

    for (size_t i = 0; i < tokens.size(); ++i) {
        piece = llama_token_to_piece(ctx, tokens[i]);

        result += piece;
    }

    // NOTE: the original tokenizer decodes bytes after collecting the pieces.
    return result;
}

bool llama_should_add_bos_token(const llama_model * model) {
    const int add_bos = llama_add_bos_token(model);

    return add_bos != -1 ? bool(add_bos) : (llama_vocab_type(model) == LLAMA_VOCAB_TYPE_SPM);
}