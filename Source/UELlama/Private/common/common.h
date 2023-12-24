// Various helper functions and utilities

#pragma once

#include "llama.h"

#include "sampling.h"

#define LOG_NO_FILE_LINE_FUNCTION

#include <cmath>
#include <string>
#include <vector>
#include <random>
#include <thread>
#include <unordered_map>
#include <tuple>

#ifdef _WIN32
#define DIRECTORY_SEPARATOR '\\'
#else
#define DIRECTORY_SEPARATOR '/'
#endif // _WIN32

#define die(msg)          do { fputs("error: " msg "\n", stderr);                exit(1); } while (0)
#define die_fmt(fmt, ...) do { fprintf(stderr, "error: " fmt "\n", __VA_ARGS__); exit(1); } while (0)

#define print_build_info() do {                                                                     \
    fprintf(stderr, "%s: build = %d (%s)\n", __func__, LLAMA_BUILD_NUMBER, LLAMA_COMMIT);           \
    fprintf(stderr, "%s: built with %s for %s\n", __func__, LLAMA_COMPILER, LLAMA_BUILD_TARGET);    \
} while(0)

// build info
extern int LLAMA_BUILD_NUMBER;
extern char const *LLAMA_COMMIT;
extern char const *LLAMA_COMPILER;
extern char const *LLAMA_BUILD_TARGET;

//
// CLI argument parsing
//
int32_t get_num_physical_cores();

//
// String parsing
//

std::string parse_samplers_input(std::string input);

// Batch utils

void llama_batch_clear(struct llama_batch & batch);

void llama_batch_add(
                 struct llama_batch & batch,
                        llama_token   id,
                          llama_pos   pos,
    const std::vector<llama_seq_id> & seq_ids,
                               bool   logits);

//
// Vocab utils
//

// tokenizes a string into a vector of tokens
// should work similar to Python's `tokenizer.encode`
std::vector<llama_token> llama_tokenize(
  const struct llama_context * ctx,
           const std::string & text,
                        bool   add_bos,
                        bool   special = false);

std::vector<llama_token> llama_tokenize(
    const struct llama_model * model,
           const std::string & text,
                        bool   add_bos,
                        bool   special = false);

// tokenizes a token into a piece
// should work similar to Python's `tokenizer.id_to_piece`
std::string llama_token_to_piece(
        const struct llama_context * ctx,
                       llama_token   token);

// TODO: these should be moved in llama.h C-style API under single `llama_detokenize` function
//       that takes into account the tokenizer type and decides how to handle the leading space
//
// detokenizes a vector of tokens into a string
// should work similar to Python's `tokenizer.decode`
// removes the leading space from the first non-BOS token
std::string llama_detokenize_spm(
                         llama_context * ctx,
        const std::vector<llama_token> & tokens);

// detokenizes a vector of tokens into a string
// should work similar to Python's `tokenizer.decode`
std::string llama_detokenize_bpe(
                         llama_context * ctx,
        const std::vector<llama_token> & tokens);

// Uses the value from the model metadata if possible, otherwise
// defaults to true when model type is SPM, otherwise false.
bool llama_should_add_bos_token(const llama_model * model);