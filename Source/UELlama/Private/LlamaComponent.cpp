// 2023 (c) Mika Pi

// ReSharper disable CppPrintfBadFormat
#include "UELlama/LlamaComponent.h"
#include <atomic>
#include <deque>
#include <thread>
#include <functional>
#include <mutex>

#define GGML_CUDA_DMMV_X 64
#define GGML_CUDA_F16
#define GGML_CUDA_MMV_Y 2
#define GGML_USE_CUBLAS
#define GGML_USE_K_QUANTS
#define K_QUANTS_PER_ITERATION 2

#include "llama.h"

using namespace std;


/*
 *    I copied these two functions from common.cpp file from ggerganov/llama.cpp until they
 *    update their code and create the function llama_detokenize in llama.h.
 *
 *    This is needed because we need support the string conversion of the new format GGUF.
*/
////////////////////////////////////////////////////////////////////////////////////////////////

string llama_token_to_piece(const struct llama_context * ctx, llama_token token) {
    vector<char> result(8, 0);
    const int n_tokens = llama_token_to_piece(ctx, token, result.data(), result.size());
    if (n_tokens < 0) 
    {
        result.resize(-n_tokens);
        int check = llama_token_to_piece(ctx, token, result.data(), result.size());
        GGML_ASSERT(check == -n_tokens);
    } 
    else
    {
        result.resize(n_tokens);
    }

    return std::string(result.data(), result.size());
}

string llama_detokenize_bpe(llama_context * ctx, const vector<llama_token> & tokens) {
    string piece;
    string result;

    for (size_t i = 0; i < tokens.size(); ++i) {
        piece = llama_token_to_piece(ctx, tokens[i]);

        result += piece;
    }
    return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////

namespace
{
    class Q
    {
    public:
        void enqueue(function<void()>);
        bool processQ();

    private:
        deque<function<void()>> q;
        mutex mutex_;
    };

    void Q::enqueue(function<void()> v)
    {
        lock_guard l(mutex_);
        q.emplace_back(move(v));
    }

    bool Q::processQ() {
        function<void()> v;
        {
            lock_guard l(mutex_);
            if (q.empty()) {
            return false;
            }
            v = move(q.front());
            q.pop_front();
        }
        v();
        return true;
    }

    vector<llama_token> my_llama_tokenize(  llama_context *ctx,
                                            const string &text,
                                            vector<llama_token> &res,
                                            bool add_bos)
    {
        UE_LOG(LogTemp, Warning, TEXT("Tokenize `%s`"), UTF8_TO_TCHAR(text.c_str()));
        // initialize to prompt numer of chars, since n_tokens <= n_prompt_chars
        res.resize(text.size() + (int)add_bos);
        const int n = llama_tokenize(ctx, text.c_str(), text.length(), res.data(), res.size(), add_bos);
        res.resize(n);

        return res;
    }

    constexpr int n_threads = 4;

    struct Params
    {
        FString prompt = "Hello";
        FString pathToModel = "/media/mika/Michigan/prj/llama-2-13b-chat.ggmlv3.q8_0.bin";
        TArray<FString> stopSequences;
    };
} // namespace

namespace Internal
{
    class Llama
    {
    public:
        Llama();
        ~Llama();

        void activate(bool bReset, Params);
        void deactivate();
        void insertPrompt(FString v);
        void process();
        void stopGenerating();

        function<void(FString)> tokenCb;
        function<void(bool)> eosCb;

    private:
        llama_model *model = nullptr;
        llama_context *ctx = nullptr;
        Q qMainToThread;
        Q qThreadToMain;
        atomic_bool running = true;
        thread thread;
        vector<vector<llama_token>> stopSequences;
        vector<llama_token> embd_inp;
        vector<llama_token> embd;
        vector<llama_token> res;
        int n_past = 0;
        vector<llama_token> last_n_tokens;
        int n_consumed = 0;
        bool eos = false;

        void threadRun();
        void unsafeActivate(bool bReset, Params);
        void unsafeDeactivate();
        void unsafeInsertPrompt(FString);
    };

    void Llama::insertPrompt(FString v)
    {
        qMainToThread.enqueue([this, v = move(v)]() mutable { unsafeInsertPrompt(move(v)); });
    }

    void Llama::unsafeInsertPrompt(FString v)
    {
        if (!ctx) {
            UE_LOG(LogTemp, Error, TEXT("Llama not activated"));
            return;
        }
        string stdV = string(" ") + TCHAR_TO_UTF8(*v);
        vector<llama_token> line_inp = my_llama_tokenize(ctx, stdV, res, false /* add bos */);
        embd_inp.insert(embd_inp.end(), line_inp.begin(), line_inp.end());
    }

    Llama::Llama() : thread([this]() { 
                threadRun();
            }) 
    {
        //todo: fix the thread start/end spam
    }

    void Llama::stopGenerating()
    {
        qMainToThread.enqueue([this]() 
        {
            eos = true;
        });
    }

    void Llama::threadRun()
    {
        UE_LOG(LogTemp, Warning, TEXT("%p Llama thread is running"), this);
        const int n_predict = -1;
        const int n_keep = 0;
        const int n_batch = 512;
        while (running)
        {
            while (qMainToThread.processQ())
                ;
            if (!model)
            {
                using namespace chrono_literals;
                this_thread::sleep_for(200ms);
                continue;
            }

            if (eos && (int)embd_inp.size() <= n_consumed)
            {
                using namespace chrono_literals;
                this_thread::sleep_for(200ms);
                continue;
            }
            eos = false;

            const int n_ctx = llama_n_ctx(ctx);
            if (embd.size() > 0)
            {
                // Note: n_ctx - 4 here is to match the logic for commandline prompt handling via
                // --prompt or --file which uses the same value.
                int max_embd_size = n_ctx - 4;
                // Ensure the input doesn't exceed the context size by truncating embd if necessary.
                if ((int)embd.size() > max_embd_size)
                {
                    uint64 skipped_tokens = embd.size() - max_embd_size;
                    UE_LOG(LogTemp,
                                 Error,
                                 TEXT("<<input too long: skipped %zu token%s>>"),
                                 skipped_tokens,
                                 skipped_tokens != 1 ? "s" : "");
                    embd.resize(max_embd_size);
                }

                // infinite text generation via context swapping
                // if we run out of context:
                // - take the n_keep first tokens from the original prompt (via n_past)
                // - take half of the last (n_ctx - n_keep) tokens and recompute the logits in batches
                if (n_past + (int)embd.size() > n_ctx)
                {
                    UE_LOG(LogTemp, Warning, TEXT("%p context resetting"), this);
                    if (n_predict == -2)
                    {
                        UE_LOG(LogTemp, Error, TEXT("context full, stopping generation"));
                        unsafeDeactivate();
                        continue;
                    }

                    const int n_left = n_past - n_keep;
                    // always keep the first token - BOS
                    n_past = max(1, n_keep);

                    // insert n_left/2 tokens at the start of embd from last_n_tokens
                    embd.insert(embd.begin(),
                                            last_n_tokens.begin() + n_ctx - n_left / 2 - embd.size(),
                                            last_n_tokens.end() - embd.size());
                }

                // evaluate tokens in batches
                // embd is typically prepared beforehand to fit within a batch, but not always

                for (int i = 0; i < (int)embd.size(); i += n_batch)
                {
                    int n_eval = (int)embd.size() - i;
                    if (n_eval > n_batch)
                    {
                        n_eval = n_batch;
                    }
                    string str = string{};
                    for (auto j = 0; j < n_eval; ++j)
                        //    TODO: Replace this llama_detokenize_bpe with llama_detokenize when can be possible.
                        str += llama_detokenize_bpe(ctx, {embd[i + j]});
                    UE_LOG(LogTemp, Warning, TEXT("%p eval tokens `%s`"), this, UTF8_TO_TCHAR(str.c_str()));
                    if (llama_eval(ctx, &embd[i], n_eval, n_past, n_threads))
                    {
                        UE_LOG(LogTemp, Error, TEXT("failed to eval"));
                        unsafeDeactivate();
                        continue;
                    }
                    n_past += n_eval;
                }
            }

            embd.clear();

            bool haveHumanTokens = false;

            if ((int)embd_inp.size() <= n_consumed)
            {
                // out of user input, sample next token
                const float temp = 0.80f;
                const int32_t top_k = 40;
                const float top_p = 0.95f;
                const float tfs_z = 1.00f;
                const float typical_p = 1.00f;
                const int32_t repeat_last_n = 64;
                const float repeat_penalty = 1.10f;
                const float alpha_presence = 0.00f;
                const float alpha_frequency = 0.00f;
                const int mirostat = 0;
                const float mirostat_tau = 5.f;
                const float mirostat_eta = 0.1f;
                const bool penalize_nl = true;

                llama_token id = 0;

                {
                    float* logits = llama_get_logits(ctx);
                    int n_vocab = llama_n_vocab(ctx);

                    vector<llama_token_data> candidates;
                    candidates.reserve(n_vocab);
                    for (llama_token token_id = 0; token_id < n_vocab; token_id++)
                    {
                        candidates.emplace_back(llama_token_data{token_id, logits[token_id], 0.0f});
                    }

                    llama_token_data_array candidates_p = {candidates.data(), candidates.size(), false};

                    // Apply penalties
                    float nl_logit = logits[llama_token_nl(ctx)];
                    int last_n_repeat = min(min((int)last_n_tokens.size(), repeat_last_n), n_ctx);
                    llama_sample_repetition_penalty(ctx,
                                                                                    &candidates_p,
                                                                                    last_n_tokens.data() + last_n_tokens.size() - last_n_repeat,
                                                                                    last_n_repeat,
                                                                                    repeat_penalty);
                    llama_sample_frequency_and_presence_penalties(ctx,
                                                                                                                &candidates_p,
                                                                                                                last_n_tokens.data() + last_n_tokens.size() -
                                                                                                                    last_n_repeat,
                                                                                                                last_n_repeat,
                                                                                                                alpha_frequency,
                                                                                                                alpha_presence);
                    if (!penalize_nl)
                    {
                        logits[llama_token_nl(ctx)] = nl_logit;
                    }

                    if (temp <= 0)
                    {
                        // Greedy sampling
                        id = llama_sample_token_greedy(ctx, &candidates_p);
                    }
                    else
                    {
                        if (mirostat == 1)
                        {
                            static float mirostat_mu = 2.0f * mirostat_tau;
                            const int mirostat_m = 100;
                            llama_sample_temperature(ctx, &candidates_p, temp);
                            id = llama_sample_token_mirostat(
                                ctx, &candidates_p, mirostat_tau, mirostat_eta, mirostat_m, &mirostat_mu);
                        }
                        else if (mirostat == 2)
                        {
                            static float mirostat_mu = 2.0f * mirostat_tau;
                            llama_sample_temperature(ctx, &candidates_p, temp);
                            id = llama_sample_token_mirostat_v2(
                                ctx, &candidates_p, mirostat_tau, mirostat_eta, &mirostat_mu);
                        }
                        else
                        {
                            // Temperature sampling
                            llama_sample_top_k(ctx, &candidates_p, top_k, 1);
                            llama_sample_tail_free(ctx, &candidates_p, tfs_z, 1);
                            llama_sample_typical(ctx, &candidates_p, typical_p, 1);
                            llama_sample_top_p(ctx, &candidates_p, top_p, 1);
                            llama_sample_temperature(ctx, &candidates_p, temp);
                            id = llama_sample_token(ctx, &candidates_p);
                        }
                    }

                    last_n_tokens.erase(last_n_tokens.begin());
                    last_n_tokens.push_back(id);
                }

                // add it to the context
                embd.push_back(id);
            }
            else
            {
                // some user input remains from prompt or interaction, forward it to processing
                while ((int)embd_inp.size() > n_consumed)
                {
                    const int tokenId = embd_inp[n_consumed];
                    embd.push_back(tokenId);
                    last_n_tokens.erase(last_n_tokens.begin());
                    last_n_tokens.push_back(embd_inp[n_consumed]);
                    haveHumanTokens = true;
                    ++n_consumed;
                    if ((int)embd.size() >= n_batch)
                    {
                        // TODO-Mika
                        break;
                    }
                }
            }

            // TODO: Revert these changes to the commented code when the llama.cpp add the llama_detokenize function.
            
            // display text
            // for (auto id : embd)
            // {
            //     FString token = llama_detokenize(ctx, id);
            //     qThreadToMain.enqueue([token = move(token), this]() {
            //         if (!tokenCb)
            //             return;
            //         tokenCb(move(token));
            //     });
            // }
            
            FString token = UTF8_TO_TCHAR(llama_detokenize_bpe(ctx, embd).c_str());
            qThreadToMain.enqueue([token = move(token), this] {
                if (!tokenCb)
                    return;
                tokenCb(move(token));
            });
            ////////////////////////////////////////////////////////////////////////

            bool const hasStopSeq = [&]
            {
                if (stopSequences.empty())
                    return false;
                if (haveHumanTokens)
                    return false;

                for (vector<llama_token> stopSeq : stopSequences)
                {
                    if (last_n_tokens.size() < stopSeq.size())
                        return false;
                    bool match = true;
                    for (unsigned i = 0U; i < stopSeq.size(); ++i)
                        if (last_n_tokens[last_n_tokens.size() - stopSeq.size() + i] != stopSeq[i])
                        {
                            match = false;
                            break;
                        }
                    if (match)
                        return true;
                }
                return false;
            }();

            if ((!embd.empty() && embd.back() == llama_token_eos(ctx)) || hasStopSeq)
            {
                UE_LOG(LogTemp, Warning, TEXT("%p EOS"), this);
                eos = true;
                const bool stopSeqSafe = hasStopSeq;

                //notify main thread we're done
                qThreadToMain.enqueue([stopSeqSafe, this] {
                    if (!eosCb)
                        return;

                    eosCb(stopSeqSafe);
                });
            }
        }
        unsafeDeactivate();
        UE_LOG(LogTemp, Warning, TEXT("%p Llama thread stopped"), this);
    }

    Llama::~Llama()
    {
        running = false;
        thread.join();
    }

    void Llama::process()
    {
        while (qThreadToMain.processQ())
            ;
    }

    void Llama::activate(bool bReset, Params params)
    {
        qMainToThread.enqueue([bReset, params = move(params), this]() mutable {
            unsafeActivate(bReset, move(params));
        });
    }

    void Llama::deactivate()
    {
        qMainToThread.enqueue([this]() { unsafeDeactivate(); });
    }

    void Llama::unsafeActivate(bool bReset, Params params)
    {
        UE_LOG(LogTemp, Warning, TEXT("%p Loading LLM model %p bReset: %d"), this, model, bReset);
        if (bReset)
            unsafeDeactivate();
        if (model)
            return;
        llama_context_params lparams = []()
        {
            llama_context_params lparams = llama_context_default_params();
            // -eps 1e-5 -t 8 -ngl 50
            lparams.n_gpu_layers = 50;
            lparams.n_ctx = 4096;
            lparams.seed = time(nullptr);
            return lparams;
        }();
        model = llama_load_model_from_file(TCHAR_TO_UTF8(*params.pathToModel), lparams);
        if (!model)
        {
            UE_LOG(LogTemp, Error, TEXT("%p unable to load model"), this);
            unsafeDeactivate();
            return;
        }
        ctx = llama_new_context_with_model(model, lparams);

        // tokenize the prompt
        string stdPrompt = string(" ") + TCHAR_TO_UTF8(*params.prompt);
        embd_inp = my_llama_tokenize(ctx, stdPrompt, res, true /* add bos */);
        if (!params.stopSequences.IsEmpty())
        {
            for (int i = 0; i < params.stopSequences.Num(); ++i)
            {
                const FString& stopSeq = params.stopSequences[i];
                string str = string{TCHAR_TO_UTF8(*stopSeq)};
                if (::isalnum(str[0]))
                    str = " " + str;
                vector<llama_token> seq = my_llama_tokenize(ctx, str, res, false /* add bos */);
                stopSequences.emplace_back(move(seq));
            }
        }
        else
            stopSequences.clear();

        const int n_ctx = llama_n_ctx(ctx);

        if ((int)embd_inp.size() > n_ctx - 4)
        {
            UE_LOG(
                LogTemp, Error, TEXT("prompt is too long (%d tokens, max %d)"), (int)embd_inp.size(), n_ctx - 4);
            unsafeDeactivate();
            return;
        }

        // do one empty run to warm up the model
        {
            const vector tmp = {
                llama_token_bos(ctx),
            };
            llama_eval(ctx, tmp.data(), tmp.size(), 0, n_threads);
            llama_reset_timings(ctx);
        }
        last_n_tokens.resize(n_ctx);
        fill(last_n_tokens.begin(), last_n_tokens.end(), 0);
        n_consumed = 0;
    }

    void Llama::unsafeDeactivate()
    {
        UE_LOG(LogTemp, Warning, TEXT("%p Unloading LLM model %p"), this, model);
        if (!model)
            return;
        llama_print_timings(ctx);
        llama_free(ctx);
        ctx = nullptr;
        llama_free_model(model);
        model = nullptr;
    }
} // namespace Internal

ULlamaComponent::ULlamaComponent(const FObjectInitializer &ObjectInitializer)
    : UActorComponent(ObjectInitializer), llama(make_unique<Internal::Llama>())
{
    PrimaryComponentTick.bCanEverTick = true;
    PrimaryComponentTick.bStartWithTickEnabled = true;
    llama->tokenCb = [this](FString NewToken) 
    { 
        OnNewTokenGenerated.Broadcast(move(NewToken));
    };
    llama->eosCb = [this](bool StopTokenCausedEos)
    {
        OnEndOfStream.Broadcast(StopTokenCausedEos);
    };
}

ULlamaComponent::~ULlamaComponent() = default;

void ULlamaComponent::Activate(bool bReset)
{
    Super::Activate(bReset);
    Params params;
    params.pathToModel = PathToModel;
    params.prompt = Prompt;
    params.stopSequences = StopSequences;
    llama->activate(bReset, move(params));
}

void ULlamaComponent::Deactivate()
{
    llama->deactivate();
    Super::Deactivate();
}

void ULlamaComponent::TickComponent(float DeltaTime,
                                    ELevelTick TickType,
                                    FActorComponentTickFunction* ThisTickFunction)
{
    Super::TickComponent(DeltaTime, TickType, ThisTickFunction);
    llama->process();
}

auto ULlamaComponent::InsertPrompt(const FString& v) -> void
{
    llama->insertPrompt(v);
}

void ULlamaComponent::StopGenerating()
{
    llama->stopGenerating();
}
