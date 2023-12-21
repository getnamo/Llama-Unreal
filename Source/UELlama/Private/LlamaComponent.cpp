// 2023 (c) Mika Pi, Modifications Getnamo

#include "UELlama/LlamaComponent.h"
#include <atomic>
#include <deque>
#include <thread>
#include <functional>
#include <mutex>
#include "HAL/PlatformTime.h"
#include "Misc/Paths.h"
#include "HAL/FileManager.h"

#if PLATFORM_ANDROID
#include "Android/AndroidPlatformFile.h"
#endif

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
        q.emplace_back(std::move(v));
    }

    bool Q::processQ() {
        function<void()> v;
        {
            lock_guard l(mutex_);
            if (q.empty()) {
                return false;
            }
            v = std::move(q.front());
            q.pop_front();
        }
        v();
        return true;
    }

    vector<llama_token> my_llama_tokenize(  llama_context *Context,
                                            const string &Text,
                                            vector<llama_token> &Res,
                                            bool AddBos)
    {
        UE_LOG(LogTemp, Warning, TEXT("Tokenize `%s`"), UTF8_TO_TCHAR(Text.c_str()));
        // initialize to Prompt numer of chars, since n_tokens <= n_prompt_chars
        Res.resize(Text.size() + (int)AddBos);
        const int n = llama_tokenize(Context, Text.c_str(), Text.length(), Res.data(), Res.size(), AddBos);
        Res.resize(n);

        return Res;
    }

    struct Params
    {
        FString Prompt = "Hello";
        FString PathToModel = "/media/mika/Michigan/prj/llama-2-13b-chat.ggmlv3.q8_0.bin";
        TArray<FString> StopSequences;
    };
} // namespace

namespace Internal
{
    class FLlama
    {
    public:
        FLlama();
        ~FLlama();

        void StartStopThread(bool bShouldRun);

        void Activate(bool bReset, const FLLMModelParams& Params);
        void Deactivate();
        void InsertPrompt(FString Prompt);
        void Process();
        void StopGenerating();
        void ResumeGenerating();

        function<void(FString, int32)> OnTokenCb;
        function<void(bool, float)> OnEosCb;
        function<void(void)> OnStartEvalCb;
        function<void(void)> OnContextResetCb;
        function<void(FString)> OnErrorCb;

        //Passthrough from component
        FLLMModelParams Params;

        bool bShouldLog = true;

        static FString ModelsRelativeRootPath();
        static FString ParsePathIntoFullPath(const FString& InRelativeOrAbsolutePath);

    private:
        llama_model *Model = nullptr;
        llama_context *Context = nullptr;
        Q qMainToThread;
        Q qThreadToMain;
        atomic_bool bRunning = false;
        thread qThread;
        vector<vector<llama_token>> StopSequences;
        vector<llama_token> embd_inp;
        vector<llama_token> Embd;
        vector<llama_token> res;
        int NPast = 0;
        vector<llama_token> last_n_tokens;
        int n_consumed = 0;
        bool Eos = false;
        bool bStartedEvalLoop = false;
        double StartEvalTime = 0.f;
        int32 StartContextLength = 0;

        void ThreadRun();
        void unsafeActivate(bool bReset);
        void UnsafeDeactivate();
        void unsafeInsertPrompt(FString);

        //backup method to check eos
        bool hasEnding(std::string const& fullString, std::string const& ending);

        void EmitErrorMessage(const FString& ErrorMessage, bool bLogErrorMessage=true);
       

    };

    void FLlama::InsertPrompt(FString v)
    {
        qMainToThread.enqueue([this, v = std::move(v)]() mutable { unsafeInsertPrompt(std::move(v)); });
    }

    void FLlama::unsafeInsertPrompt(FString v)
    {
        if (!Context) {
            UE_LOG(LogTemp, Error, TEXT("Llama not activated"));
            return;
        }
        string stdV = string(" ") + TCHAR_TO_UTF8(*v);
        vector<llama_token> line_inp = my_llama_tokenize(Context, stdV, res, false /* add bos */);
        embd_inp.insert(embd_inp.end(), line_inp.begin(), line_inp.end());
    }

    FLlama::FLlama() 
    {
        //We no longer startup the thread unless initialized
    }

    void FLlama::StartStopThread(bool bShouldRun) {
        if (bShouldRun)
        {
            if (bRunning)
            {
                return;
            }
            bRunning = true;

            qThread = thread([this]() {
                ThreadRun();
            });
        }
        else
        {
            bRunning = false;
            if (qThread.joinable())
            {
                qThread.join();
            }
        }
    }

    void FLlama::StopGenerating()
    {
        qMainToThread.enqueue([this]() 
        {
            Eos = true;
        });
    }
    void FLlama::ResumeGenerating()
    {
        qMainToThread.enqueue([this]()
        {
            Eos = false;
        });
    }

    bool FLlama::hasEnding(std::string const& fullString, std::string const& ending) {
        if (fullString.length() >= ending.length()) {
            return (0 == fullString.compare(fullString.length() - ending.length(), ending.length(), ending));
        }
        else {
            return false;
        }
    }

    void FLlama::EmitErrorMessage(const FString& ErrorMessage, bool bLogErrorMessage)
    {
        const FString ErrorMessageSafe = ErrorMessage;

        if (bLogErrorMessage)
        {
            UE_LOG(LogTemp, Error, TEXT("%s"), *ErrorMessageSafe);
        }

        qThreadToMain.enqueue([this, ErrorMessageSafe] {
            if (!OnErrorCb)
            {
                return;
            }

            OnErrorCb(ErrorMessageSafe);
        });
    }

    FString FLlama::ModelsRelativeRootPath()
    {
        FString AbsoluteFilePath;

#if PLATFORM_ANDROID
        //This is the path we're allowed to sample on android
        AbsoluteFilePath = FPaths::Combine(FPaths::Combine(FString(FAndroidMisc::GamePersistentDownloadDir()), "Models/"));
#else

        AbsoluteFilePath = FPaths::ConvertRelativePathToFull(FPaths::Combine(FPaths::ProjectSavedDir(), "Models/"));

#endif
        
        return AbsoluteFilePath;
    }

    FString FLlama::ParsePathIntoFullPath(const FString& InRelativeOrAbsolutePath)
    {
        FString FinalPath;

        //Is it a relative path?
        if (InRelativeOrAbsolutePath.StartsWith(TEXT(".")))
        {
            //relative path
            //UE_LOG(LogTemp, Log, TEXT("model returning relative path"));
            FinalPath = FPaths::ConvertRelativePathToFull(ModelsRelativeRootPath() + InRelativeOrAbsolutePath);
        }
        else
        {
            //Already an absolute path
            //UE_LOG(LogTemp, Log, TEXT("model returning absolute path"));
            FinalPath = FPaths::ConvertRelativePathToFull(InRelativeOrAbsolutePath);
        }

        return FinalPath;
    }

    void FLlama::ThreadRun()
    {
        UE_LOG(LogTemp, Warning, TEXT("%p Llama thread is running"), this);
        const int n_predict = -1;
        const int NKeep = 0;
        const int NBatch = Params.BatchCount;

        while (bRunning)
        {
            while (qMainToThread.processQ())
                ;
            if (!Model)
            {
                using namespace chrono_literals;
                this_thread::sleep_for(200ms);
                continue;
            }

            if (Eos && (int)embd_inp.size() <= n_consumed)
            {
                using namespace chrono_literals;
                this_thread::sleep_for(200ms);
                continue;
            }
            if (Eos == false && !bStartedEvalLoop)
            {
                bStartedEvalLoop = true;
                StartEvalTime = FPlatformTime::Seconds();
                StartContextLength = NPast; //(int32)last_n_tokens.size(); //(int32)embd_inp.size();

                qThreadToMain.enqueue([this] 
                {    
                    if (!OnStartEvalCb)
                    {
                        return;
                    }
                    OnStartEvalCb();
                });
            }


            Eos = false;

            const int NCtx = llama_n_ctx(Context);
            if (Embd.size() > 0)
            {
                // Note: NCtx - 4 here is to match the logic for commandline Prompt handling via
                // --Prompt or --file which uses the same value.
                int MaxEmbdSize = NCtx - 4;
                // Ensure the input doesn't exceed the context size by truncating embd if necessary.
                if ((int)Embd.size() > MaxEmbdSize)
                {
                    uint64 SkippedTokens = Embd.size() - MaxEmbdSize;
                    FString ErrorMsg = FString::Printf(TEXT("<<input too long: skipped %zu token%s>>"), 
                        SkippedTokens,
                        SkippedTokens != 1 ? "s" : "");
                    EmitErrorMessage(ErrorMsg);
                    Embd.resize(MaxEmbdSize);
                }

                // infinite Text generation via context swapping
                // if we run out of context:
                // - take the NKeep first tokens from the original Prompt (via NPast)
                // - take half of the last (NCtx - NKeep) tokens and recompute the logits in batches
                if (NPast + (int)Embd.size() > NCtx)
                {
                    UE_LOG(LogTemp, Warning, TEXT("%p context resetting"), this);
                    if (n_predict == -2)
                    {
                        FString ErrorMsg = TEXT("context full, stopping generation");
                        EmitErrorMessage(ErrorMsg);
                        UnsafeDeactivate();
                        continue;
                    }

                    const int NLeft = NPast - NKeep;
                    // always keep the first token - BOS
                    NPast = max(1, NKeep);

                    // insert NLeft/2 tokens at the start of embd from last_n_tokens
                    Embd.insert(Embd.begin(),
                                            last_n_tokens.begin() + NCtx - NLeft / 2 - Embd.size(),
                                            last_n_tokens.end() - Embd.size());
                }

                // evaluate tokens in batches
                // embd is typically prepared beforehand to fit within a batch, but not always

                for (int i = 0; i < (int)Embd.size(); i += NBatch)
                {
                    int NEval = (int)Embd.size() - i;
                    if (NEval > NBatch)
                    {
                        NEval = NBatch;
                    }
                    string Str = string{};
                    for (auto j = 0; j < NEval; ++j)
                        //    TODO: Replace this llama_detokenize_bpe with llama_detokenize when can be possible.
                        Str += llama_detokenize_bpe(Context, {Embd[i + j]});

                    if (bShouldLog)
                    {
                        UE_LOG(LogTemp, Warning, TEXT("%p eval tokens `%s`"), this, UTF8_TO_TCHAR(Str.c_str()));
                    }
                    if (llama_eval(Context, &Embd[i], NEval, NPast, Params.Threads))
                    {
                        FString ErrorMsg = TEXT("failed to eval");
                        EmitErrorMessage(ErrorMsg);
                        UnsafeDeactivate();
                        continue;
                    }
                    NPast += NEval;
                }
            }

            Embd.clear();

            bool haveHumanTokens = false;
            const FLLMModelAdvancedParams& P = Params.Advanced;

            if ((int)embd_inp.size() <= n_consumed)
            {
                llama_token id = 0;

                {
                    float* logits = llama_get_logits(Context);
                    int n_vocab = llama_n_vocab(Context);

                    vector<llama_token_data> candidates;
                    candidates.reserve(n_vocab);
                    for (llama_token token_id = 0; token_id < n_vocab; token_id++)
                    {
                        candidates.emplace_back(llama_token_data{token_id, logits[token_id], 0.0f});
                    }

                    llama_token_data_array candidates_p = {candidates.data(), candidates.size(), false};

                    // Apply penalties
                    float nl_logit = logits[llama_token_nl(Context)];
                    int last_n_repeat = min(min((int)last_n_tokens.size(), P.RepeatLastN), NCtx);
                    llama_sample_repetition_penalty(Context,
                                                    &candidates_p,
                                                    last_n_tokens.data() + last_n_tokens.size() - last_n_repeat,
                                                    last_n_repeat,
                                                    P.RepeatPenalty);
                    llama_sample_frequency_and_presence_penalties(  Context,
                                                                    &candidates_p,
                                                                    last_n_tokens.data() + last_n_tokens.size() - last_n_repeat,
                                                                    last_n_repeat,
                                                                    P.AlphaFrequency,
                                                                    P.AlphaPresence);
                    if (!P.PenalizeNl)
                    {
                        logits[llama_token_nl(Context)] = nl_logit;
                    }

                    if (P.Temp <= 0)
                    {
                        // Greedy sampling
                        id = llama_sample_token_greedy(Context, &candidates_p);
                    }
                    else
                    {
                        if (P.Mirostat == 1)
                        {
                            static float mirostat_mu = 2.0f * P.MirostatTau;
                            const int mirostat_m = 100;
                            llama_sample_temperature(Context, &candidates_p, P.Temp);
                            id = llama_sample_token_mirostat(
                                Context, &candidates_p, P.MirostatTau, P.MirostatEta, mirostat_m, &mirostat_mu);
                        }
                        else if (P.Mirostat == 2)
                        {
                            static float mirostat_mu = 2.0f * P.MirostatTau;
                            llama_sample_temperature(Context, &candidates_p, P.Temp);
                            id = llama_sample_token_mirostat_v2(
                                Context, &candidates_p, P.MirostatTau, P.MirostatEta, &mirostat_mu);
                        }
                        else
                        {
                            // Temperature sampling
                            llama_sample_top_k(Context, &candidates_p, P.TopK, 1);
                            llama_sample_tail_free(Context, &candidates_p, P.TfsZ, 1);
                            llama_sample_typical(Context, &candidates_p, P.TypicalP, 1);
                            llama_sample_top_p(Context, &candidates_p, P.TopP, 1);
                            llama_sample_temperature(Context, &candidates_p, P.Temp);
                            id = llama_sample_token(Context, &candidates_p);
                        }
                    }

                    last_n_tokens.erase(last_n_tokens.begin());
                    last_n_tokens.push_back(id);
                }

                // add it to the context
                Embd.push_back(id);
            }
            else
            {
                // some user input remains from Prompt or interaction, forward it to processing
                while ((int)embd_inp.size() > n_consumed)
                {
                    const int tokenId = embd_inp[n_consumed];
                    Embd.push_back(tokenId);
                    last_n_tokens.erase(last_n_tokens.begin());
                    last_n_tokens.push_back(embd_inp[n_consumed]);
                    haveHumanTokens = true;
                    ++n_consumed;
                    if ((int)Embd.size() >= NBatch)
                    {
                        break;
                    }
                }
            }

            // TODO: Revert these changes to the commented code when the llama.cpp add the llama_detokenize function.
            
            // display Text
            // for (auto id : embd)
            // {
            //     FString token = llama_detokenize(Context, id);
            //     qThreadToMain.enqueue([token = move(token), this]() {
            //         if (!OnTokenCb)
            //             return;
            //         OnTokenCb(move(token));
            //     });
            // }
            
            FString token = UTF8_TO_TCHAR(llama_detokenize_bpe(Context, Embd).c_str());

            //Debug block
            //NB: appears full history is not being input back to the model,
            // does Llama not need input copying for proper context?
            //FString history1 = UTF8_TO_TCHAR(llama_detokenize_bpe(Context, embd_inp).c_str()); 
            //FString history2 = UTF8_TO_TCHAR(llama_detokenize_bpe(Context, last_n_tokens).c_str());
            //UE_LOG(LogTemp, Log, TEXT("history1: %s, history2: %s"), *history1, *history2);
            int32 NewContextLength = NPast; //(int32)last_n_tokens.size();

            
            qThreadToMain.enqueue([token = std::move(token), NewContextLength,  this] {
                if (!OnTokenCb)
                    return;
                OnTokenCb(std::move(token), NewContextLength);
            });
            ////////////////////////////////////////////////////////////////////////

            bool const hasStopSeq = [&]
            {
                if (StopSequences.empty())
                    return false;
                if (haveHumanTokens)
                    return false;                

                for (vector<llama_token> stopSeq : StopSequences)
                {
                    FString sequence = UTF8_TO_TCHAR(llama_detokenize_bpe(Context, stopSeq).c_str());
                    sequence = sequence.TrimStartAndEnd();

                    vector<llama_token> endSeq;
                    for (unsigned i = 0U; i < stopSeq.size(); ++i)
                    {
                        endSeq.push_back(last_n_tokens[last_n_tokens.size() - stopSeq.size() + i]);
                    }
                    FString endString = UTF8_TO_TCHAR(llama_detokenize_bpe(Context, endSeq).c_str());
                    
                    if (bShouldLog) 
                    {
                        UE_LOG(LogTemp, Log, TEXT("stop vs end: #%s# vs #%s#"), *sequence, *endString);
                    }
                    if (endString.Contains(sequence))
                    {
                        UE_LOG(LogTemp, Log, TEXT("String match found, eos triggered."));
                        return true;
                    }
                    

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

            if ((!Embd.empty() && Embd.back() == llama_token_eos(Context)) || hasStopSeq)
            {
                UE_LOG(LogTemp, Warning, TEXT("%p EOS"), this);
                Eos = true;
                const bool stopSeqSafe = hasStopSeq;
                const int32 DeltaTokens = NewContextLength - StartContextLength;
                const double EosTime = FPlatformTime::Seconds();
                const float TokensPerSecond = double(DeltaTokens) / (EosTime - StartEvalTime);

                bStartedEvalLoop = false;
                

                //notify main thread we're done
                qThreadToMain.enqueue([stopSeqSafe, TokensPerSecond, this] 
                {
                    if (!OnEosCb)
                        return;
                    OnEosCb(stopSeqSafe, TokensPerSecond);
                });
            }
        }
        UnsafeDeactivate();
        UE_LOG(LogTemp, Warning, TEXT("%p Llama thread stopped"), this);
    }

    FLlama::~FLlama()
    {
        bRunning = false;
        if (qThread.joinable())
        {
            qThread.join();
        }
    }

    void FLlama::Process()
    {
        while (qThreadToMain.processQ())
            ;
    }

    void FLlama::Activate(bool bReset, const FLLMModelParams& InParams)
    {
        Params = InParams;
        qMainToThread.enqueue([bReset, this]() mutable {
            unsafeActivate(bReset);
        });
    }

    void FLlama::Deactivate()
    {
        qMainToThread.enqueue([this]() { UnsafeDeactivate(); });
    }

    void FLlama::unsafeActivate(bool bReset)
    {
        UE_LOG(LogTemp, Warning, TEXT("%p Loading LLM model %p bReset: %d"), this, Model, bReset);
        if (bReset)
            UnsafeDeactivate();
        if (Model)
            return;
        
        llama_context_params lparams = [this]()
        {
            llama_context_params lparams = llama_context_default_params();
            // -eps 1e-5 -t 8 -ngl 50
            lparams.n_gpu_layers = Params.GPULayers;
            lparams.n_ctx = Params.MaxContextLength;

            bool bIsRandomSeed = Params.Seed == -1;

            if(bIsRandomSeed){
                lparams.seed = time(nullptr);
            }
            else
            {
                lparams.seed = Params.Seed;
            }


            return lparams;
        }();

        FString FullModelPath = ParsePathIntoFullPath(Params.PathToModel);

        UE_LOG(LogTemp, Log, TEXT("File at %s exists? %d"), *FullModelPath, FPaths::FileExists(FullModelPath));

        Model = llama_load_model_from_file(TCHAR_TO_UTF8(*FullModelPath), lparams);
        if (!Model)
        {
            FString ErrorMessage = FString::Printf(TEXT("%p unable to load model at %s"), this, *FullModelPath);

            EmitErrorMessage(ErrorMessage);
            UnsafeDeactivate();
            return;
        }
        Context = llama_new_context_with_model(Model, lparams);
        NPast = 0;

        UE_LOG(LogTemp, Warning, TEXT("%p model context set to %p"), this, Context);

        // tokenize the Prompt
        string stdPrompt = string(" ") + TCHAR_TO_UTF8(*Params.Prompt);
        embd_inp = my_llama_tokenize(Context, stdPrompt, res, true /* add bos */);
        if (!Params.StopSequences.IsEmpty())
        {
            for (int i = 0; i < Params.StopSequences.Num(); ++i)
            {
                const FString& stopSeq = Params.StopSequences[i];
                string str = string{TCHAR_TO_UTF8(*stopSeq)};
                if (::isalnum(str[0]))
                    str = " " + str;
                vector<llama_token> seq = my_llama_tokenize(Context, str, res, false /* add bos */);
                StopSequences.emplace_back(std::move(seq));
            }
        }
        else
            StopSequences.clear();

        const int n_ctx = llama_n_ctx(Context);

        if ((int)embd_inp.size() > n_ctx - 4)
        {
            FString ErrorMessage = FString::Printf(TEXT("prompt is too long (%d tokens, max %d)"), (int)embd_inp.size(), n_ctx - 4);
            EmitErrorMessage(ErrorMessage);
            UnsafeDeactivate();
            return;
        }

        // do one empty run to warm up the model
        {
            const vector tmp = {
                llama_token_bos(Context),
            };
            llama_eval(Context, tmp.data(), tmp.size(), 0, Params.Threads);
            llama_reset_timings(Context);
        }
        last_n_tokens.resize(n_ctx);
        fill(last_n_tokens.begin(), last_n_tokens.end(), 0);
        n_consumed = 0;
    }

    void FLlama::UnsafeDeactivate()
    {
        bStartedEvalLoop = false;
        StopSequences.clear();
        UE_LOG(LogTemp, Warning, TEXT("%p Unloading LLM model %p"), this, Model);
        if (!Model)
            return;
        llama_print_timings(Context);
        llama_free(Context);
        Context = nullptr;

        //Todo: potentially not reset model if same model is loaded
        llama_free_model(Model);
        Model = nullptr;

        //Reset signal.
        qThreadToMain.enqueue([this] {
            if (!OnContextResetCb)
            {
                return;
            }
            
            OnContextResetCb();
        });

        
        
    }
} // namespace Internal

ULlamaComponent::ULlamaComponent(const FObjectInitializer &ObjectInitializer)
    : UActorComponent(ObjectInitializer), llama(make_unique<Internal::FLlama>())
{
    PrimaryComponentTick.bCanEverTick = true;
    PrimaryComponentTick.bStartWithTickEnabled = true;
    llama->OnTokenCb = [this](FString NewToken, int32 NewContextLength) 
    { 
        if (bSyncPromptHistory)
        {
            ModelState.PromptHistory.Append(NewToken);
        }
        ModelState.ContextLength = NewContextLength;
        OnNewTokenGenerated.Broadcast(std::move(NewToken));
    };
    llama->OnEosCb = [this](bool StopTokenCausedEos, float TokensPerSecond)
    {
        ModelState.LastTokensPerSecond = TokensPerSecond;
        OnEndOfStream.Broadcast(StopTokenCausedEos, TokensPerSecond);
    };
    llama->OnStartEvalCb = [this]()
    {
        OnStartEval.Broadcast();
    };
    llama->OnContextResetCb = [this]()
    {
        if (bSyncPromptHistory) 
        {
            ModelState.PromptHistory.Empty();
        }
        OnContextReset.Broadcast();
    };
    llama->OnErrorCb = [this](FString ErrorMessage)
    {
        OnError.Broadcast(ErrorMessage);
    };
}

ULlamaComponent::~ULlamaComponent() = default;

void ULlamaComponent::Activate(bool bReset)
{
    Super::Activate(bReset);

    //if it hasn't been started, this will start it
    llama->StartStopThread(true);
    llama->bShouldLog = bDebugLogModelOutput;
    llama->Activate(bReset, ModelParams);
}

void ULlamaComponent::Deactivate()
{
    llama->Deactivate();
    Super::Deactivate();
}

void ULlamaComponent::TickComponent(float DeltaTime,
                                    ELevelTick TickType,
                                    FActorComponentTickFunction* ThisTickFunction)
{
    Super::TickComponent(DeltaTime, TickType, ThisTickFunction);
    llama->Process();
}

auto ULlamaComponent::InsertPrompt(const FString& v) -> void
{
    llama->InsertPrompt(v);
}

void ULlamaComponent::StartStopQThread(bool bShouldRun)
{
    llama->StartStopThread(bShouldRun);
}

void ULlamaComponent::StopGenerating()
{
    llama->StopGenerating();
}

void ULlamaComponent::ResumeGenerating()
{
    llama->ResumeGenerating();
}

TArray<FString> ULlamaComponent::DebugListDirectoryContent(const FString& InPath)
{
    TArray<FString> Entries;

    FString FullPathDirectory;

    if (InPath.Contains(TEXT("<ProjectDir>")))
    {
        FString Remainder = InPath.Replace(TEXT("<ProjectDir>"), TEXT(""));

        FullPathDirectory = FPaths::ProjectDir() + Remainder;
    }
    else if (InPath.Contains(TEXT("<Content>")))
    {
        FString Remainder = InPath.Replace(TEXT("<Content>"), TEXT(""));

        FullPathDirectory = FPaths::ProjectContentDir() + Remainder;
    }
    else if (InPath.Contains(TEXT("<External>")))
    {
        FString Remainder = InPath.Replace(TEXT("<Content>"), TEXT(""));

#if PLATFORM_ANDROID
        FString ExternalStoragePath = FString(FAndroidMisc::GamePersistentDownloadDir());
        FullPathDirectory = ExternalStoragePath + Remainder;
#else
        UE_LOG(LogTemp, Warning, TEXT("Externals not valid in this context!"));
        FullPathDirectory = Internal::FLlama::ParsePathIntoFullPath(Remainder);
#endif
    }
    else
    {
        FullPathDirectory = Internal::FLlama::ParsePathIntoFullPath(InPath);
    }
    
    IFileManager& FileManager = IFileManager::Get();

    FullPathDirectory = FPaths::ConvertRelativePathToFull(FullPathDirectory);

    FullPathDirectory = FileManager.ConvertToAbsolutePathForExternalAppForRead(*FullPathDirectory);

    Entries.Add(FullPathDirectory);

    UE_LOG(LogTemp, Log, TEXT("Listing contents of <%s>"), *FullPathDirectory);

    
    

    // Find directories
    TArray<FString> Directories;
    FString FinalPath = FullPathDirectory / TEXT("*");
    FileManager.FindFiles(Directories, *FinalPath, false, true);
    for (FString Entry : Directories)
    {
        FString FullPath = FullPathDirectory / Entry;
        if (FileManager.DirectoryExists(*FullPath)) // Filter for directories
        {
            UE_LOG(LogTemp, Log, TEXT("Found directory: %s"), *Entry);
            Entries.Add(Entry);
        }
    }

    // Find files
    TArray<FString> Files;
    FileManager.FindFiles(Files, *FullPathDirectory, TEXT("*.*")); // Find all entries
    for (FString Entry : Files)
    {
        FString FullPath = FullPathDirectory / Entry;
        if (!FileManager.DirectoryExists(*FullPath)) // Filter out directories
        {
            UE_LOG(LogTemp, Log, TEXT("Found file: %s"), *Entry);
            Entries.Add(Entry);
        }
    }

    return Entries;
}