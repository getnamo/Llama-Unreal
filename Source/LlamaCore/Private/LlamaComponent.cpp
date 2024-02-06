// 2023 (c) Mika Pi, Modifications Getnamo

#include "LlamaComponent.h"
#include <atomic>
#include <deque>
#include <thread>
#include <functional>
#include <mutex>
#include "HAL/PlatformTime.h"
#include "Misc/Paths.h"
#include "HAL/FileManager.h"
#include "common/common.h"
#include "common/gguf.h"

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

namespace
{
    class Q
    {
    public:
        void Enqueue(function<void()>);
        bool ProcessQ();

    private:
        deque<function<void()>> q;
        mutex mutex_;
    };

    void Q::Enqueue(function<void()> v)
    {
        lock_guard l(mutex_);
        q.emplace_back(std::move(v));
    }

    bool Q::ProcessQ() {
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
        const int n = llama_tokenize(llama_get_model(Context), Text.c_str(), Text.length(), Res.data(), Res.size(), AddBos, false);   //do not tokenize special for first pass
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

        void UpdateParams(const FLLMModelParams& Params);

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
        vector<llama_token> EmbdInput;
        vector<llama_token> Embd;
        vector<llama_token> Res;
        int NPast = 0;
        vector<llama_token> LastNTokens;
        int NConsumed = 0;
        bool Eos = false;
        bool bStartedEvalLoop = false;
        double StartEvalTime = 0.f;
        int32 StartContextLength = 0;

        void ThreadRun();
        void UnsafeActivate(bool bReset);
        void UnsafeDeactivate();
        void UnsafeInsertPrompt(FString);

        //backup method to check eos, until we get proper prompt templating support
        bool HasEnding(std::string const& FullString, std::string const& Ending);

        void EmitErrorMessage(const FString& ErrorMessage, bool bLogErrorMessage=true);
    };

    void FLlama::InsertPrompt(FString v)
    {
        qMainToThread.Enqueue([this, v = std::move(v)]() mutable { UnsafeInsertPrompt(std::move(v)); });
    }

    void FLlama::UnsafeInsertPrompt(FString v)
    {
        if (!Context) {
            UE_LOG(LogTemp, Error, TEXT("Llama not activated"));
            return;
        }
        string stdV = string(" ") + TCHAR_TO_UTF8(*v);
        vector<llama_token> line_inp = my_llama_tokenize(Context, stdV, Res, false /* add bos */);
        EmbdInput.insert(EmbdInput.end(), line_inp.begin(), line_inp.end());
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
        qMainToThread.Enqueue([this]() 
        {
            Eos = true;
        });
    }
    void FLlama::ResumeGenerating()
    {
        qMainToThread.Enqueue([this]()
        {
            Eos = false;
        });
    }

    void FLlama::UpdateParams(const FLLMModelParams& InParams)
    {
        FLLMModelParams SafeParams = InParams;
        qMainToThread.Enqueue([this, SafeParams]() mutable
        {
            this->Params = SafeParams;
        });
    }

    bool FLlama::HasEnding(std::string const& FullString, std::string const& Ending) {
        if (FullString.length() >= Ending.length()) {
            return (0 == FullString.compare(FullString.length() - Ending.length(), Ending.length(), Ending));
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

        qThreadToMain.Enqueue([this, ErrorMessageSafe] {
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
        const int NPredict = -1;
        const int NKeep = 0;
        const int NBatch = Params.BatchCount;

        while (bRunning)
        {
            while (qMainToThread.ProcessQ())
                ;
            if (!Model)
            {
                using namespace chrono_literals;
                this_thread::sleep_for(200ms);
                continue;
            }

            if (Eos && (int)EmbdInput.size() <= NConsumed)
            {
                using namespace chrono_literals;
                this_thread::sleep_for(200ms);
                continue;
            }
            if (Eos == false && !bStartedEvalLoop)
            {
                bStartedEvalLoop = true;
                StartEvalTime = FPlatformTime::Seconds();
                StartContextLength = NPast; //(int32)LastNTokens.size(); //(int32)embd_inp.size();

                qThreadToMain.Enqueue([this] 
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
                    if (NPredict == -2)
                    {
                        FString ErrorMsg = TEXT("context full, stopping generation");
                        EmitErrorMessage(ErrorMsg);
                        UnsafeDeactivate();
                        continue;
                    }

                    const int NLeft = NPast - NKeep;
                    // always keep the first token - BOS
                    NPast = max(1, NKeep);

                    // insert NLeft/2 tokens at the start of embd from LastNTokens
                    Embd.insert(Embd.begin(),
                                            LastNTokens.begin() + NCtx - NLeft / 2 - Embd.size(),
                                            LastNTokens.end() - Embd.size());
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
                    

                    if (bShouldLog)
                    {
                        string Str = string{};
                        for (auto j = 0; j < NEval; ++j)
                        {
                            Str += llama_detokenize_bpe(Context, { Embd[i + j] });
                        }
                        UE_LOG(LogTemp, Warning, TEXT("%p eval tokens `%s`"), this, UTF8_TO_TCHAR(Str.c_str()));
                    }

                    if (llama_decode(Context, llama_batch_get_one(&Embd[i], NEval, NPast, 0)))
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

            bool bHaveHumanTokens = false;
            const FLLMModelAdvancedParams& P = Params.Advanced;

            if ((int)EmbdInput.size() <= NConsumed)
            {
                llama_token ID = 0;

                {
                    float* Logits = llama_get_logits(Context);
                    int NVocab = llama_n_vocab(llama_get_model(Context));

                    vector<llama_token_data> Candidates;
                    Candidates.reserve(NVocab);
                    for (llama_token TokenID = 0; TokenID < NVocab; TokenID++)
                    {
                        Candidates.emplace_back(llama_token_data{TokenID, Logits[TokenID], 0.0f});
                    }

                    llama_token_data_array CandidatesP = {Candidates.data(), Candidates.size(), false};

                    // Apply penalties
                    float NLLogit = Logits[llama_token_nl(llama_get_model(Context))];
                    int LastNRepeat = min(min((int)LastNTokens.size(), P.RepeatLastN), NCtx);

                    llama_sample_repetition_penalties(  Context,
                                                        &CandidatesP,
                                                        LastNTokens.data() + LastNTokens.size() - LastNRepeat,
                                                        LastNRepeat,
                                                        P.RepeatPenalty,
                                                        P.AlphaFrequency,
                                                        P.AlphaPresence);
                    if (!P.PenalizeNl)
                    {
                        Logits[llama_token_nl(llama_get_model(Context))] = NLLogit;
                    }

                    if (P.Temp <= 0)
                    {
                        // Greedy sampling
                        ID = llama_sample_token_greedy(Context, &CandidatesP);
                    }
                    else
                    {
                        if (P.Mirostat == 1)
                        {
                            static float MirostatMu = 2.0f * P.MirostatTau;
                            llama_sample_temp(Context, &CandidatesP, P.Temp);
                            ID = llama_sample_token_mirostat(
                                Context, &CandidatesP, P.MirostatTau, P.MirostatEta, P.MirostatM, &MirostatMu);
                        }
                        else if (P.Mirostat == 2)
                        {
                            static float MirostatMu = 2.0f * P.MirostatTau;
                            llama_sample_temp(Context, &CandidatesP, P.Temp);
                            ID = llama_sample_token_mirostat_v2(
                                Context, &CandidatesP, P.MirostatTau, P.MirostatEta, &MirostatMu);
                        }
                        else
                        {
                            // Temperature sampling
                            llama_sample_top_k(Context, &CandidatesP, P.TopK, 1);
                            llama_sample_tail_free(Context, &CandidatesP, P.TfsZ, 1);
                            llama_sample_typical(Context, &CandidatesP, P.TypicalP, 1);
                            llama_sample_top_p(Context, &CandidatesP, P.TopP, 1);
                            llama_sample_temp(Context, &CandidatesP, P.Temp);
                            ID = llama_sample_token(Context, &CandidatesP);
                        }
                    }

                    LastNTokens.erase(LastNTokens.begin());
                    LastNTokens.push_back(ID);
                }

                // add it to the context
                Embd.push_back(ID);
            }
            else
            {
                // some user input remains from Prompt or interaction, forward it to processing
                while ((int)EmbdInput.size() > NConsumed)
                {
                    const int tokenId = EmbdInput[NConsumed];
                    Embd.push_back(tokenId);
                    LastNTokens.erase(LastNTokens.begin());
                    LastNTokens.push_back(EmbdInput[NConsumed]);
                    bHaveHumanTokens = true;
                    ++NConsumed;
                    if ((int)Embd.size() >= NBatch)
                    {
                        break;
                    }
                }
            }

            // TODO: Revert these changes to the commented code when the llama.cpp add the llama_detokenize function.
            
            // display Text
            // for (auto Id : embd)
            // {
            //     FString token = llama_detokenize(Context, Id);
            //     qThreadToMain.Enqueue([token = move(token), this]() {
            //         if (!OnTokenCb)
            //             return;
            //         OnTokenCb(move(token));
            //     });
            // }
            
            FString Token = UTF8_TO_TCHAR(llama_detokenize_bpe(Context, Embd).c_str());

            //Debug block
            //NB: appears full history is not being input back to the model,
            // does Llama not need input copying for proper context?
            //FString history1 = UTF8_TO_TCHAR(llama_detokenize_bpe(Context, embd_inp).c_str()); 
            //FString history2 = UTF8_TO_TCHAR(llama_detokenize_bpe(Context, LastNTokens).c_str());
            //UE_LOG(LogTemp, Log, TEXT("history1: %s, history2: %s"), *history1, *history2);
            int32 NewContextLength = NPast; //(int32)LastNTokens.size();

            
            qThreadToMain.Enqueue([token = std::move(Token), NewContextLength,  this] {
                if (!OnTokenCb)
                    return;
                OnTokenCb(std::move(token), NewContextLength);
            });
            ////////////////////////////////////////////////////////////////////////

                
            auto StringStopTest = [&]
            {
                if (StopSequences.empty())
                    return false;
                if (bHaveHumanTokens)
                    return false;                

                for (vector<llama_token> StopSeq : StopSequences)
                {
                    FString Sequence = UTF8_TO_TCHAR(llama_detokenize_bpe(Context, StopSeq).c_str());
                    Sequence = Sequence.TrimStartAndEnd();

                    vector<llama_token> EndSeq;
                    for (unsigned i = 0U; i < StopSeq.size(); ++i)
                    {
                        EndSeq.push_back(LastNTokens[LastNTokens.size() - StopSeq.size() + i]);
                    }
                    FString EndString = UTF8_TO_TCHAR(llama_detokenize_bpe(Context, EndSeq).c_str());
                    
                    if (bShouldLog) 
                    {
                        UE_LOG(LogTemp, Log, TEXT("stop vs end: #%s# vs #%s#"), *Sequence, *EndString);
                    }
                    if (EndString.Contains(Sequence))
                    {
                        UE_LOG(LogTemp, Warning, TEXT("String match found, String EOS triggered."));
                        return true;
                    }
                    

                    if (LastNTokens.size() < StopSeq.size())
                        return false;
                    bool bMatch = true;
                    for (unsigned i = 0U; i < StopSeq.size(); ++i)
                        if (LastNTokens[LastNTokens.size() - StopSeq.size() + i] != StopSeq[i])
                        {
                            bMatch = false;
                            break;
                        }
                    if (bMatch)
                        return true;
                }
                return false;
            };

            bool EOSTriggered = false;
            bool bStandardTokenEOS = (!Embd.empty() && Embd.back() == llama_token_eos(llama_get_model(Context)));

            //check
            if (!bStandardTokenEOS)
            {
                EOSTriggered = StringStopTest();
            }
            else
            {
                UE_LOG(LogTemp, Warning, TEXT("%p Standard EOS triggered"), this);
                EOSTriggered = true;
            }

            if (EOSTriggered)
            {
                //UE_LOG(LogTemp, Warning, TEXT("%p EOS"), this);
                Eos = true;
                const bool StopSeqSafe = EOSTriggered;
                const int32 DeltaTokens = NewContextLength - StartContextLength;
                const double EosTime = FPlatformTime::Seconds();
                const float TokensPerSecond = double(DeltaTokens) / (EosTime - StartEvalTime);

                bStartedEvalLoop = false;
                

                //notify main thread we're done
                qThreadToMain.Enqueue([StopSeqSafe, TokensPerSecond, this] 
                {
                    if (!OnEosCb)
                    {
                        return;
                    }
                    OnEosCb(StopSeqSafe, TokensPerSecond);
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
        while (qThreadToMain.ProcessQ())
            ;
    }

    void FLlama::Activate(bool bReset, const FLLMModelParams& InParams)
    {
        FLLMModelParams SafeParams = InParams;
        
        qMainToThread.Enqueue([bReset, this, SafeParams]() mutable
        {
            this->Params = SafeParams;
            UnsafeActivate(bReset);
        });
    }

    void FLlama::Deactivate()
    {
        qMainToThread.Enqueue([this]() { UnsafeDeactivate(); });
    }

    void FLlama::UnsafeActivate(bool bReset)
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

        llama_model_params mParams = llama_model_default_params();
        mParams.n_gpu_layers = Params.MaxContextLength;

        FString FullModelPath = ParsePathIntoFullPath(Params.PathToModel);

        UE_LOG(LogTemp, Log, TEXT("File at %s exists? %d"), *FullModelPath, FPaths::FileExists(FullModelPath));

        Model = llama_load_model_from_file(TCHAR_TO_UTF8(*FullModelPath), mParams);
        if (!Model)
        {
            FString ErrorMessage = FString::Printf(TEXT("%p unable to load model at %s"), this, *FullModelPath);

            EmitErrorMessage(ErrorMessage);
            UnsafeDeactivate();
            return;
        }

        //Read GGUF info
        gguf_ex_read_0(TCHAR_TO_UTF8(*FullModelPath));

        Context = llama_new_context_with_model(Model, lparams);
        NPast = 0;

        UE_LOG(LogTemp, Warning, TEXT("%p model context set to %p"), this, Context);

        // tokenize the Prompt
        string StdPrompt = string(" ") + TCHAR_TO_UTF8(*Params.Prompt);
        EmbdInput = my_llama_tokenize(Context, StdPrompt, Res, true /* add bos */);
        if (!Params.StopSequences.IsEmpty())
        {
            for (int i = 0; i < Params.StopSequences.Num(); ++i)
            {
                const FString& stopSeq = Params.StopSequences[i];
                string str = string{TCHAR_TO_UTF8(*stopSeq)};
                if (::isalnum(str[0]))
                    str = " " + str;
                vector<llama_token> seq = my_llama_tokenize(Context, str, Res, false /* add bos */);
                StopSequences.emplace_back(std::move(seq));
            }
        }
        else
        {
            StopSequences.clear();
        }

        const int NCtx = llama_n_ctx(Context);

        if ((int)EmbdInput.size() > NCtx - 4)
        {
            FString ErrorMessage = FString::Printf(TEXT("prompt is too long (%d tokens, max %d)"), (int)EmbdInput.size(), NCtx - 4);
            EmitErrorMessage(ErrorMessage);
            UnsafeDeactivate();
            return;
        }

        // do one empty run to warm up the model
        llama_set_n_threads(Context, Params.Threads, Params.Threads);

        {
            vector<llama_token> Tmp = {
                llama_token_bos(llama_get_model(Context)),
            };
            llama_decode(Context, llama_batch_get_one(Tmp.data(), Tmp.size(), 0, 0));
            llama_reset_timings(Context);
        }
        LastNTokens.resize(NCtx);
        fill(LastNTokens.begin(), LastNTokens.end(), 0);
        NConsumed = 0;
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
        qThreadToMain.Enqueue([this] {
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

            //Track partials - Sentences
            if (ModelParams.Advanced.bEmitPartials)
            {
                bool bSplitFound = false;
                //Check new token for separators
                for (const FString& Separator : ModelParams.Advanced.PartialsSeparators)
                {
                    if (NewToken.Contains(Separator))
                    {
                        bSplitFound = true;
                    }
                }

                if (bSplitFound)
                {
                    //Sync Chat history on partial period
                    ModelState.ChatHistory = GetStructuredHistory();

                    //Grab partial from last message
                    const FStructuredChatMessage& Message = ModelState.ChatHistory.History.Last();

                    //Confirm it's from the assistant
                    if (Message.Role == EChatTemplateRole::Assistant)
                    {
                        //Look for period preceding this one
                        FString Sentence = GetLastSentence(Message.Content);

                        if (!Sentence.IsEmpty())
                        {
                            OnPartialParsed.Broadcast(Sentence);
                        }
                    }
                }

            }
        }
        ModelState.ContextLength = NewContextLength;
        OnNewTokenGenerated.Broadcast(std::move(NewToken));
    };
    llama->OnEosCb = [this](bool StopTokenCausedEos, float TokensPerSecond)
    {
        ModelState.LastTokensPerSecond = TokensPerSecond;

        if (ModelParams.Advanced.bSyncStructuredChatHistory)
        {
            ModelState.ChatHistory = GetStructuredHistory();
        }
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

    //NB this list should be static...
    //For now just add Chat ML
    FChatTemplate Template;
    Template.System = TEXT("<|im_start|>system");
    Template.User = TEXT("<|im_start|>user");
    Template.Assistant = TEXT("<|im_start|>assistant");
    Template.CommonSuffix = TEXT("<|im_end|>");
    Template.Delimiter = TEXT("\n");

    CommonChatTemplates.Add(TEXT("ChatML"), Template);

    //Temp hack default to ChatML
    ModelParams.ChatTemplate = Template;


    //All sentence ending formatting.
    ModelParams.Advanced.PartialsSeparators.Add(TEXT("."));
    ModelParams.Advanced.PartialsSeparators.Add(TEXT("?"));
    ModelParams.Advanced.PartialsSeparators.Add(TEXT("!"));
}

ULlamaComponent::~ULlamaComponent() = default;

void ULlamaComponent::Activate(bool bReset)
{
    Super::Activate(bReset);

    //Check our role
    if (ModelParams.ModelRole != EChatTemplateRole::Unknown)
    {
    }

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

void ULlamaComponent::InsertPrompt(const FString& Prompt)
{
    llama->InsertPrompt(Prompt);
}

FString ULlamaComponent::WrapPromptForRole(const FString& Content, EChatTemplateRole Role, bool AppendModelRolePrefix)
{
    FString FinalInputText = TEXT("");
    if (Role == EChatTemplateRole::User)
    {
        FinalInputText = ModelParams.ChatTemplate.User + ModelParams.ChatTemplate.Delimiter + Content + ModelParams.ChatTemplate.CommonSuffix + ModelParams.ChatTemplate.Delimiter;
    }
    else if (Role == EChatTemplateRole::Assistant)
    {
        FinalInputText = ModelParams.ChatTemplate.Assistant + ModelParams.ChatTemplate.Delimiter + Content + ModelParams.ChatTemplate.CommonSuffix + ModelParams.ChatTemplate.Delimiter;
    }
    else if (Role == EChatTemplateRole::System)
    {
        FinalInputText = ModelParams.ChatTemplate.System + ModelParams.ChatTemplate.Delimiter + Content + ModelParams.ChatTemplate.CommonSuffix + ModelParams.ChatTemplate.Delimiter;
    }
    else
    {
        return Content;
    }

    if (AppendModelRolePrefix) 
    {
        //Preset role reply
        FinalInputText += GetModelRolePrefix();
    }

    return FinalInputText;
}

FString ULlamaComponent::GetModelRolePrefix()
{
    FString Prefix = TEXT("");

    if (ModelParams.ModelRole != EChatTemplateRole::Unknown)
    {
        if (ModelParams.ModelRole == EChatTemplateRole::Assistant)
        {
            Prefix += ModelParams.ChatTemplate.Assistant + ModelParams.ChatTemplate.Delimiter;
        }
        else if (ModelParams.ModelRole == EChatTemplateRole::User)
        {
            Prefix += ModelParams.ChatTemplate.User + ModelParams.ChatTemplate.Delimiter;
        }
        else if (ModelParams.ModelRole == EChatTemplateRole::System)
        {
            Prefix += ModelParams.ChatTemplate.System + ModelParams.ChatTemplate.Delimiter;
        }
    }
    return Prefix;
}

void ULlamaComponent::InsertPromptTemplated(const FString& Content, EChatTemplateRole Role)
{
    llama->InsertPrompt(WrapPromptForRole(Content, Role, true));
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

void ULlamaComponent::SyncParamsToLlama()
{
    llama->UpdateParams(ModelParams);
}

FString ULlamaComponent::GetTemplateStrippedPrompt()
{
    FString CleanPrompt;
    
    CleanPrompt = ModelState.PromptHistory.Replace(*ModelParams.ChatTemplate.User, TEXT(""));
    CleanPrompt = CleanPrompt.Replace(*ModelParams.ChatTemplate.Assistant, TEXT(""));
    CleanPrompt = CleanPrompt.Replace(*ModelParams.ChatTemplate.System, TEXT(""));
    CleanPrompt = CleanPrompt.Replace(*ModelParams.ChatTemplate.CommonSuffix, TEXT(""));

    return CleanPrompt;
}

FStructuredChatMessage ULlamaComponent::FirstChatMessageInHistory(const FString& History, FString& Remainder)
{
    FStructuredChatMessage Message;
    Message.Role = EChatTemplateRole::Unknown;

    int32 StartIndex = INDEX_NONE;
    FString StartRole = TEXT("");
    int32 StartSystem = History.Find(ModelParams.ChatTemplate.System, ESearchCase::CaseSensitive, ESearchDir::FromStart, -1);
    int32 StartAssistant = History.Find(ModelParams.ChatTemplate.Assistant, ESearchCase::CaseSensitive, ESearchDir::FromStart, -1);
    int32 StartUser = History.Find(ModelParams.ChatTemplate.User, ESearchCase::CaseSensitive, ESearchDir::FromStart, -1);

    //Early exit
    if (StartSystem == INDEX_NONE &&
        StartAssistant == INDEX_NONE &&
        StartUser == INDEX_NONE)
    {
        //Failed end find
        Remainder = TEXT("");
        return Message;
    }

    //so they aren't the lowest (-1)
    if (StartSystem == INDEX_NONE)
    {
        StartSystem = INT32_MAX;
    }
    if (StartAssistant == INDEX_NONE)
    {
        StartAssistant = INT32_MAX;
    }
    if (StartUser == INDEX_NONE)
    {
        StartUser = INT32_MAX;
    }

    
    if (StartSystem <= StartAssistant &&
        StartSystem <= StartUser)
    {
        StartIndex = StartSystem;
        StartRole = ModelParams.ChatTemplate.System;
        Message.Role = EChatTemplateRole::System;
    }

    else if (
        StartUser <= StartAssistant &&
        StartUser <= StartSystem)
    {
        StartIndex = StartUser;
        StartRole = ModelParams.ChatTemplate.User;
        Message.Role = EChatTemplateRole::User;
    }

    else if (
        StartAssistant <= StartUser &&
        StartAssistant <= StartSystem)
    {
        StartIndex = StartAssistant;
        StartRole = ModelParams.ChatTemplate.Assistant;
        Message.Role = EChatTemplateRole::Assistant;
    }

    //Look for system role first
    if (StartIndex != INDEX_NONE)
    {
        const FString& CommonSuffix = ModelParams.ChatTemplate.CommonSuffix;

        StartIndex = StartIndex + StartRole.Len();

        int32 EndIndex = History.Find(CommonSuffix, ESearchCase::CaseSensitive, ESearchDir::FromStart, StartIndex);

        if (EndIndex != INDEX_NONE)
        {
            int32 Count = EndIndex - StartIndex;
            Message.Content = History.Mid(StartIndex, Count).TrimStartAndEnd();

            EndIndex = EndIndex + CommonSuffix.Len();

            Remainder = History.RightChop(EndIndex);
        }
        else
        {
            //No ending, assume all content belongs to this bit
            Message.Content = History.RightChop(StartIndex).TrimStartAndEnd();
            Remainder = TEXT("");
        }
    }
    return Message;
}

FStructuredChatHistory ULlamaComponent::GetStructuredHistory()
{
    FString WorkingHistory = ModelState.PromptHistory;
    FStructuredChatHistory Chat;


    while (!WorkingHistory.IsEmpty())
    {
        FStructuredChatMessage Message = FirstChatMessageInHistory(WorkingHistory, WorkingHistory);

        //Only add proper role messages
        if (Message.Role != EChatTemplateRole::Unknown)
        {
            Chat.History.Add(Message);
        }
    }
    return Chat;
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

//Simple utility functions to find the last sentence
bool ULlamaComponent::IsSentenceEndingPunctuation(const TCHAR Char)
{
    return Char == TEXT('.') || Char == TEXT('!') || Char == TEXT('?');
}

FString ULlamaComponent::GetLastSentence(const FString& InputString)
{
    int32 LastPunctuationIndex = INDEX_NONE;
    int32 PrecedingPunctuationIndex = INDEX_NONE;

    // Find the last sentence-ending punctuation
    for (int32 i = InputString.Len() - 1; i >= 0; --i)
    {
        if (IsSentenceEndingPunctuation(InputString[i]))
        {
            LastPunctuationIndex = i;
            break;
        }
    }

    // If no punctuation found, return the entire string
    if (LastPunctuationIndex == INDEX_NONE)
    {
        return InputString;
    }

    // Find the preceding sentence-ending punctuation
    for (int32 i = LastPunctuationIndex - 1; i >= 0; --i)
    {
        if (IsSentenceEndingPunctuation(InputString[i]))
        {
            PrecedingPunctuationIndex = i;
            break;
        }
    }

    // Extract the last sentence
    int32 StartIndex = PrecedingPunctuationIndex == INDEX_NONE ? 0 : PrecedingPunctuationIndex + 1;
    return InputString.Mid(StartIndex, LastPunctuationIndex - StartIndex + 1).TrimStartAndEnd();
}

